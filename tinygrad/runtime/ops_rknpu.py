from __future__ import annotations
import ctypes, functools, mmap, queue, threading
from tinygrad.helpers import to_mv, mv_address, Target
from tinygrad.device import BufferSpec
from tinygrad.dtype import dtypes, PtrDType, DType
from tinygrad.uop.ops import Ops, UOp, GroupOp
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HCQArgsState, MMIOInterface
from tinygrad.runtime.ops_cpu import CPUSignal, CPUWorker, CPUComputeQueue, CPUProgram
from tinygrad.renderer.cstyle import ClangJITRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler
from tinygrad.runtime.support.elf import jit_loader

# *** ctypes bindings for libhack.so ***

_LIB_PATH = "/home/alexey/src/hack/build/libhack.so"

try:
  _lib = ctypes.CDLL(_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
except OSError as e:
  raise RuntimeError(f"Failed to load libhack.so from {_LIB_PATH}: {e}") from e

# int npu_open()
_lib.npu_open.restype = ctypes.c_int
_lib.npu_open.argtypes = []

# int npu_close(int fd)
_lib.npu_close.restype = ctypes.c_int
_lib.npu_close.argtypes = [ctypes.c_int]

# int npu_reset(int fd)
_lib.npu_reset.restype = ctypes.c_int
_lib.npu_reset.argtypes = [ctypes.c_int]

# void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint32_t *handle)
_lib.mem_allocate.restype = ctypes.c_void_p
_lib.mem_allocate.argtypes = [
  ctypes.c_int,           # fd
  ctypes.c_size_t,        # size
  ctypes.POINTER(ctypes.c_uint64),  # dma_addr
  ctypes.POINTER(ctypes.c_uint64),  # obj
  ctypes.c_uint32,        # flags
  ctypes.POINTER(ctypes.c_uint32),  # handle
]

# void npu_mul/add/sub(int fd, uint64_t dst_dma, uint64_t dst_obj, uint64_t srcA_dma, uint64_t srcB_dma, int elements)
for _fn in ['npu_mul', 'npu_add', 'npu_sub']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr)
_lib.mem_destroy.restype = None
_lib.mem_destroy.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64]


def _mem_allocate(fd: int, size: int, flags: int = 0):
  """Allocate DMA memory via rk3588-npu. Returns (va_addr, dma_addr, obj_addr, handle)."""
  dma_addr = ctypes.c_uint64(0)
  obj_addr = ctypes.c_uint64(0)
  handle   = ctypes.c_uint32(0)
  va = _lib.mem_allocate(fd, size, ctypes.byref(dma_addr), ctypes.byref(obj_addr), flags, ctypes.byref(handle))
  if va is None or va == 0:
    raise MemoryError(f"mem_allocate failed for size={size}")
  return int(va), int(dma_addr.value), int(obj_addr.value), int(handle.value)


def _mem_destroy(fd: int, handle: int, obj_addr: int):
  """Free DMA memory allocated by mem_allocate."""
  _lib.mem_destroy(fd, handle, obj_addr)


# *** RKNPU Allocator ***

class RKNPUAllocator(HCQAllocator):
  def __init__(self, dev: RKNPUDevice):
    super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)

  def _alloc(self, size: int, options: BufferSpec) -> HCQBuffer:
    va, dma_addr, obj_addr, handle = _mem_allocate(self.dev.fd, size)
    view = MMIOInterface(va, size, fmt='B')
    return HCQBuffer(va_addr=va, size=size, meta=(handle, obj_addr, dma_addr, size), view=view, owner=self.dev)

  def _do_free(self, buf: HCQBuffer, options: BufferSpec | None = None):
    handle, obj_addr, _dma_addr, size = buf.meta
    # munmap the userspace mapping before destroying the kernel object
    ctypes.cdll.LoadLibrary("libc.so.6").munmap(ctypes.c_void_p(buf.va_addr), ctypes.c_size_t(size))
    _mem_destroy(self.dev.fd, handle, obj_addr)

  def _as_buffer(self, src: HCQBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.va_addr, src.size)

  def _map(self, buf: HCQBuffer): return None  # unified address space, no extra mapping needed


# *** RKNPU Renderer ***

# NPU DPU-supported element-wise operations (extend as hardware support grows)
_NPU_SUPPORTED_OPS = {Ops.MUL, Ops.ADD}

# Map from set of ALU ops to NPU function name
_NPU_OP_NAMES = {
  frozenset({Ops.MUL}): "npu_mul",
  frozenset({Ops.ADD}): "npu_add",
}

def _classify_npu(uops: list[UOp]) -> dict | None:
  """Analyze linearized UOps to determine if the kernel is NPU-accelerable.
     Returns operation descriptor dict or None if not supported."""
  params = [u for u in uops if u.op == Ops.PARAM]
  if not params:
    return None

  # Collect float ALU ops (ignore integer index arithmetic)
  compute_ops = {u.op for u in uops if u.op in GroupOp.ALU and u.dtype.scalar() == dtypes.half}

  # All float computation must be NPU-supported
  if not compute_ops or not compute_ops.issubset(_NPU_SUPPORTED_OPS):
    return None

  # All buffers must be fp16 pointers
  if not all(isinstance(u.dtype, PtrDType) and u.dtype.base == dtypes.half for u in params):
    return None

  # Reject if computation involves fp16 constants (e.g. x * 2.0) — NPU does buffer-to-buffer only
  if any(u.op == Ops.CONST and u.dtype.scalar() == dtypes.half for u in uops):
    return None

  # Validate buffer count: count unique LOADs (input buffers) + 1 output == total params
  n_loads = sum(1 for u in uops if u.op == Ops.LOAD and u.dtype.scalar() == dtypes.half)
  n_stores = sum(1 for u in uops if u.op == Ops.STORE)
  if n_stores != 1 or len(params) != n_loads + 1:
    return None

  fn = _NPU_OP_NAMES.get(frozenset(compute_ops))
  if fn is None:
    return None

  return {"fn": fn, "n_elements": params[0].dtype.size, "n_bufs": len(params)}


class RkCompiler(ClangJITCompiler):
  def compile(self, src:str) -> bytes:
    return self.compile_to_obj(src)

class RkRenderer(ClangJITRenderer):
  def __init__(self, target: Target):
    super().__init__(target)
    self.compiler = RkCompiler()

  def render(self, uops: list[UOp]) -> str:
    npu = _classify_npu(uops)
    if npu is not None:
      # NPU-accelerable: render an NPU kernel body that calls npu_mul/add/sub directly
      name, _kernel, bufs = self._render(uops)
      names = [n for n, _ in bufs]
      n = str(npu["n_elements"])

      # Generate NPU dispatch body using fd + DMA args
      body = [f"  {npu['fn']}(npu_fd, dma_0, obj_0, dma_1, dma_2, {n});"]
      for i in range(3, len(names)):
        body.append(f"  {npu['fn']}(npu_fd, dma_0, obj_0, dma_0, dma_{i}, {n});")
      return self.render_kernel(name, body, bufs, uops)

    return super().render(uops)

  def _render_defines(self, uops) -> list[str]:
    defines = super()._render_defines(uops)
    defines += [
      'void npu_mul(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_add(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_sub(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
    ]
    return defines

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    defines = '\n'.join(self._render_defines(uops))

    # Build outer function parameter list: original params + dma_i/obj_i for each pointer buffer
    ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
    ex_params = []
    for name, (dtype, mutable) in bufs:
      if isinstance(dtype, PtrDType):
        ex_params.append(f"{self.render_dtype(dtype, mutable)}{self.buffer_suffix} {name}")
      else:
        ex_params.append(f"int {name}")
    for i in ptr_indices:
      ex_params.append(f"unsigned long long dma_{i}")
      ex_params.append(f"unsigned long long obj_{i}")

    # Add npu_fd as the last parameter (device fd, passed from queue)
    ex_params.append("int npu_fd")

    # Check if the kernel body uses DMA args (i.e. NPU kernel)
    uses_dma = any(f"dma_{i}" in line for line in kernel for i in ptr_indices)

    if uses_dma:
      # NPU kernel: body uses DMA/fd args, emit directly in the outer function
      body = '\n'.join(kernel)
      return f"{defines}\nvoid {function_name}({', '.join(ex_params)}) {{\n{body}\n}}"
    else:
      # CPU kernel: render inner function, wrap with outer that forwards VA args
      inner_body = self._render_body(function_name, kernel, bufs, uops, prefix)
      inner_body = inner_body.replace(f"void {function_name}(", f"static void {function_name}_inner(", 1)
      inner_args = ', '.join([name for name, _ in bufs])
      entry = f"void {function_name}({', '.join(ex_params)}) {{\n  {function_name}_inner({inner_args});\n}}"
      return f"{defines}\n{inner_body}\n{entry}"


# *** RKNPU Program & Queue ***

class RKNPUProgram(CPUProgram):
  def __init__(self, dev, name:str, lib:bytes, runtimevars:dict[str, int]|None=None, **kwargs):
    # Always relocate and JIT — unified code path for both NPU and CPU kernels
    lib = jit_loader(lib, base=0, link_libs=['m', ''])
    super().__init__(dev, name, lib, runtimevars, **kwargs)

class RKNPUComputeQueue(CPUComputeQueue):
  def _rknpu_exec(self, tid, prg, dev_fd, bufs, n_dma, *args):
    """Execute kernel with VA pointers, DMA/OBJ addresses, and device fd."""
    import platform
    va_args = list(map(ctypes.c_uint64, args[:bufs]))
    dma_args = list(map(ctypes.c_uint64, args[bufs:bufs+n_dma]))
    vals = list(map(ctypes.c_int64 if platform.machine() == "arm64" else ctypes.c_int32, args[bufs+n_dma:]))
    prg.fxn(*va_args, *dma_args, *vals, ctypes.c_int(dev_fd))

  def exec(self, prg, args_state:HCQArgsState, global_size, local_size):
    # Extract DMA/OBJ from HCQBuffer.meta for each pointer buffer
    dma_args = []
    for buf in args_state.bufs:
      _handle, obj_addr, dma_addr, _size = buf.meta
      dma_args.extend([dma_addr, obj_addr])
    dev_fd = args_state.bufs[0].owner.fd
    return self.cmd(self._rknpu_exec, prg, dev_fd, len(args_state.bufs), len(dma_args),
                    *[x.va_addr for x in args_state.bufs], *dma_args, *args_state.vals)


# *** RKNPU Device ***

class RKNPUDevice(HCQCompiled):
  def __init__(self, device: str = ""):
    self.fd: int = _lib.npu_open()
    if self.fd < 0:
      raise RuntimeError(f"npu_open() failed, fd={self.fd}. Is /dev/dri/card1 accessible?")

    self.tasks: queue.Queue = queue.Queue()
    CPUWorker(self, self.tasks, thread_id=0).start()

    super().__init__(
      device,
      RKNPUAllocator(self),
      [RkRenderer, ClangJITRenderer, CPULLVMRenderer],
      functools.partial(RKNPUProgram, self),
      CPUSignal,
      RKNPUComputeQueue,
    )

  def finalize(self):
    super().finalize()
    _lib.npu_close(self.fd)
