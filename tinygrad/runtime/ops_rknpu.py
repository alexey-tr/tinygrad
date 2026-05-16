from __future__ import annotations
import ctypes, functools, mmap, queue, threading, math
from tinygrad.helpers import to_mv, from_mv, mv_address, cpu_profile, Target
from tinygrad.device import BufferSpec
from tinygrad.dtype import dtypes, PtrDType, DType
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HCQArgsState, MMIOInterface
from tinygrad.runtime.ops_cpu import CPUSignal, CPUWorker, CPUComputeQueue, CPUProgram, _in_worker_task
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

# void npu_mul/add/sub_scalar(int fd, uint64_t dst_dma, uint64_t dst_obj, uint64_t src_dma, _Float16 scalar, int elements)
for _fn in ['npu_mul_scalar', 'npu_add_scalar', 'npu_sub_scalar']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint16, ctypes.c_int]

# void npu_neg(int fd, uint64_t dst_dma, uint64_t dst_obj, uint64_t src_dma, int elements)
_lib.npu_neg.restype = None
_lib.npu_neg.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

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


libc = ctypes.CDLL("libc.so.6")
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

def _mem_destroy(fd: int, handle: int, obj_addr: int):
  """Free DMA memory allocated by mem_allocate."""
  _lib.mem_destroy(fd, handle, obj_addr)


# *** RKNPU Allocator ***

class RKNPUAllocator(HCQAllocator):
  def __init__(self, dev: RKNPUDevice):
    super().__init__(dev, supports_copy_from_disk=False, supports_transfer=True)

  def _alloc(self, size: int, options: BufferSpec) -> HCQBuffer:
    # rknpu driver requires page-aligned size for mmap when using NON_CONTIGUOUS
    aligned_size = (size + 4095) & ~4095
    # RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_IOMMU | RKNPU_MEM_WRITE_COMBINE = 1 | 16 | 4 = 21
    va, dma_addr, obj_addr, handle = _mem_allocate(self.dev.fd, aligned_size, flags=21)
    view = MMIOInterface(va, size, fmt='B')
    return HCQBuffer(va_addr=va, size=size, meta=(handle, obj_addr, dma_addr, aligned_size), view=view, owner=self.dev)

  def _do_free(self, buf: HCQBuffer, options: BufferSpec | None = None):
    handle, obj_addr, _dma_addr, aligned_size = buf.meta
    # munmap the userspace mapping before destroying the kernel object
    ctypes.cdll.LoadLibrary("libc.so.6").munmap(ctypes.c_void_p(buf.va_addr), ctypes.c_size_t(aligned_size))
    _mem_destroy(self.dev.fd, handle, obj_addr)

  def _as_buffer(self, src: HCQBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.va_addr, src.size)

  # Override _copyin/_copyout to use direct memmove. The base HCQAllocator would use hw_copy_queue_t
  # (RKNPUCopyQueue), which runs npu_add_scalar and treats all data as fp16 — corrupting non-fp16 buffers.
  # RKNPU memory is CPU-accessible (unified address space), so memmove works directly.
  def _copyin(self, dest: HCQBuffer, src: memoryview):
    self.dev.synchronize()
    with cpu_profile(f'TINY -> {self.dev.device}', f"{self.dev.device}:COPY"): ctypes.memmove(int(dest.va_addr), from_mv(src), len(src))

  def _copyout(self, dest: memoryview, src: HCQBuffer):
    self.dev.synchronize()
    with cpu_profile(f'{self.dev.device} -> TINY', f"{self.dev.device}:COPY"): ctypes.memmove(from_mv(dest), int(src.va_addr), len(dest))

  def _map(self, buf: HCQBuffer): return None  # unified address space, no extra mapping needed


# *** RKNPU Renderer ***

# NPU function names for supported element-wise ops
_NPU_FN = {Ops.MUL: "npu_mul", Ops.ADD: "npu_add", Ops.SUB: "npu_sub"}
# Scalar variants (one operand is a fp16 constant)
_NPU_FN_SCALAR = {Ops.MUL: "npu_mul_scalar", Ops.ADD: "npu_add_scalar", Ops.SUB: "npu_sub_scalar"}

# Post-devectorization shape: GEP(LOAD(CAST(INDEX(PARAM, ...))))
_param_gep = UPat(Ops.GEP, src=(UPat(Ops.LOAD, src=(UPat(Ops.CAST, src=(UPat(Ops.INDEX, src=(UPat(Ops.PARAM), UPat())),)),)),))
_const_fp16 = UPat(Ops.CONST, dtype=dtypes.half, name="c")

# Pre-matcher: tag fp16 ALU ops whose operands trace to PARAM loads.
rknpu_pm = PatternMatcher([
  # vector OP vector
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB), dtype=dtypes.half, name="u", src=(_param_gep, _param_gep)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_FN[u.op])),
  # vector OP scalar  (note: `scalar - vector` is canonicalized by tinygrad to `v*(-1)+scalar`)
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB), dtype=dtypes.half, name="u", src=(_param_gep, _const_fp16)),
   lambda u, c: UOp(Ops.CUSTOM, u.dtype, (u.src[0], c), _NPU_FN_SCALAR[u.op])),
  # unary negate
  (UPat(Ops.NEG, dtype=dtypes.half, name="u", src=(_param_gep,)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, "npu_neg")),
])


class RkCompiler(ClangJITCompiler):
  def compile(self, src:str) -> bytes:
    return self.compile_to_obj(src)

class RkRenderer(ClangJITRenderer):
  pre_matcher = rknpu_pm

  string_rewrite = PatternMatcher([
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"({ctx[x.src[0]]} * {ctx[x.src[1]]})" if x.arg in ("npu_mul", "npu_mul_scalar") else None),
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"({ctx[x.src[0]]} + {ctx[x.src[1]]})" if x.arg in ("npu_add", "npu_add_scalar") else None),
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"({ctx[x.src[0]]} - {ctx[x.src[1]]})" if x.arg in ("npu_sub", "npu_sub_scalar") else None),
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"(-{ctx[x.src[0]]})" if x.arg == "npu_neg" else None),
  ]) + ClangJITRenderer.string_rewrite

  def __init__(self, target: Target):
    super().__init__(target)
    self.compiler = RkCompiler()

  def render(self, uops: list[UOp]) -> str:
    # rknpu_pm rewrites eligible fp16 ALU ops to CUSTOM nodes tagged with the NPU fn name.
    # If any such node survived to render time, this is an NPU-accelerable kernel.
    _all_npu_fns = set(_NPU_FN.values()) | set(_NPU_FN_SCALAR.values()) | {"npu_neg"}
    npu_ops = [u for u in uops if u.op is Ops.CUSTOM and u.arg in _all_npu_fns]
    has_loops = any(u.op is Ops.RANGE for u in uops)
    if npu_ops and not has_loops:
      fn = npu_ops[0].arg
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      params = [u for u in uops if u.op is Ops.PARAM]
      n = str(params[0].dtype.size)

      body = []
      if fn == "npu_neg":
        # unary: npu_neg(fd, dst_dma, dst_obj, src_dma, n)
        body.append(f"  npu_neg(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{ptr_indices[1]}, {n});")
      elif fn in _NPU_FN_SCALAR.values():
        # scalar: second src is a CONST — embed literal; only one buffer besides dst
        # The CUSTOM node src[0]=param_gep, src[1]=CONST
        # bufs[0]=dst, bufs[1]=src; scalar value comes from the CONST UOp
        const_ops = [u for u in uops if u.op is Ops.CONST and u.dtype == dtypes.half]
        # Emit scalar as a __fp16 float literal — the compiler handles the conversion correctly
        scalar_f = float(const_ops[0].arg) if const_ops else 0.0
        if math.isinf(scalar_f): scalar_str = "-__builtin_inff()" if scalar_f < 0 else "__builtin_inff()"
        elif math.isnan(scalar_f): scalar_str = '__builtin_nanf("")'
        else: scalar_str = f"{scalar_f!r}f"
        body.append(f"  {fn}(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{ptr_indices[1]}, (__fp16){scalar_str}, {n});")
      else:
        # vector OP vector
        body.append(f"  {fn}(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{ptr_indices[1]}, dma_{ptr_indices[2]}, {n});")
        for i in ptr_indices[3:]:
          body.append(f"  {fn}(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{ptr_indices[0]}, dma_{i}, {n});")
      return self.render_kernel(name, body, bufs, uops)

    return super().render(uops)

  def _render_defines(self, uops) -> list[str]:
    defines = super()._render_defines(uops)
    defines += [
      'void npu_mul(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_add(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_sub(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_mul_scalar(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __fp16 scalar, int elements);',
      'void npu_add_scalar(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __fp16 scalar, int elements);',
      'void npu_sub_scalar(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __fp16 scalar, int elements);',
      'void npu_neg(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, int elements);',
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
      inner_body = inner_body.replace(f"void {function_name}(", f"static void cpu_{function_name}(", 1)
      inner_args = ', '.join([name for name, _ in bufs])
      entry = f"void {function_name}({', '.join(ex_params)}) {{\n  cpu_{function_name}({inner_args});\n}}"
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
    vals = list(args[bufs+n_dma:])
    if 'core_id' in prg.runtimevars: vals[prg.runtimevars['core_id']] = tid
    vals_mapped = list(map(ctypes.c_int64 if platform.machine() == "arm64" else ctypes.c_int32, vals))
    prg.fxn(*va_args, *vals_mapped, *dma_args, ctypes.c_int(dev_fd))

  def exec(self, prg, args_state:HCQArgsState, global_size, local_size):
    # Extract DMA/OBJ from HCQBuffer.meta for each pointer buffer
    dma_args = []
    for buf in args_state.bufs:
      if buf.meta is not None:
        _handle, obj_addr, dma_addr, _size = buf.meta
      else:
        obj_addr, dma_addr = 0, 0
      dma_args.extend([dma_addr, obj_addr])
    dev_fd = prg.dev.fd
    return self.cmd(self._rknpu_exec, prg, dev_fd, len(args_state.bufs), len(dma_args),
                    *[x.va_addr for x in args_state.bufs], *dma_args, *args_state.vals, threads=(global_size or (1,))[0])

# Minimum byte size to use NPU for copy; below this threshold, memmove is faster
_COPY_NPU_MIN_BYTES = 4096

class RKNPUCopyQueue(CPUComputeQueue):
  """Copy queue that uses npu_add_scalar(src, 0, n) as a hardware-accelerated buffer copy for fp16 data."""
  def __init__(self, **kwargs):
    super().__init__()  # HWQueue doesn't accept kwargs; ignore queue_idx from HCQGraph
  def _npu_copy(self, tid, fd, dst_va, src_va, dst_dma, dst_obj, src_dma, nbytes, is_fp16):
    elements = nbytes // 2  # fp16 elements
    # NPU copy only works for fp16 data; all RKNPU buffers share one fd so no cross-fd concern.
    # Using npu_add_scalar on non-fp16 data corrupts it (DPU is hardwired for fp16 precision).
    can_npu = is_fp16 and dst_dma != 0 and src_dma != 0 and fd != 0 and elements > 0 and nbytes >= _COPY_NPU_MIN_BYTES
    if can_npu:
      # Hardware copy: dst = src + 0  (identity via NPU pipeline)
      _lib.npu_add_scalar(fd, dst_dma, dst_obj, src_dma, 0, elements)
      # Handle trailing odd byte with CPU fallback
      if nbytes % 2:
        ctypes.memmove(dst_va + nbytes - 1, src_va + nbytes - 1, 1)
    else:
      # Fallback: CPU memmove (cross-device, small copy, non-fp16, or non-RKNPU buffer)
      ctypes.memmove(dst_va, src_va, nbytes)

  def copy(self, dest: HCQBuffer, src: HCQBuffer, copy_size: int):
    # Extract DMA metadata from both buffers.
    # Source may be a non-RKNPU buffer (e.g. CPU) where meta is not a 4-tuple — fall back to memmove.
    if isinstance(dest.meta, tuple) and len(dest.meta) == 4:
      _dh, dst_obj, dst_dma, _ = dest.meta
    else:
      dst_obj, dst_dma = 0, 0
    if isinstance(src.meta, tuple) and len(src.meta) == 4:
      _sh, _sobj, src_dma, _ = src.meta
    else:
      src_dma = 0
    # All RKNPU device instances share one fd, so any RKNPU buffer is valid for NPU submission.
    fd = dest.owner.fd if (dest.owner and hasattr(dest.owner, 'fd')) else 0
    # Only use NPU path for fp16 buffers — the DPU pipeline is hardwired for fp16 precision.
    from tinygrad.dtype import dtypes
    is_fp16 = dest.dtype == dtypes.half if hasattr(dest, 'dtype') and dest.dtype is not None else False
    return self.cmd(self._npu_copy, fd, dest.va_addr, src.va_addr, dst_dma, dst_obj, src_dma, copy_size, is_fp16)

  def _submit(self, dev):
    # Submit to the device's copy_tasks queue (separate worker thread) to avoid deadlock
    # with the compute queue which runs on dev.tasks
    dev.copy_tasks.put(self._q[:])


# *** RKNPU Device ***

class RKNPUSignal(CPUSignal):
  def _sleep(self, time_spent_since_last_sleep_ms: int):
    if self.is_timeline and self.owner is not None and not getattr(_in_worker_task, 'active', False):
      # Also drain copy_tasks: HCQGraph submits copies to a separate worker thread, so a compute
      # kernel could start before its copy dependency finishes without this join.
      self.owner.tasks.join()
      self.owner.copy_tasks.join()
      if self.owner.error_state is not None: raise self.owner.error_state


class RKNPUDevice(HCQCompiled):
  _shared_fd: int = -1
  _fd_refcount: int = 0
  _fd_lock: threading.Lock = threading.Lock()

  def __init__(self, device: str = ""):
    with RKNPUDevice._fd_lock:
      if RKNPUDevice._shared_fd < 0:
        RKNPUDevice._shared_fd = _lib.npu_open()
        if RKNPUDevice._shared_fd < 0:
          raise RuntimeError(f"npu_open() failed, fd={RKNPUDevice._shared_fd}. Is /dev/dri/card1 accessible?")
      RKNPUDevice._fd_refcount += 1
      self.fd: int = RKNPUDevice._shared_fd

    # Compute queue worker
    self.tasks: queue.Queue = queue.Queue()
    CPUWorker(self, self.tasks, thread_id=0).start()

    # Copy queue worker (separate thread to avoid deadlock with compute queue in HCQGraph)
    self.copy_tasks: queue.Queue = queue.Queue()
    CPUWorker(self, self.copy_tasks, thread_id=0).start()

    super().__init__(
      device,
      RKNPUAllocator(self),
      [RkRenderer, ClangJITRenderer, CPULLVMRenderer],
      functools.partial(RKNPUProgram, self),
      RKNPUSignal,
      RKNPUComputeQueue,
      RKNPUCopyQueue,
    )

  def finalize(self):
    super().finalize()
    with RKNPUDevice._fd_lock:
      RKNPUDevice._fd_refcount -= 1
      fd_to_close = RKNPUDevice._shared_fd if RKNPUDevice._fd_refcount == 0 else -1
      if fd_to_close >= 0:
        RKNPUDevice._shared_fd = -1
    if fd_to_close >= 0:
      _lib.npu_close(fd_to_close)
