from __future__ import annotations
import ctypes, functools, re, mmap, queue, threading
from tinygrad.helpers import to_mv, mv_address, Target
from tinygrad.device import BufferSpec
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.uop.ops import Ops, UOp, GroupOp
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HCQArgsState, HCQProgram, MMIOInterface
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

# void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr)
_lib.mem_destroy.restype = None
_lib.mem_destroy.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64]

# void npu_mul/add/sub(uint64_t dst_dma, uint64_t dst_obj, uint64_t srcA_dma, uint64_t srcB_dma, int elements)
for _fn in ['npu_mul', 'npu_add', 'npu_sub']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

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
  # LOADs are UPCAST-invariant: each input buffer produces one LOAD per loop iteration regardless of vector width
  n_loads = sum(1 for u in uops if u.op == Ops.LOAD and u.dtype.scalar() == dtypes.half)
  n_stores = sum(1 for u in uops if u.op == Ops.STORE)
  # For pure element-wise: exactly 1 store (output), N loads (inputs), N-1 MULs
  if n_stores != 1 or len(params) != n_loads + 1:
    return None

  fn = _NPU_OP_NAMES.get(frozenset(compute_ops))
  if fn is None:
    return None

  return {"fn": fn, "n_elements": params[0].dtype.size, "n_bufs": len(params)}


# NPU kernel marker prefix used to bypass JIT compilation
_NPU_MARKER = b"NPU:"

class RkCompiler(ClangJITCompiler):
  def compile(self, src:str) -> bytes:
    # NPU kernels are encoded as metadata, not compiled C
    if src.startswith("NPU:"):
      return src.encode()
    return self.compile_to_obj(src)

class RkRenderer(ClangJITRenderer):
  def __init__(self, target: Target):
    super().__init__(target)
    self.compiler = RkCompiler()

  def render(self, uops: list[UOp]) -> str:
    npu = _classify_npu(uops)
    if npu is None:
      return super().render(uops)

    # NPU-accelerable: encode dispatch info as metadata string (not C code)
    # Format: "NPU:<fn_name>:<n_elements>:<n_bufs>"
    return f"NPU:{npu['fn']}:{npu['n_elements']}:{npu['n_bufs']}"

  def _render_defines(self, uops) -> list[str]:
    defines = super()._render_defines(uops)
    defines += ['#include "/home/alexey/src/hack/rk3588-npu/include/npu_hw.h"']
    return defines

# *** RKNPU Program & Queue ***

class RKNPUProgram(CPUProgram):
  def __init__(self, dev, name:str, lib:bytes, runtimevars:dict[str, int]|None=None, **kwargs):
    self.npu_info = None
    if lib.startswith(_NPU_MARKER):
      # Parse NPU metadata: "NPU:<fn>:<n_elements>:<n_bufs>"
      parts = lib.decode().split(":")
      self.npu_info = {"fn": parts[1], "n_elements": int(parts[2]), "n_bufs": int(parts[3])}
      self.fxn = None  # no JIT function needed
      # Initialize HCQProgram directly, skipping CPUProgram's JIT loader
      HCQProgram.__init__(self, HCQArgsState, dev, name, kernargs_alloc_size=0)
    else:
      # Regular C kernel: relocate and load via CPUProgram
      lib = jit_loader(lib, base=0, link_libs=['m', ''])
      super().__init__(dev, name, lib, runtimevars, **kwargs)

class RKNPUComputeQueue(CPUComputeQueue):
  def _npu_exec(self, tid, fn_name, n_elements, *dma_args):
    """Execute NPU dispatch directly with DMA addresses from HCQBuffer.meta."""
    # dma_args layout: [dst_dma, dst_obj, srcA_dma, srcA_obj, srcB_dma, srcB_obj, ...]
    dst_dma, dst_obj = dma_args[0], dma_args[1]
    inputs_dma = [dma_args[i] for i in range(2, len(dma_args), 2)]  # skip obj for inputs

    fn = getattr(_lib, fn_name)
    # First op: fn(dst_dma, dst_obj, in[0]_dma, in[1]_dma, n_elements)
    fn(dst_dma, dst_obj, inputs_dma[0], inputs_dma[1], n_elements)
    # Chain remaining inputs: fn(dst_dma, dst_obj, dst_dma, in[i]_dma, n_elements)
    for i in range(2, len(inputs_dma)):
      fn(dst_dma, dst_obj, dst_dma, inputs_dma[i], n_elements)

  def exec(self, prg, args_state:HCQArgsState, global_size, local_size):
    if isinstance(prg, RKNPUProgram) and prg.npu_info is not None:
      npu = prg.npu_info
      # Extract DMA/OBJ from HCQBuffer.meta: (handle, obj_addr, dma_addr, size)
      dma_args = []
      for buf in args_state.bufs:
        _handle, obj_addr, dma_addr, _size = buf.meta
        dma_args.extend([dma_addr, obj_addr])
      return self.cmd(self._npu_exec, npu["fn"], npu["n_elements"], *dma_args)
    return super().exec(prg, args_state, global_size, local_size)

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
