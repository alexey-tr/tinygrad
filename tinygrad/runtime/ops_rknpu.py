from __future__ import annotations
import ctypes, functools, mmap, queue, threading, math, re

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
def _strip_ansi(s: str) -> str: return _ANSI_RE.sub('', s)  # kernel display names are ANSI-colored; match on plain
from tinygrad.helpers import to_mv, from_mv, mv_address, cpu_profile, Target
from tinygrad.device import BufferSpec
from tinygrad.dtype import dtypes, PtrDType, DType
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, AxisType
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

# void npu_drain(int fd) — synchronizes the async submission pipeline
_lib.npu_drain.restype = None
_lib.npu_drain.argtypes = [ctypes.c_int]

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

# void npu_mul/add/sub/max(int fd, uint64_t dst_dma, uint64_t dst_obj, uint64_t srcA_dma, uint64_t srcB_dma, int elements)
for _fn in ['npu_mul', 'npu_add', 'npu_sub', 'npu_max']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# void npu_mul/add/sub/max_scalar(int fd, uint64_t dst_dma, uint64_t dst_obj, uint64_t src_dma, _Float16 scalar, int elements)
for _fn in ['npu_mul_scalar', 'npu_add_scalar', 'npu_sub_scalar', 'npu_max_scalar']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint16, ctypes.c_int]

# void npu_neg(int fd, uint64_t dst_dma, uint64_t dst_obj, uint64_t src_dma, int elements)
_lib.npu_neg.restype = None
_lib.npu_neg.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# fp32 vector-vector (bindings only; ERDMA 32-bit limitation means these produce incorrect results)
for _fn in ['npu_mul_f32', 'npu_add_f32', 'npu_sub_f32']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# fp32 scalar and unary ops (these work correctly)
for _fn in ['npu_mul_scalar_f32', 'npu_add_scalar_f32', 'npu_sub_scalar_f32']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_float, ctypes.c_int]

_lib.npu_neg_f32.restype = None
_lib.npu_neg_f32.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# int8 vector-vector and unary
for _fn in ['npu_mul_i8', 'npu_add_i8', 'npu_sub_i8', 'npu_max_i8']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

_lib.npu_neg_i8.restype = None
_lib.npu_neg_i8.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# int16 vector-vector, scalar, and unary
for _fn in ['npu_mul_i16', 'npu_add_i16', 'npu_sub_i16', 'npu_max_i16']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

for _fn in ['npu_mul_scalar_i16', 'npu_add_scalar_i16', 'npu_sub_scalar_i16', 'npu_max_scalar_i16']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int16, ctypes.c_int]

_lib.npu_neg_i16.restype = None
_lib.npu_neg_i16.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# bf16 vector-vector, scalar, and unary (scalar binding uses c_uint16 — bit-compatible with __bf16, see fp16 above)
for _fn in ['npu_mul_bf16', 'npu_add_bf16', 'npu_sub_bf16', 'npu_max_bf16']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

for _fn in ['npu_mul_scalar_bf16', 'npu_add_scalar_bf16', 'npu_sub_scalar_bf16', 'npu_max_scalar_bf16']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint16, ctypes.c_int]

_lib.npu_neg_bf16.restype = None
_lib.npu_neg_bf16.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# matmul: void npu_matmul_<dtype>(int fd, void* dst_va, u64 dst_dma, u64 dst_obj,
#                                  const void* a_va, u64 a_dma,
#                                  const void* b_va, u64 b_dma,
#                                  int M, int K, int N, int relu, float bias, const float *pcbias)
# `relu` (0/1) fuses max(0,x); `bias` (scalar) fuses x+bias; `pcbias` (host fp32 array[N] or
# NULL) fuses the per-channel bias x+bias[n] — all in the BS stage, same submit. relu AND
# per-channel bias are auto-dispatched (see _matmul_epilogue / _try_match_matmul); scalar `bias`
# is ctypes-only (render passes 0.0) — a const-add epilogue can't be reliably told from the
# matmul's own ADDs. pcbias is fp16-only in the runtime (bf16/int8 wrappers take but ignore it).
for _fn in ['npu_matmul_fp16', 'npu_matmul_bf16', 'npu_matmul_int8', 'npu_matmul_fp16_bt', 'npu_matmul_int8_bt']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                 ctypes.c_void_p]

# void npu_conv_fp16(int fd, void* dst_va, u64 dst_dma, u64 dst_obj, const void* feat_va, u64 feat_dma,
#                    const void* weight_va, u64 weight_dma, int N,Cin,IH,IW,Cout,KH,KW,sh,sw,fp16_out)
_lib.npu_conv_fp16.restype = None
_lib.npu_conv_fp16.argtypes = [ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                               ctypes.c_void_p, ctypes.c_uint64,
                               ctypes.c_void_p, ctypes.c_uint64,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# void npu_sum_lastaxis_fp16(int fd, void* dst_va, u64 dst_dma, u64 dst_obj,
#                            const void* src_va, u64 src_dma, int M, int K)
# Sum over the contiguous last axis: dst[m] = sum_k src[m*K + k] (rowsum via matmul-by-ones).
_lib.npu_sum_lastaxis_fp16.restype = None
_lib.npu_max_lastaxis_fp16.restype = None
_lib.npu_max_lastaxis_fp16.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                       ctypes.c_void_p, ctypes.c_uint64,
                                       ctypes.c_int, ctypes.c_int]
_lib.npu_sum_lastaxis_fp16.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                       ctypes.c_void_p, ctypes.c_uint64,
                                       ctypes.c_int, ctypes.c_int]

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


# DRM_IOCTL_RKNPU_MEM_SYNC = DRM_IOWR(DRM_COMMAND_BASE+0x05, struct rknpu_mem_sync[32 bytes])
_RKNPU_MEM_SYNC = 0xC0206445
_MEM_SYNC_TO_DEVICE   = 1   # flush CPU cache -> DRAM (before NPU reads CPU-written input)
_MEM_SYNC_FROM_DEVICE = 2   # invalidate CPU cache (before CPU reads NPU-written output)
class _rknpu_mem_sync(ctypes.Structure):
  _fields_ = [("flags", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
              ("obj_addr", ctypes.c_uint64), ("offset", ctypes.c_uint64), ("size", ctypes.c_uint64)]

def _mem_sync(fd: int, obj_addr: int, size: int, flags: int):
  """Cache maintenance for a CACHEABLE DMA buffer. No-op-safe: only call on cacheable allocs
  (returns EINVAL on write-combine). Required because the NPU DMAs around the CPU cache."""
  import fcntl
  fcntl.ioctl(fd, _RKNPU_MEM_SYNC, _rknpu_mem_sync(flags=flags, obj_addr=obj_addr, offset=0, size=size))


# *** RKNPU Allocator ***

class RKNPUAllocator(HCQAllocator):
  def __init__(self, dev: RKNPUDevice):
    super().__init__(dev, supports_copy_from_disk=False, supports_transfer=True)

  def _alloc(self, size: int, options: BufferSpec) -> HCQBuffer:
    # rknpu driver requires page-aligned size for mmap when using NON_CONTIGUOUS
    aligned_size = (size + 4095) & ~4095
    # RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE | RKNPU_MEM_IOMMU = 1 | 2 | 16 = 19.
    # CACHEABLE (not WRITE_COMBINE): CPU reads from cached memory are ~9x faster (12 vs 1.4 GB/s),
    # which dominates copy-out. Cost: the NPU DMAs around the CPU cache, so _copyin must flush
    # (TO_DEVICE) and _copyout/_as_buffer must invalidate (FROM_DEVICE) — see _mem_sync.
    va, dma_addr, obj_addr, handle = _mem_allocate(self.dev.fd, aligned_size, flags=19)
    view = MMIOInterface(va, size, fmt='B')
    return HCQBuffer(va_addr=va, size=size, meta=(handle, obj_addr, dma_addr, aligned_size), view=view, owner=self.dev)

  def _do_free(self, buf: HCQBuffer, options: BufferSpec | None = None):
    handle, obj_addr, _dma_addr, aligned_size = buf.meta
    # munmap the userspace mapping before destroying the kernel object
    ctypes.cdll.LoadLibrary("libc.so.6").munmap(ctypes.c_void_p(buf.va_addr), ctypes.c_size_t(aligned_size))
    _mem_destroy(self.dev.fd, handle, obj_addr)

  def _as_buffer(self, src: HCQBuffer) -> memoryview:
    self.dev.synchronize()
    _mem_sync(self.dev.fd, src.meta[1], src.meta[3], _MEM_SYNC_FROM_DEVICE)  # invalidate: NPU wrote DRAM
    return to_mv(src.va_addr, src.size)

  # Override _copyin/_copyout to use direct memmove. The base HCQAllocator would use hw_copy_queue_t
  # (RKNPUCopyQueue), which runs npu_add_scalar and treats all data as fp16 — corrupting non-fp16 buffers.
  # RKNPU memory is CPU-accessible (unified address space), so memmove works directly. Buffers are
  # CACHEABLE (fast CPU reads), so cache maintenance brackets the CPU<->NPU handoff.
  def _copyin(self, dest: HCQBuffer, src: memoryview):
    self.dev.synchronize()
    with cpu_profile(f'TINY -> {self.dev.device}', f"{self.dev.device}:COPY"): ctypes.memmove(int(dest.va_addr), from_mv(src), len(src))
    _mem_sync(self.dev.fd, dest.meta[1], dest.meta[3], _MEM_SYNC_TO_DEVICE)  # flush: NPU reads DRAM

  def _copyout(self, dest: memoryview, src: HCQBuffer):
    self.dev.synchronize()
    _mem_sync(self.dev.fd, src.meta[1], src.meta[3], _MEM_SYNC_FROM_DEVICE)  # invalidate: NPU wrote DRAM
    with cpu_profile(f'{self.dev.device} -> TINY', f"{self.dev.device}:COPY"): ctypes.memmove(from_mv(dest), int(src.va_addr), len(dest))

  def _map(self, buf: HCQBuffer): return None  # unified address space, no extra mapping needed


# *** RKNPU Renderer ***

# NPU function names keyed by (op, dtype)
_NPU_FN = {
  (Ops.MUL, dtypes.half):  "npu_mul",
  (Ops.ADD, dtypes.half):  "npu_add",
  (Ops.SUB, dtypes.half):  "npu_sub",
  (Ops.MAX, dtypes.half):  "npu_max",
  (Ops.MUL, dtypes.int8):  "npu_mul_i8",
  (Ops.ADD, dtypes.int8):  "npu_add_i8",
  (Ops.SUB, dtypes.int8):  "npu_sub_i8",
  (Ops.MAX, dtypes.int8):  "npu_max_i8",
  (Ops.MUL, dtypes.int16): "npu_mul_i16",
  (Ops.ADD, dtypes.int16): "npu_add_i16",
  (Ops.SUB, dtypes.int16): "npu_sub_i16",
  (Ops.MAX, dtypes.int16): "npu_max_i16",
  (Ops.MUL, dtypes.bfloat16): "npu_mul_bf16",
  (Ops.ADD, dtypes.bfloat16): "npu_add_bf16",
  (Ops.SUB, dtypes.bfloat16): "npu_sub_bf16",
  (Ops.MAX, dtypes.bfloat16): "npu_max_bf16",
}
# Scalar variants (one operand is a constant). fp32 vector-vector and fp32 mul-scalar excluded:
# - vector-vector: ERDMA 32-bit limitation
# - mul-scalar:    DPU MUL EW path doesn't handle fp32 EW_OP_VALUE correctly (hangs)
_NPU_FN_SCALAR = {
  (Ops.MUL, dtypes.half):  "npu_mul_scalar",
  (Ops.ADD, dtypes.half):  "npu_add_scalar",
  (Ops.SUB, dtypes.half):  "npu_sub_scalar",
  (Ops.MAX, dtypes.half):  "npu_max_scalar",
  (Ops.ADD, dtypes.float): "npu_add_scalar_f32",
  (Ops.SUB, dtypes.float): "npu_sub_scalar_f32",
  (Ops.MUL, dtypes.int16): "npu_mul_scalar_i16",
  (Ops.ADD, dtypes.int16): "npu_add_scalar_i16",
  (Ops.SUB, dtypes.int16): "npu_sub_scalar_i16",
  (Ops.MAX, dtypes.int16): "npu_max_scalar_i16",
  (Ops.MUL, dtypes.bfloat16): "npu_mul_scalar_bf16",
  (Ops.ADD, dtypes.bfloat16): "npu_add_scalar_bf16",
  (Ops.SUB, dtypes.bfloat16): "npu_sub_scalar_bf16",
  (Ops.MAX, dtypes.bfloat16): "npu_max_scalar_bf16",
}
_NPU_NEG = {dtypes.half: "npu_neg", dtypes.float: "npu_neg_f32", dtypes.int8: "npu_neg_i8", dtypes.int16: "npu_neg_i16",
            dtypes.bfloat16: "npu_neg_bf16"}

# Matmul auto-dispatch, keyed by (input dtype, output dtype). The int8 wrapper accumulates in a
# wider type than its inputs (int8 x int8 -> int32), so output dtype differs from inputs — hence
# matching on the pair, not just the output. tinygrad produces an int32-output kernel for int8
# inputs when the matmul uses acc_dtype=int32 (the full-accumulator path the wrapper implements;
# a plain int8->int8 matmul truncates differently and must NOT map here).
_NPU_MATMUL = {
  (dtypes.half, dtypes.half):  "npu_matmul_fp16",
  (dtypes.int8, dtypes.int32): "npu_matmul_int8",
}
# bf16 matmul (npu_matmul_bf16: bf16 in -> fp32 out) is implemented + validated in the runtime
# and callable directly via ctypes, but is NOT auto-dispatched: tinygrad lowers a bf16 matmul by
# casting the inputs to the fp32 accumulation dtype, so the matmul kernel's input PARAMs arrive as
# fp32 (or ushort via bitcast) — never bfloat16. We can't route an fp32-PARAM matmul to the bf16
# wrapper (it expects 2-byte bf16 inputs) and the NPU has no native fp32 matmul, so bf16 falls back
# to CPU. (The DPU also can't emit bf16 output — no FP32->BF16 downcast bit — hence fp32 output.)

# multiply-class ops that count as a matmul "product" (MULACC is fused multiply-add; absent in some
# tinygrad versions, hence the guarded set).
_MM_MUL_OPS = {Ops.MUL} | ({Ops.MULACC} if hasattr(Ops, 'MULACC') else set())

# Select/compare/transcendental float ops that a plain matmul accumulation NEVER emits (it is
# only MUL/ADD over the K loop). Their presence means an elementwise EPILOGUE was fused onto the
# accumulator. MUL/ADD are deliberately excluded — they ARE the matmul. (Mirrors the _EXTRA
# watchlist in _try_match_reduce, minus MUL.)
_MM_EPILOGUE_OPS = {Ops.SUB, Ops.NEG, Ops.MAX, Ops.WHERE, Ops.RECIPROCAL, Ops.SQRT,
                    Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.CMPLT, Ops.CMPNE}

def _is_zero_const(u): return u.op is Ops.CONST and u.arg == 0

def _where_is_relu(w):
  # ReLU lowers to where(0 < x, x, 0)  (or the symmetric where(x < 0, 0, x)).
  if w.op is not Ops.WHERE or len(w.src) != 3: return False
  cond, tval, fval = w.src
  if cond.op is not Ops.CMPLT or len(cond.src) != 2: return False
  lo, hi = cond.src
  if _is_zero_const(lo) and tval is hi and _is_zero_const(fval): return True   # where(0<x, x, 0)
  if _is_zero_const(hi) and _is_zero_const(tval) and fval is lo: return True   # where(x<0, 0, x)
  return False

def _max_is_relu(m):
  return m.op is Ops.MAX and len(m.src) == 2 and (_is_zero_const(m.src[0]) or _is_zero_const(m.src[1]))

def _matmul_epilogue(uops):
  """Classify the fused epilogue on a matmul kernel: 'plain', 'relu', or None.

  None means a select/compare/transcendental epilogue is present that we can't fuse (sigmoid,
  gelu, tanh, exp, ...); the caller MUST fall back to CPU rather than run a plain matmul and
  silently drop it.

  No dtype filter: plain matmuls — fp16 AND int8(->int32), aligned/unaligned/M-tiled — emit
  ZERO select/compare/transcendental ops (verified by probe), so any such op means a fused
  epilogue. We don't filter on float/half because the int8 path's ReLU compares in int32, not
  float. The relu-SHAPE check (where(0<x,x,0) / max(x,0)) is specific enough that it would not
  match an index/padding mask (where(idx<bound, val, 0)) even if one appeared.

  LIMITATION (pre-existing, unchanged by this matcher): a pure MUL/ADD-const epilogue — e.g.
  (a@b)*2.0 or (a@b)+1.0 — uses only MUL/ADD, indistinguishable from the matmul body by op
  type, so it classifies as 'plain' and the scale/offset is dropped. ReLU and a per-channel
  bias-vector (4th PARAM, folded via pcbias — see _try_match_matmul/_validate_bias, and it
  composes with relu since the bias ADD is invisible here) are handled, as are transcendental
  activations (-> CPU); a scalar const-affine fusion is not."""
  present = {u.op for u in uops if u.op in _MM_EPILOGUE_OPS}
  if not present: return 'plain'
  wheres = [u for u in uops if u.op is Ops.WHERE]
  maxes  = [u for u in uops if u.op is Ops.MAX]
  if present <= {Ops.WHERE, Ops.CMPLT} and wheres and all(_where_is_relu(w) for w in wheres): return 'relu'
  if present == {Ops.MAX} and maxes and all(_max_is_relu(m) for m in maxes): return 'relu'
  return None

def _affine_offset(off):
  """Decompose an INDEX offset uop into (base, {range: linear_coeff}) by symbolic substitution
  (invariant under UPCAST/UNROLL/vectorization). Returns None if the offset isn't affine in its
  ranges. Mirrors the coeff-recovery idiom in _try_match_conv/_try_match_reduce."""
  rs = [u for u in off.toposort() if u.op is Ops.RANGE]
  base = off.substitute({x: x.const_like(0) for x in rs}).simplify()
  if base.op is not Ops.CONST: return None
  co = {}
  for r in rs:
    b = off.substitute({x: x.const_like(1 if x is r else 0) for x in rs}).simplify()
    if b.op is not Ops.CONST: return None
    co[r] = b.arg - base.arg
  return base.arg, co

def _validate_bias(uops, B_p, bias_p, N):
  """True iff `bias_p` is a per-channel bias vector broadcast-ADDed over the output's N (channel)
  axis — the `+ bias[n]` of `a@b + bias[n]` (e.g. nn.Linear). SOUND: any structural deviation
  returns False so the matmul matcher falls back to CPU rather than fold the wrong thing.

  The N (output-channel) axis is identified from the B=[K,N] operand, NOT the output or A — tinygrad
  upcasts those into non-affine stores once M is large. B's index depends (nonzero coeff) only on the
  K (reduce) range and the N range(s), so an N range is just a non-reduce range that indexes B (true
  for the [N,K] transposed weight too). M never indexes B, so a per-row bias[m] — depending on an M
  range — is rejected even when M==N. A genuine bias[n] depends ONLY on N ranges (hence invariant over
  M and the K contraction) and, with its vectorized lane width, tiles exactly [0,N): each channel once."""
  if bias_p.dtype.base is not dtypes.half: return False     # pcbias fold is fp16-only in the runtime
  if bias_p.dtype.size != N: return False                   # must be exactly the N-length channel vector
  def is_reduce(r): return len(r.arg) > 1 and r.arg[1] is AxisType.REDUCE
  n_ranges = set()                                          # B's non-reduce nonzero-coeff ranges = N axis
  for ix in [u for u in uops if u.op is Ops.INDEX and u.src[0] is B_p]:
    r = _affine_offset(ix.src[1])                           # tolerate (skip) non-affine B index nodes
    if r is None: continue
    n_ranges |= {rng for rng, c in r[1].items() if c != 0 and not is_reduce(rng)}
  if not n_ranges: return False
  bias_idx = [u for u in uops if u.op is Ops.INDEX and u.src[0] is bias_p]
  if not bias_idx: return False
  for ix in bias_idx:
    r = _affine_offset(ix.src[1])
    if r is None: return False
    base, co = r
    if base != 0: return False                              # bias starts at channel 0 (N not tiled)
    terms = []
    for rng, c in co.items():
      if c == 0: continue
      if rng not in n_ranges: return False                 # touches M/K -> not a per-channel bias[n]
      ext = rng.src[0].arg if rng.src and rng.src[0].op is Ops.CONST else None
      if ext is None: return False
      terms.append((c, ext))
    V = 1                                                   # vectorized lanes of this load (innermost dim)
    for c in uops:
      if c.op is Ops.CAST and ix in c.src and c.dtype.count > 1: V = c.dtype.count
    stride = V                                              # require contiguous tiling of exactly [0,N)
    for c, e in sorted(terms):
      if c != stride: return False
      stride *= e
    if stride != N: return False
  return True

def _try_match_matmul(uops):
  """Return (M, N, K, fn, relu, bias_pidx) if this uop list is a tinygrad-lowered matmul of a
  supported dtype AND the wrapper can actually handle the size, else None. `bias_pidx` is the PARAM
  arg of a fused per-channel bias vector (`a@b + bias[n]`), or None for a plain/relu matmul.

  Detection is size-based, not kernel-name-based, because tinygrad's loop-opt passes (UPCAST,
  UNROLL, etc.) rename `r_M_N_K` into multi-axis forms like `r_50_4_2_4_4_16_4`. The PARAM
  pointer sizes are invariant under those passes, so we recover (M, N, K) from them:
    out_sz = M*N,  p1_sz = M*K,  p2_sz = K*N   (PARAM-arg order: 0=output, 1=A, 2=B)
    => K*K = p1_sz * p2_sz / out_sz
    => M = p1_sz / K,  N = p2_sz / K
  A REDUCE_AXIS or RANGE somewhere in the AST is required so we don't latch onto a pure
  element-wise kernel that happens to have matching PARAM sizes (e.g. M=K=N=1).

  Final check: the wrapper's pick_tile algorithm has a CBUF feasibility envelope. If the
  matcher accepts a size the wrapper then rejects ("cannot tile"), the wrapper returns
  without writing dst — silently corrupting downstream consumers (the kernel runs but its
  output is the buffer's prior contents). So we MUST mirror the wrapper's tile-feasibility
  check here and fall back to CPU when it would fail."""
  import os, math
  dbg = os.environ.get('NPU_MATMUL_DEBUG') == '1'
  if not any(u.op in (Ops.REDUCE_AXIS, Ops.RANGE) for u in uops):
    return None
  # A real matmul contracts over K: it accumulates K products. The robust signature (verified by a
  # probe across looped/unrolled matmuls vs every EW op) is a DEFINE_REG accumulator (looped K) OR
  # >= 2 float/half MULTIPLIES (unrolled K => K product terms). A perfect-square-length elementwise
  # op (out=p1=p2 => M=N=K=sqrt) factors to the SAME PARAM sizes but has no accumulation — without
  # this guard it silently runs as a fake NxNxN matmul (e.g. (64,64) EW mul -> 4096=64^2 -> bogus
  # 64x64x64 matmul, wrong results). NOTE: do NOT key on float ADD — `a - b` lowers to
  # ADD(a, MUL(b,-1)), a single non-accumulating add that would wrongly pass; multiplies don't lie
  # (mul/add/max rewrite to CUSTOM => 0 float MULs; sub has exactly 1).
  n_fmul = sum(1 for u in uops if u.op in _MM_MUL_OPS and u.dtype.scalar() in (dtypes.float, dtypes.half))
  if not (any(u.op is Ops.DEFINE_REG for u in uops) or n_fmul >= 2):
    if dbg: print("[mm-match] reject: no K-accumulation (elementwise kernel?)")
    return None
  # Classify any fused elementwise epilogue: plain matmul, fused ReLU (DPU BS stage), or an
  # unrecognized activation we must NOT silently drop (-> CPU fallback). See _matmul_epilogue.
  epi = _matmul_epilogue(uops)
  if epi is None:
    if dbg: print("[mm-match] reject: fused epilogue is not plain or relu (-> CPU)")
    return None
  relu = 1 if epi == 'relu' else 0
  params = sorted([u for u in uops if u.op is Ops.PARAM], key=lambda u: u.arg)
  # 3 PARAMs = plain/relu matmul (out, A, B); a 4th PARAM is the per-channel bias of a@b+bias[n].
  if len(params) not in (3, 4): return None
  if params[0].arg != 0 or params[1].arg != 1 or params[2].arg != 2: return None
  if len(params) == 4 and params[3].arg != 3: return None
  # PARAM dtypes are PtrDType wrappers; .base unwraps to the scalar element type.
  out_dt = params[0].dtype.base
  in_dt  = params[1].dtype.base
  if params[2].dtype.base != in_dt: return None    # A and B must share a dtype
  fn = _NPU_MATMUL.get((in_dt, out_dt))
  if fn is None: return None
  out_sz, p1_sz, p2_sz = params[0].dtype.size, params[1].dtype.size, params[2].dtype.size
  if out_sz < 1 or p1_sz < 1 or p2_sz < 1: return None
  k_sq_num = p1_sz * p2_sz
  if k_sq_num % out_sz != 0: return None
  k_sq = k_sq_num // out_sz
  K = math.isqrt(k_sq)
  if K * K != k_sq or K < 1: return None
  if p1_sz % K != 0 or p2_sz % K != 0: return None
  M = p1_sz // K
  N = p2_sz // K
  if M * N != out_sz or M < 1 or N < 1: return None

  # Mirror pick_tile feasibility: K_pad weight column must fit one CBUF bank, AND there must
  # be enough remaining banks for at least one Mt slab of input data. The wrapper enforces
  # Mt>=4 (or 1 when M==1) and Nt>=N_align (16 fp16/bf16, 32 int8). Element width follows the
  # INPUT dtype (int8 inputs are 1 byte even though the output is int32).
  elem_bytes = 2 if in_dt in (dtypes.half, dtypes.bfloat16) else 1
  N_align = 16 if elem_bytes == 2 else 32
  K_pad = ((K + 31) // 32) * 32
  CBUF_BANK = 32768
  CBUF_BANKS_USABLE = 11   # 12 total - 1 reserved
  if K_pad * elem_bytes > CBUF_BANK: return None
  Nt_min = N_align
  weight_banks_min = (K_pad * Nt_min * elem_bytes + CBUF_BANK - 1) // CBUF_BANK
  data_banks_avail = CBUF_BANKS_USABLE - weight_banks_min
  Mt_floor = 1 if M == 1 else 4
  if data_banks_avail < 1: return None
  if Mt_floor * K_pad * elem_bytes > data_banks_avail * CBUF_BANK: return None

  # PARAM2 (B) LAYOUT CHECK. npu_matmul_* tiles its 2nd operand assuming row-major [K,N] (it
  # transposes-while-tiling). A genuinely-transposed [N,K]-contiguous operand (e.g.
  # B.T.contiguous().realize(), or any materialized transposed weight) has identical byte size K*N,
  # so the size-based recovery above can't tell it from [K,N]. Inspect the stride structure: in [K,N]
  # the contraction k is the strided dim and the output dim n is contiguous-ish (max loop coeff <= min
  # reduce coeff); [N,K] inverts that. (Calibrated on the matmul suite: [K,N] gives loop in {1,4},
  # reduce in {16,17,64,128,16384,32768} or empty; [N,K] gives loop={K}, reduce={1}.) [K,N] -> normal;
  # [N,K] fp16 -> the pre-transposed fast path npu_matmul_fp16_bt (contiguous weight pack, no
  # transpose); [N,K] for bf16/int8 (no _bt variant) or can't-tell -> reject (-> CPU).
  if N > 1:
    Rc, Lc, p2_nonaffine = set(), set(), False
    for ix in [u for u in uops if u.op is Ops.INDEX and u.src[0] is params[2]]:
      off = ix.src[1]; rs = [u for u in off.toposort() if u.op is Ops.RANGE]
      base = off.substitute({x: x.const_like(0) for x in rs}).simplify()
      if base.op is not Ops.CONST: p2_nonaffine = True; continue
      for r in rs:
        b = off.substitute({x: x.const_like(1 if x is r else 0) for x in rs}).simplify()
        if b.op is Ops.CONST and (b.arg - base.arg):
          (Rc if (len(r.arg) > 1 and r.arg[1] is AxisType.REDUCE) else Lc).add(b.arg - base.arg)
    if   p2_nonaffine: layout = 'unknown'
    elif Rc and Lc:    layout = 'kn' if max(Lc) <= min(Rc) else ('nk' if max(Rc) <= min(Lc) else 'unknown')
    elif Rc:           layout = 'kn' if min(Rc) > 1 else 'nk'
    elif Lc:           layout = 'kn' if min(Lc) == 1 else 'nk'
    else:              layout = 'kn'                                   # both unrolled / degenerate
    if layout == 'unknown':
      if dbg: print(f"[mm-match] reject: PARAM2 layout unclear (reduce={sorted(Rc)} loop={sorted(Lc)})")
      return None
    if layout == 'nk':
      bt = {'npu_matmul_fp16': 'npu_matmul_fp16_bt',
            'npu_matmul_int8': 'npu_matmul_int8_bt'}.get(fn)          # pre-transposed weight fast path
      if bt is None:
        if dbg: print(f"[mm-match] reject: transposed [N,K] weight, no _bt variant for {fn}")
        return None
      fn = bt
      if dbg: print(f"[mm-match] PARAM2 is [N,K] -> {fn}")

  # PER-CHANNEL BIAS. A 4th PARAM is the `+ bias[n]` of `a@b + bias[n]`. Only the fp16 wrappers fold
  # pcbias (bf16/int8 ignore it), and the bias must be a genuine broadcast-over-N vector — otherwise
  # fall back to CPU (never silently drop a 4th input). The bias ADD is invisible to _matmul_epilogue
  # (plain ADD), so relu composes: relu(a@b + bias[n]) classifies as 'relu' AND folds the bias.
  bias_pidx = None
  if len(params) == 4:
    if fn not in ('npu_matmul_fp16', 'npu_matmul_fp16_bt'):
      if dbg: print(f"[mm-match] reject: 4th (bias) PARAM but {fn} has no pcbias fold -> CPU")
      return None
    if not _validate_bias(uops, params[2], params[3], N):
      if dbg: print("[mm-match] reject: 4th PARAM is not a broadcast-over-N bias -> CPU")
      return None
    bias_pidx = params[3].arg

  if dbg: print(f"[mm-match] M={M} K={K} N={N} relu={relu} bias={bias_pidx is not None} "
                f"(out={out_sz} p1={p1_sz} p2={p2_sz})")
  return (M, N, K, fn, relu, bias_pidx)

def _try_match_reduce(uops):
  """Return (M, K) if this kernel is a pure SUM-reduce of an fp16 tensor over its contiguous
  last axis/axes — routable to npu_sum_lastaxis_fp16 (rowsum via matmul-by-ones, validated in
  reduce_proto.py / reduce_wrapper_test.py) — else None.

  The hazard is silent corruption: (M,K).sum(axis=-1) and (K,M).sum(axis=0) both factor the
  input as M*K, but only the last-axis case lays each output's K summands out contiguously.
  We discriminate on the LINEAR COEFFICIENT of each range var in the input index (recovered by
  symbolic substitution, so it is invariant under UPCAST/UNROLL — tinygrad vectorizes the reduce,
  e.g. K=64 becomes 16 iters x stride-4 of a 4-wide load, so a naive 'stride==1' test fails).

  For SUM the within-block order is irrelevant; correctness needs only that each output reduces
  its own contiguous K-block. We verify:
    - 2 fp16 PARAMs (0=out, 1=in), out_sz | in_sz, K = in_sz//out_sz >= 8 (>= one fp16 pixel);
    - no other float/half ALU (a fused softmax/layernorm reduce -> fall back to CPU);
    - LOOP extents multiply to M (loops enumerate exactly the outputs);
    - the access perfectly tiles [0, in_sz) (bijection => every element summed once), with the
      derived innermost vector width V = in_sz/(loop_ext*red_ext) and red_ext*V == K;
    - loops are the OUTER, block-aligned dims (input coeff >= K) and reduces the INNER (coeff < K),
      so output m reduces exactly [m*K, (m+1)*K)."""
  reds  = [u for u in uops if u.op is Ops.RANGE and len(u.arg) > 1 and u.arg[1] is AxisType.REDUCE]
  loops = [u for u in uops if u.op is Ops.RANGE and len(u.arg) > 1 and u.arg[1] is AxisType.LOOP]
  if not reds: return None
  params = sorted((u for u in uops if u.op is Ops.PARAM), key=lambda u: u.arg)
  if len(params) != 2 or params[0].arg != 0 or params[1].arg != 1: return None
  out_p, in_p = params[0], params[1]
  if out_p.dtype.base is not dtypes.half or in_p.dtype.base is not dtypes.half: return None
  out_sz, in_sz = out_p.dtype.size, in_p.dtype.size
  if out_sz < 1 or in_sz % out_sz != 0: return None
  M, K = out_sz, in_sz // out_sz
  if K < 8: return None

  # pure reduce: ONE accumulator op (ADD=sum, MAX=max) and no other float/half ALU => else fused
  _EXTRA = {Ops.MUL, Ops.SUB, Ops.NEG, Ops.WHERE, Ops.RECIPROCAL, Ops.SQRT,
            Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.CMPLT, Ops.CMPNE}
  if any(u.op in _EXTRA and u.dtype.scalar() in (dtypes.float, dtypes.half) for u in uops): return None
  has_max = any(u.op is Ops.MAX and u.dtype.scalar() in (dtypes.float, dtypes.half) for u in uops)
  kind = 'max' if has_max else 'sum'
  # max path: only the captured M=1,K=256 geometry (npu_max256 pack recipe)
  if kind == 'max' and (out_sz != 1 or in_sz != 256): return None

  in_off = next((u.src[1] for u in uops if u.op is Ops.INDEX and u.src[0] is in_p), None)
  if in_off is None: return None
  off_ranges = [u for u in in_off.toposort() if u.op is Ops.RANGE]
  def extent(r): return r.src[0].arg if r.src and r.src[0].op is Ops.CONST else None
  def coeff(var):  # linear coeff of var in in_off (other ranges -> 0); None if non-linear
    a = in_off.substitute({r: r.const_like(0) for r in off_ranges}).simplify()
    b = in_off.substitute({r: r.const_like(1 if r is var else 0) for r in off_ranges}).simplify()
    return (b.arg - a.arg) if (a.op is Ops.CONST and b.op is Ops.CONST) else None

  dims = []  # (coeff, extent, is_reduce) for every range that indexes the input
  for r in loops + reds:
    c, e = coeff(r), extent(r)
    if c is None or e is None or e < 1: return None
    dims.append((c, e, r in reds))
  loop_ext = 1
  for r in loops:
    loop_ext *= extent(r)
  red_ext = 1
  for r in reds:
    red_ext *= extent(r)
  if loop_ext != M: return None
  if loop_ext * red_ext == 0 or in_sz % (loop_ext * red_ext) != 0: return None
  V = in_sz // (loop_ext * red_ext)                  # innermost contiguous vector lanes
  if red_ext * V != K: return None                   # reduce (+vector) covers exactly the K-block

  # perfect contiguous tiling => bijection onto [0, in_sz): vector fills [0,V), each range stacks
  stride = V
  for c, e, _is_red in sorted(dims, key=lambda d: d[0]):
    if c != stride: return None
    stride *= e
  if stride != in_sz: return None

  # block alignment: loops outer (>=K), reduces inner (<K) => each output gets a clean K-block
  if any(c < K for c, _e, is_red in dims if not is_red): return None
  if any(c >= K for c, _e, is_red in dims if is_red): return None
  return (M, K, kind)

def _try_match_conv(uops):
  """Return NCHW conv geometry {N,Cin,IH,IW,Cout,KH,KW,OH,OW,sh,sw} if this kernel is a
  tinygrad-lowered direct conv2d (the CNA's native op), else None.

  The CNA is a direct-convolution engine; matmul is just its 1x1 case (see npu_matmul.c:
  weight_width/height, conv_x/y_stride, datain_w/h/c, weight_kernels). tinygrad lowers conv2d to
  `(pool(x) * weight).sum(cin,kh,kw)` — the input is read through an overlapping windowed
  ShapeTracker. We recover geometry from the INPUT param's affine index, which survives
  UPCAST/UNROLL/vectorization cleanly (spatial dims stay LOOP ranges, the channel/kernel
  contraction is REDUCE ranges and/or unrolled constant offsets), plus the three PARAM byte sizes.
  Output/weight indices are NOT relied on — output-channel UPCAST makes the output index non-affine.

  Safety: a candidate (Cin, IW, KW) factorization is accepted ONLY if the exact set of input bytes
  it would read equals what the kernel actually reads, so a non-conv (or wrong geometry) can never
  mis-dispatch — important because a wrong `elements`/geometry on this NPU is an OOB DMA that wedges
  the SoC. KH=KW=1 collapses to the matmul lowering and is left to _try_match_matmul. Out of scope
  (returns None -> CPU fallback): dilation!=1, padding!=0, groups!=1, bias, non-fp16.

  NOTE: not yet wired into render() — that pairs with adding an `npu_conv_fp16` CNA entry point
  (generalizing gen_matmul_fp16's 1x1 config to weight_width/height + conv strides)."""
  params = sorted((u for u in uops if u.op is Ops.PARAM), key=lambda u: u.arg)
  if len(params) != 3 or [p.arg for p in params] != [0, 1, 2]: return None
  out_p, in_p, w_p = params
  if any(p.dtype.base is not dtypes.half for p in params): return None
  out_sz, in_sz, w_sz = out_p.dtype.size, in_p.dtype.size, w_p.dtype.size

  def extent(r): return r.src[0].arg if r.src and r.src[0].op is Ops.CONST else None
  def is_reduce(r): return len(r.arg) > 1 and r.arg[1] is AxisType.REDUCE

  # affine {range: coeff} for the INPUT param, plus per-node (const, [(coeff,extent)...], vec) so we
  # can reconstruct the exact set of input bytes read.
  in_idx = [u for u in uops if u.op is Ops.INDEX and u.src[0] is in_p]
  if not in_idx: return None
  ic, nodes = {}, []
  for ix in in_idx:
    off = ix.src[1]
    rs = [u for u in off.toposort() if u.op is Ops.RANGE]
    base = off.substitute({x: x.const_like(0) for x in rs}).simplify()
    if base.op is not Ops.CONST: return None
    terms = []
    for r in rs:
      b = off.substitute({x: x.const_like(1 if x is r else 0) for x in rs}).simplify()
      if b.op is not Ops.CONST: return None
      cf = b.arg - base.arg
      if cf == 0: continue
      if ic.get(r, cf) != cf: return None
      ic[r] = cf
      e = extent(r)
      if e is None: return None
      terms.append((cf, e))
    V = 1
    for c in uops:
      if c.op is Ops.CAST and ix in c.src and c.dtype.count > 1: V = c.dtype.count
    nodes.append((base.arg, terms, V))

  loops = [r for r in ic if not is_reduce(r)]
  if not any(is_reduce(r) for r in ic) or len(loops) < 2: return None   # need contraction + 2 spatial
  loops.sort(key=lambda r: ic[r])
  ow, oh = loops[0], loops[1]                          # ow inner (stride sw), oh next (stride sh*IW)
  OW, OH, sw, sh_IW = extent(ow), extent(oh), ic[ow], ic[oh]
  N = 1
  for r in loops[2:]: N *= extent(r)                   # remaining loop ranges = batch
  if N < 1 or out_sz % (N * OH * OW): return None
  Cout = out_sz // (N * OH * OW)
  if Cout < 1 or w_sz % Cout: return None
  W = w_sz // Cout                                     # = Cin * KH * KW

  # exact set of input offsets the kernel reads (vec/tiling-agnostic); bail if too large to enumerate
  visits = 1
  for _c, terms, Vv in nodes:
    n = Vv
    for _cf, e in terms: n *= e
    visits += n
  if visits > (1 << 20): return None
  actual = set()
  def rec(i, terms, acc, V, out):
    if i == len(terms):
      for l in range(V): out.add(acc + l)
      return
    cf, e = terms[i]
    for v in range(e): rec(i + 1, terms, acc + v * cf, V, out)
  for c, terms, V in nodes: rec(0, terms, c, V, actual)

  def divisors(n): return [d for d in range(1, n + 1) if n % d == 0]
  for Cin in divisors(W):
    if in_sz % N or (in_sz // N) % Cin: continue
    IHIW = (in_sz // N) // Cin
    KHKW = W // Cin
    if KHKW <= 1: continue                             # KH=KW=1 -> matmul's job
    for IW in divisors(IHIW):
      IH = IHIW // IW
      if sh_IW % IW: continue
      sh = sh_IW // IW
      if sh < 1 or sw < 1 or IW < 1: continue
      for KW in divisors(KHKW):
        KH = KHKW // KW
        if IW < KW or IH < KH: continue
        if OW != (IW - KW) // sw + 1 or OH != (IH - KH) // sh + 1: continue
        pred = set()
        for n in range(N):
          for cin in range(Cin):
            for y in range(OH):
              for x in range(OW):
                b = n * Cin * IHIW + cin * IHIW + (y * sh) * IW + (x * sw)
                for ky in range(KH):
                  row = b + ky * IW
                  for kx in range(KW): pred.add(row + kx)
        if pred == actual:
          return dict(N=N, Cin=Cin, IH=IH, IW=IW, Cout=Cout, KH=KH, KW=KW,
                      OH=OH, OW=OW, sh=sh, sw=sw)
  return None

def _conv_signature(uops):
  """True if the kernel is conv-shaped — used to stop a conv that _try_match_conv couldn't parse from
  being mis-grabbed by _try_match_matmul (which size-factors it into a bogus matmul -> silent garbage).
  Signal: the INPUT (param1) has a REDUCE range with coefficient > 16 (= the channel stride IH*IW). A
  matmul's contraction is contiguous/vectorized in its input (coeff = 1 or the vector width, <= 16), so
  it never trips this; a spatial conv's cin stride IH*IW is large. (The weight param can't be used —
  output-channel UPCAST makes it non-affine.) Cost: misses tiny-spatial (IH*IW<=16) convs."""
  params = sorted((u for u in uops if u.op is Ops.PARAM), key=lambda u: u.arg)
  if len(params) != 3 or [p.arg for p in params] != [0, 1, 2]: return False
  def is_reduce(r): return len(r.arg) > 1 and r.arg[1] is AxisType.REDUCE
  in_p = params[1]
  for ix in [u for u in uops if u.op is Ops.INDEX and u.src[0] is in_p]:
    off = ix.src[1]; rs = [u for u in off.toposort() if u.op is Ops.RANGE]
    base = off.substitute({x: x.const_like(0) for x in rs}).simplify()
    if base.op is not Ops.CONST: continue
    for r in rs:
      if not is_reduce(r): continue
      b = off.substitute({x: x.const_like(1 if x is r else 0) for x in rs}).simplify()
      if b.op is Ops.CONST and (b.arg - base.arg) > 16: return True
  return False

# Post-devectorization shape: GEP(LOAD(CAST(INDEX(PARAM, ...))))
_param_gep = UPat(Ops.GEP, src=(UPat(Ops.LOAD, src=(UPat(Ops.CAST, src=(UPat(Ops.INDEX, src=(UPat(Ops.PARAM), UPat())),)),)),))
_const_fp16 = UPat(Ops.CONST, dtype=dtypes.half, name="c")
_const_fp32 = UPat(Ops.CONST, dtype=dtypes.float, name="c")
_const_i16  = UPat(Ops.CONST, dtype=dtypes.int16, name="c")
_const_bf16 = UPat(Ops.CONST, dtype=dtypes.bfloat16, name="c")

def _is_mul_neg1(m):
  """True if m is the CUSTOM npu_mul_scalar(x, -1.0) that tinygrad emits for the `-b` in `a - b`.
  (tinygrad lowers SUB to ADD(a, b*(-1)) before this pre-matcher runs, and the inner b*(-1) is
  itself rewritten to npu_mul_scalar by the vector-OP-scalar rule above.)"""
  return (m.op is Ops.CUSTOM and m.arg in ("npu_mul_scalar", "npu_mul_scalar_bf16", "npu_mul_scalar_i16")
          and len(m.src) == 2 and m.src[1].op is Ops.CONST and float(m.src[1].arg) == -1.0)

# Pre-matcher: tag fp16/fp32/int8 ALU ops whose operands trace to PARAM loads.
rknpu_pm = PatternMatcher([
  # fp16: vector OP vector
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB, Ops.MAX), dtype=dtypes.half, name="u", src=(_param_gep, _param_gep)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_FN[(u.op, u.dtype)])),
  # fp16: vector OP scalar  (note: `scalar - vector` is canonicalized by tinygrad to `v*(-1)+scalar`)
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB, Ops.MAX), dtype=dtypes.half, name="u", src=(_param_gep, _const_fp16)),
   lambda u, c: UOp(Ops.CUSTOM, u.dtype, (u.src[0], c), _NPU_FN_SCALAR[(u.op, u.dtype)])),
  # fp16: unary negate
  (UPat(Ops.NEG, dtype=dtypes.half, name="u", src=(_param_gep,)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_NEG[u.dtype])),
  # fp16: recover vector-vector subtract. `a - b` arrives as ADD(a, npu_mul_scalar(b, -1)) — see
  # _is_mul_neg1 — so fold it back into a single npu_sub(a, b). Both ADD operand orders (commutative).
  (UPat(Ops.ADD, dtype=dtypes.half, name="u", src=(_param_gep, UPat(Ops.CUSTOM, name="m"))),
   lambda u, m: UOp(Ops.CUSTOM, u.dtype, (u.src[0], m.src[0]), _NPU_FN[(Ops.SUB, dtypes.half)]) if _is_mul_neg1(m) else None),
  (UPat(Ops.ADD, dtype=dtypes.half, name="u", src=(UPat(Ops.CUSTOM, name="m"), _param_gep)),
   lambda u, m: UOp(Ops.CUSTOM, u.dtype, (u.src[1], m.src[0]), _NPU_FN[(Ops.SUB, dtypes.half)]) if _is_mul_neg1(m) else None),
  # fp32: ADD/SUB scalar only (MUL scalar hangs; vector-vector broken due to ERDMA 32-bit limit)
  (UPat((Ops.ADD, Ops.SUB), dtype=dtypes.float, name="u", src=(_param_gep, _const_fp32)),
   lambda u, c: UOp(Ops.CUSTOM, u.dtype, (u.src[0], c), _NPU_FN_SCALAR[(u.op, u.dtype)])),
  # fp32: unary negate
  (UPat(Ops.NEG, dtype=dtypes.float, name="u", src=(_param_gep,)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_NEG[u.dtype])),
  # int8: vector OP vector
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB, Ops.MAX), dtype=dtypes.int8, name="u", src=(_param_gep, _param_gep)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_FN[(u.op, u.dtype)])),
  # int8: unary negate
  (UPat(Ops.NEG, dtype=dtypes.int8, name="u", src=(_param_gep,)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_NEG[u.dtype])),
  # int16: vector OP vector
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB, Ops.MAX), dtype=dtypes.int16, name="u", src=(_param_gep, _param_gep)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_FN[(u.op, u.dtype)])),
  # int16: vector OP scalar
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB, Ops.MAX), dtype=dtypes.int16, name="u", src=(_param_gep, _const_i16)),
   lambda u, c: UOp(Ops.CUSTOM, u.dtype, (u.src[0], c), _NPU_FN_SCALAR[(u.op, u.dtype)])),
  # int16: unary negate
  (UPat(Ops.NEG, dtype=dtypes.int16, name="u", src=(_param_gep,)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_NEG[u.dtype])),
  # bf16: vector OP vector. Note: the aarch64 clang backend on RK3588 cannot lower `__bf16 + __bf16`
  # (the CPU is armv8.2-a; bf16 fadd needs armv8.6-a +bf16), so this CUSTOM only renders to valid C
  # when the kernel hits the NPU fast path (`has_loops=False`) and the call becomes a single
  # `npu_*_bf16(...)`. For looped kernels, tinygrad's emulation must take over instead, which it
  # does because we leave `is_dtype_supported(bfloat16, RKNPU)=False` — that triggers
  # `pm_dtype_decomps` to rewrite bf16 ops into ushort/bitshift form before this matcher fires.
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB, Ops.MAX), dtype=dtypes.bfloat16, name="u", src=(_param_gep, _param_gep)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_FN[(u.op, u.dtype)])),
  # bf16: vector OP scalar
  (UPat((Ops.MUL, Ops.ADD, Ops.SUB, Ops.MAX), dtype=dtypes.bfloat16, name="u", src=(_param_gep, _const_bf16)),
   lambda u, c: UOp(Ops.CUSTOM, u.dtype, (u.src[0], c), _NPU_FN_SCALAR[(u.op, u.dtype)])),
  # bf16: unary negate
  (UPat(Ops.NEG, dtype=dtypes.bfloat16, name="u", src=(_param_gep,)),
   lambda u: UOp(Ops.CUSTOM, u.dtype, u.src, _NPU_NEG[u.dtype])),
])


class RkCompiler(ClangJITCompiler):
  def compile(self, src:str) -> bytes:
    return self.compile_to_obj(src)

class RkRenderer(ClangJITRenderer):
  pre_matcher = rknpu_pm

  string_rewrite = PatternMatcher([
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"({ctx[x.src[0]]} * {ctx[x.src[1]]})" if x.arg in ("npu_mul", "npu_mul_scalar", "npu_mul_f32", "npu_mul_scalar_f32", "npu_mul_i8", "npu_mul_i16", "npu_mul_scalar_i16", "npu_mul_bf16", "npu_mul_scalar_bf16") else None),
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"({ctx[x.src[0]]} + {ctx[x.src[1]]})" if x.arg in ("npu_add", "npu_add_scalar", "npu_add_f32", "npu_add_scalar_f32", "npu_add_i8", "npu_add_i16", "npu_add_scalar_i16", "npu_add_bf16", "npu_add_scalar_bf16") else None),
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"({ctx[x.src[0]]} - {ctx[x.src[1]]})" if x.arg in ("npu_sub", "npu_sub_scalar", "npu_sub_f32", "npu_sub_scalar_f32", "npu_sub_i8", "npu_sub_i16", "npu_sub_scalar_i16", "npu_sub_bf16", "npu_sub_scalar_bf16") else None),
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"(-{ctx[x.src[0]]})" if x.arg in ("npu_neg", "npu_neg_f32", "npu_neg_i8", "npu_neg_i16", "npu_neg_bf16") else None),
    (UPat(Ops.CUSTOM, name="x"), lambda ctx, x: f"(({ctx[x.src[0]]}) > ({ctx[x.src[1]]}) ? ({ctx[x.src[0]]}) : ({ctx[x.src[1]]}))" if x.arg in ("npu_max", "npu_max_scalar", "npu_max_i8", "npu_max_i16", "npu_max_scalar_i16", "npu_max_bf16", "npu_max_scalar_bf16") else None),
  ]) + ClangJITRenderer.string_rewrite

  def __init__(self, target: Target):
    super().__init__(target)
    self.compiler = RkCompiler()
    # Names of kernels rendered via an NPU fast path. These call a single libhack
    # npu_*() over the WHOLE buffer, so they must execute exactly once — NOT be
    # data-parallel-split across global_size threads (which would re-submit the full
    # op N times and serialize them on the NPU's per-core FIFO). RKNPUComputeQueue.exec
    # forces threads=1 for these. See _try_match_* / the EW fast path below.
    self._npu_kernel_names: set[str] = set()

  def render(self, uops: list[UOp]) -> str:
    # *** Conv fast path ***
    # tinygrad lowers conv2d to (pool(x) * weight).sum(cin,kh,kw); _try_match_conv recovers NCHW
    # geometry from the input PARAM's windowed index (None for 1x1 -> matmul, and for
    # dilation/padding/groups/non-fp16). PARAMs emit in arg order: 0=output, 1=input, 2=weight.
    # Route the whole kernel to one npu_conv_fp16() (fp16 output, matching tinygrad's fp16 conv).
    cv = _try_match_conv(uops)
    if cv is not None:
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      if len(ptr_indices) == 3:
        i_out, i_in, i_w = ptr_indices[0], ptr_indices[1], ptr_indices[2]
        g = cv
        body = [f"  npu_conv_fp16(npu_fd, "
                f"(void*){bufs[i_out][0]}, dma_{i_out}, obj_{i_out}, "
                f"(const void*){bufs[i_in][0]}, dma_{i_in}, "
                f"(const void*){bufs[i_w][0]}, dma_{i_w}, "
                f"{g['N']}, {g['Cin']}, {g['IH']}, {g['IW']}, {g['Cout']}, "
                f"{g['KH']}, {g['KW']}, {g['sh']}, {g['sw']}, 1);"]
        self._npu_kernel_names.add(_strip_ansi(name))
        return self.render_kernel(name, body, bufs, uops)
    # Conv-shaped but unparsed (heavy UPCAST): fall back to CPU rather than let the matmul matcher
    # size-factor it into a bogus matmul (silent wrong results). Correctness over coverage.
    if _conv_signature(uops):
      return super().render(uops)

    # *** Matmul fast path ***
    # tinygrad lowers `a @ b` to a reduce-loop kernel named "r_M_N_K" (with ANSI color codes).
    # When the AST contains: 3 PARAMs (output, A, B) with ptr sizes M*N / M*K / K*N, one
    # STORE, a REDUCE_AXIS or RANGE loop, and the output dtype is in _NPU_MATMUL, redirect
    # the whole kernel to a single npu_matmul_<dtype>() call. A 4th PARAM that is a per-channel
    # bias vector (a@b + bias[n], e.g. nn.Linear) is folded via pcbias (fp16 wrappers only).
    # Otherwise fall through to the element-wise path / CPU.
    mm = _try_match_matmul(uops)
    if mm is not None:
      M, N, K, fn, relu, bias_pidx = mm
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      # PARAMs emit to bufs in arg order: 0=output, 1=A, 2=B, (3=per-channel bias) (verified via probe).
      n_ptr = 4 if bias_pidx is not None else 3
      if len(ptr_indices) == n_ptr:
        i_out, i_A, i_B = ptr_indices[0], ptr_indices[1], ptr_indices[2]
        pre = []
        # pcbias: the wrapper reads a host fp32 array[N] during weight packing. The bias PARAM is fp16
        # (matched in _validate_bias), so upcast it into a stack temp here (N is a compile-time literal).
        if bias_pidx is not None:
          i_bias = ptr_indices[3]
          pre = [f"  float npu_pcbias[{N}];",
                 f"  for (int _i = 0; _i < {N}; _i++) "
                 f"npu_pcbias[_i] = (float)((const __fp16*){bufs[i_bias][0]})[_i];"]
        bias_arg = "npu_pcbias" if bias_pidx is not None else "(const float*)0"
        body = pre + [f"  {fn}(npu_fd, "
                f"(void*){bufs[i_out][0]}, dma_{i_out}, obj_{i_out}, "
                f"(const void*){bufs[i_A][0]}, dma_{i_A}, "
                f"(const void*){bufs[i_B][0]}, dma_{i_B}, "
                f"{M}, {K}, {N}, {relu}, 0.0f, {bias_arg});"]
        self._npu_kernel_names.add(_strip_ansi(name))
        return self.render_kernel(name, body, bufs, uops)

    # *** Reduce fast path ***
    # A pure SUM-reduce over the contiguous last axis (M,K)->(M,) is a matmul-by-ones.
    # PARAMs emit to bufs in arg order: 0=output, 1=input (same convention as matmul).
    rd = _try_match_reduce(uops)
    if rd is not None:
      M, K, kind = rd
      fn = "npu_max_lastaxis_fp16" if kind == 'max' else "npu_sum_lastaxis_fp16"
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      if len(ptr_indices) == 2:
        i_out, i_in = ptr_indices[0], ptr_indices[1]
        body = [f"  {fn}(npu_fd, "
                f"(void*){bufs[i_out][0]}, dma_{i_out}, obj_{i_out}, "
                f"(const void*){bufs[i_in][0]}, dma_{i_in}, "
                f"{M}, {K});"]
        self._npu_kernel_names.add(_strip_ansi(name))
        return self.render_kernel(name, body, bufs, uops)

    # rknpu_pm rewrites eligible fp16 ALU ops to CUSTOM nodes tagged with the NPU fn name.
    # If any such node survived to render time, this is an NPU-accelerable kernel.
    _all_npu_fns = set(_NPU_FN.values()) | set(_NPU_FN_SCALAR.values()) | set(_NPU_NEG.values())
    npu_ops = [u for u in uops if u.op is Ops.CUSTOM and u.arg in _all_npu_fns]
    # Guards against the pre-matcher having matched only a sub-expression of a complex kernel.
    # The NPU dispatch only emits ONE function call and would silently drop unmatched compute.
    # (1) Check pointer count: scalar/unary expects 2 ptrs, vector-vector expects 3 (or fewer
    #     after in-place dedup). Extra ptrs mean unmatched inputs.
    # (2) Check that no other float/half ALU ops remain — those would be dropped.
    # (3) All CUSTOMs must share one NPU fn name — otherwise we'd need >1 NPU call but emit only 1.
    # (4) Exactly one STORE — fused multi-output kernels would lose the other outputs.
    # The loop is irrelevant: tinygrad's UPCAST vectorizes element-wise kernels into N copies of the
    # same per-lane CUSTOM (e.g. 4 lanes × 64 iters for upcast-4-256). All copies invoke the same
    # NPU op on the same buffer pair, so we bypass the loop and emit a single call that processes
    # all elements; the NPU iterates internally.
    _FLOAT_ALU = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE, Ops.RECIPROCAL, Ops.SQRT,
                  Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.TRUNC, Ops.CMPLT, Ops.CMPNE}
    extra_float_alu = any(u.op in _FLOAT_ALU and u.dtype.scalar() in (dtypes.float, dtypes.half) for u in uops)
    unique_npu_fns = {u.arg for u in npu_ops}
    n_stores = sum(1 for u in uops if u.op is Ops.STORE)
    if npu_ops and len(unique_npu_fns) == 1 and not extra_float_alu and n_stores == 1:
      fn = next(iter(unique_npu_fns))
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      is_scalar_or_unary = fn in _NPU_FN_SCALAR.values() or fn in _NPU_NEG.values()
      max_ptrs = 2 if is_scalar_or_unary else 3
      if len(ptr_indices) > max_ptrs:
        return super().render(uops)
      params = [u for u in uops if u.op is Ops.PARAM]
      n_elem = params[0].dtype.size
      # The DPU writes one 16-byte-aligned pixel slot at a time, so the per-dtype channel
      # count is 16/element_bytes: fp16/bf16=8, int8=16, fp32=4 (fp32 isn't acceleratable
      # anyway). Buffers smaller than one full pixel force out-of-bounds DMA — fall back.
      min_elem = 16 if fn.endswith('_i8') else 8
      if n_elem < min_elem:
        return super().render(uops)
      n = str(n_elem)

      body = []
      # For in-place ops (e.g. x += 3), bufs may dedupe so dst and src share a single ptr_index
      src_idx = ptr_indices[1] if len(ptr_indices) > 1 else ptr_indices[0]
      if fn in _NPU_NEG.values():
        # unary: fn(fd, dst_dma, dst_obj, src_dma, n)
        body.append(f"  {fn}(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{src_idx}, {n});")
      elif fn in _NPU_FN_SCALAR.values():
        # scalar: second src is a CONST — embed literal; only one pointer buffer besides dst
        if fn.endswith('_i16'):
          const_ops = [u for u in uops if u.op is Ops.CONST and u.dtype == dtypes.int16]
          scalar_int = int(const_ops[0].arg) if const_ops else 0
          scalar_cast = ""
          scalar_str = f"(short){scalar_int}"
        else:
          is_f32 = fn.endswith('_f32')
          is_bf16 = fn.endswith('_bf16')
          const_dtype = dtypes.float if is_f32 else (dtypes.bfloat16 if is_bf16 else dtypes.half)
          scalar_cast = "" if is_f32 else ("(__bf16)" if is_bf16 else "(__fp16)")
          const_ops = [u for u in uops if u.op is Ops.CONST and u.dtype == const_dtype]
          scalar_f = float(const_ops[0].arg) if const_ops else 0.0
          if math.isinf(scalar_f): scalar_str = "-__builtin_inff()" if scalar_f < 0 else "__builtin_inff()"
          elif math.isnan(scalar_f): scalar_str = '__builtin_nanf("")'
          else: scalar_str = f"{scalar_f!r}f"
        body.append(f"  {fn}(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{src_idx}, {scalar_cast}{scalar_str}, {n});")
      else:
        # vector OP vector
        body.append(f"  {fn}(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{ptr_indices[1]}, dma_{ptr_indices[2]}, {n});")
        for i in ptr_indices[3:]:
          body.append(f"  {fn}(npu_fd, dma_{ptr_indices[0]}, obj_{ptr_indices[0]}, dma_{ptr_indices[0]}, dma_{i}, {n});")
      self._npu_kernel_names.add(_strip_ansi(name))
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
      'void npu_mul_scalar_f32(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, float scalar, int elements);',
      'void npu_add_scalar_f32(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, float scalar, int elements);',
      'void npu_sub_scalar_f32(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, float scalar, int elements);',
      'void npu_neg_f32(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, int elements);',
      'void npu_mul_i8(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_add_i8(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_sub_i8(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_neg_i8(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, int elements);',
      'void npu_mul_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_add_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_sub_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_neg_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, int elements);',
      'void npu_mul_scalar_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, short scalar, int elements);',
      'void npu_add_scalar_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, short scalar, int elements);',
      'void npu_sub_scalar_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, short scalar, int elements);',
      'void npu_mul_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_add_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_sub_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_neg_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, int elements);',
      'void npu_mul_scalar_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __bf16 scalar, int elements);',
      'void npu_add_scalar_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __bf16 scalar, int elements);',
      'void npu_sub_scalar_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __bf16 scalar, int elements);',
      'void npu_max(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_max_scalar(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __fp16 scalar, int elements);',
      'void npu_max_i8(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_max_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_max_scalar_i16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, short scalar, int elements);',
      'void npu_max_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
      'void npu_max_scalar_bf16(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long src_dma, __bf16 scalar, int elements);',
      'void npu_matmul_fp16(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_matmul_fp16_bt(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_matmul_int8_bt(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_conv_fp16(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *feat_va, unsigned long long feat_dma, const void *weight_va, unsigned long long weight_dma, int N, int Cin, int IH, int IW, int Cout, int KH, int KW, int sh, int sw, int fp16_out);',
      'void npu_matmul_bf16(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_matmul_int8(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_max_lastaxis_fp16(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *src_va, unsigned long long src_dma, int M, int K);',
      'void npu_sum_lastaxis_fp16(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *src_va, unsigned long long src_dma, int M, int K);',
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
    # NPU fast-path kernels emit a single npu_*() over the whole buffer and must run ONCE.
    # tinygrad may split an elementwise/matmul kernel into global_size>1 data-parallel threads;
    # for an NPU kernel that re-submits the FULL op once per thread and serializes them on the
    # NPU FIFO (≈N× slowdown + N concurrent submits on one fd). Force threads=1 for NPU kernels.
    is_npu = _strip_ansi(prg.name) in getattr(prg.dev.renderer, "_npu_kernel_names", ())
    threads = 1 if is_npu else (global_size or (1,))[0]
    return self.cmd(self._rknpu_exec, prg, dev_fd, len(args_state.bufs), len(dma_args),
                    *[x.va_addr for x in args_state.bufs], *dma_args, *args_state.vals, threads=threads)

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
      # tasks.join only guarantees all kernels were submitted; the NPU may still be running
      # them async. npu_drain submits a blocking barrier that returns once the per-core
      # todo_list FIFO has fully drained.
      _lib.npu_drain(self.owner.fd)
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
