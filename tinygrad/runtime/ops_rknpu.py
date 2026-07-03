from __future__ import annotations
import ctypes, functools, mmap, os, queue, threading, math, re

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
def _strip_ansi(s: str) -> str: return _ANSI_RE.sub('', s)  # kernel display names are ANSI-colored; match on plain
from tinygrad.helpers import to_mv, from_mv, mv_address, cpu_profile, Target
from tinygrad.device import BufferSpec
from tinygrad.dtype import dtypes, PtrDType, DType
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, AxisType
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HCQArgsState, MMIOInterface
from tinygrad.runtime.ops_cpu import CPUSignal, CPUWorker, CPUComputeQueue, CPUProgram, CPUAllocator, _in_worker_task
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
# per-channel bias are auto-dispatched (see _sched_match_matmul); scalar `bias`
# is ctypes-only (render passes 0.0) — a const-add epilogue can't be reliably told from the
# matmul's own ADDs. pcbias is fp16-only in the runtime (bf16/int8 wrappers take but ignore it).
for _fn in ['npu_matmul_fp16', 'npu_matmul_bf16', 'npu_matmul_int8', 'npu_matmul_fp16_bt', 'npu_matmul_int8_bt',
            'npu_matmul_fp16_silu']:
  getattr(_lib, _fn).restype = None
  getattr(_lib, _fn).argtypes = [ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                 ctypes.c_void_p]

# void npu_matmul_fp16_batched(int fd, void* dst_va, u64 dst_dma, u64 dst_obj, const void* a_va, u64 a_dma,
#   const void* b_va, u64 b_dma, int batch, int M, int K, int N, int weight_t, int relu)
# `batch` contiguous [M,K]@[K,N] slices in ONE chained, multicore submission (attention q@kᵀ / attn@v).
_lib.npu_matmul_fp16_batched.restype = None
_lib.npu_matmul_fp16_batched.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

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

# void npu_exp_fp16(int fd, uint64_t dst_dma, uint64_t src_dma, int N)
_lib.npu_exp_fp16.restype  = None
_lib.npu_exp_fp16.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# void npu_div(int fd, uint64_t dst_dma, uint64_t dst_obj, uint64_t srcA_dma, uint64_t srcB_dma, int elements)
_lib.npu_div.restype  = None
_lib.npu_div.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                         ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

# void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr)
_lib.mem_destroy.restype = None
_lib.mem_destroy.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64]

# void mm_wcache_drop(int fd, const void *b_va) — invalidate the matmul packed-weight cache for a
# weight VA when its buffer is freed (the cache keys on b_va, which the allocator recycles).
_lib.mm_wcache_drop.restype = None
_lib.mm_wcache_drop.argtypes = [ctypes.c_int, ctypes.c_void_p]


# *** RKNPU Allocator ***

class RKNPUAllocator(CPUAllocator):
  """RKNPU buffers are plain cached RAM, identical to the CPU backend's: matmul/conv/reduce pack from
  the buffer VA into their own internal DMA scratch and unpack back (they ignore the caller dma) and
  every other op runs on the CPU, so no DMA mapping or cache maintenance is ever needed. We therefore
  reuse CPUAllocator's anonymous-mmap _alloc / _as_buffer / _map verbatim and add ONLY the RKNPU
  weight-cache hook: the C matmul runtime caches packed weights keyed on the weight VA, and the
  allocator recycles VAs, so the cache must be dropped when a VA's content changes (copyin) or the
  buffer is freed. _copyin/_copyout are direct memmove (the base would route through a copy queue)."""
  def __init__(self, dev: RKNPUDevice):
    HCQAllocator.__init__(self, dev, supports_copy_from_disk=False, supports_transfer=True)

  def _copyin(self, dest: HCQBuffer, src: memoryview):
    self.dev.synchronize()
    # VAs are pooled/recycled, so copyin — not free — is where a VA's content actually changes; drop
    # any packed-weight tiles cached from the OLD content or a matmul on this VA would reuse them.
    _lib.mm_wcache_drop(self.dev.fd, ctypes.c_void_p(dest.va_addr))
    with cpu_profile(f'TINY -> {self.dev.device}', f"{self.dev.device}:COPY"): ctypes.memmove(int(dest.va_addr), from_mv(src), len(src))

  def _copyout(self, dest: memoryview, src: HCQBuffer):
    self.dev.synchronize()
    with cpu_profile(f'{self.dev.device} -> TINY', f"{self.dev.device}:COPY"): ctypes.memmove(from_mv(dest), int(src.va_addr), len(dest))

  def _do_free(self, buf: HCQBuffer, options: BufferSpec | None = None):
    # meta is the Python mmap object from CPUAllocator._alloc; dropping the buffer unmaps it (GC).
    _lib.mm_wcache_drop(self.dev.fd, ctypes.c_void_p(buf.va_addr))


# *** RKNPU Renderer ***

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

def _strip_cast(u):
  while u.op in (Ops.CAST, Ops.GEP, Ops.BITCAST) and len(u.src) == 1: u = u.src[0]
  return u

def _affine_offset(off):
  """Decompose an INDEX offset uop into (base, {range: linear_coeff}) by symbolic substitution
  (invariant under UPCAST/UNROLL/vectorization). Returns None if the offset isn't affine in its
  ranges. Mirrors the coeff-recovery idiom in _try_match_reduce/_sched_match_matmul."""
  rs = [u for u in off.toposort() if u.op is Ops.RANGE]
  base = off.substitute({x: x.const_like(0) for x in rs}).simplify()
  if base.op is not Ops.CONST: return None
  co = {}
  for r in rs:
    b = off.substitute({x: x.const_like(1 if x is r else 0) for x in rs}).simplify()
    if b.op is not Ops.CONST: return None
    co[r] = b.arg - base.arg
  return base.arg, co

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


class RkCompiler(ClangJITCompiler):
  def compile(self, src:str) -> bytes:
    return self.compile_to_obj(src)

class RkRenderer(ClangJITRenderer):
  # No pre_matcher: element-wise ops are NOT rewritten to npu_* CUSTOM nodes, so they render natively
  # on the CPU via super().render() (EW-on-NPU needs the caller buffer to be DMA-mapped, which the
  # RAM-only allocator no longer provides). Only matmul/conv/reduce go to the NPU — matched directly
  # in render() (not via pre_matcher). bf16 EW decomposes via pm_dtype_decomps in codegen.

  # Keep EXP2 + RECIPROCAL NATIVE (ClangRenderer drops them -> they get decomposed to polynomials).
  # With them native, transcendental activations lower to real Ops.EXP2 / Ops.RECIPROCAL uops, so the
  # matmul matcher can PRECISELY identify silu = x*sigmoid(x) = MUL(acc, RECIPROCAL(ADD(1, EXP2(...))))
  # and distinguish it from sigmoid (no outer MUL-by-acc) / gelu (tanh) — impossible when all three
  # expand to byte-identical CMPLT/WHERE/SUB poly blobs. Mirrors DSPRenderer overriding SQRT; the
  # __builtin_ form links with no libm dependency. (CPU-fallback exp/div now use the accurate builtin
  # instead of the poly approximation — same or better numerics.)
  code_for_op = {**ClangJITRenderer.code_for_op,
                 Ops.EXP2: lambda x,dtype: f"__builtin_exp2({x})" if dtype == dtypes.float64 else f"__builtin_exp2f({x})",
                 Ops.RECIPROCAL: lambda x,dtype: f"(1.0/{x})" if dtype == dtypes.float64 else f"(1.0f/{x})"}

  def __init__(self, target: Target):
    super().__init__(target)
    self.compiler = RkCompiler()
    # Names of kernels rendered via an NPU fast path (matmul/conv/reduce). These call a single libhack
    # npu_*() over the WHOLE buffer, so they must execute exactly once — NOT be data-parallel-split
    # across global_size threads (which would re-submit the full op N times and serialize them on the
    # NPU's per-core FIFO). RKNPUComputeQueue.exec forces threads=1 for these. See _try_match_*.
    self._npu_kernel_names: set[str] = set()

  def render(self, uops: list[UOp]) -> str:
    # *** Conv fast path ***
    # tinygrad lowers conv2d to (pool(x) * weight).sum(cin,kh,kw). The schedule-level _sched_match_conv
    # (stashed in _sched_ann_current by the get_runner hook) recovers NCHW geometry from the clean
    # pre-opt AST and tags it kind='conv'; render emits one npu_conv_fp16() from it. Anything it can't
    # prove (dilation/padding/groups/unit-kernel-dim/non-fp16, or a matmul annotation) -> CPU. The old
    # post-linearize _try_match_conv was deleted: it mis-dispatched rect/k1x3/dilation/groups convs to
    # garbage AND missed valid k5/strided ones; the schedule matcher fixes both.
    cv = _sched_ann_current if (_sched_ann_current is not None and _sched_ann_current.get('kind') == 'conv') else None
    if cv is not None:
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      if len(ptr_indices) == 3:
        i_out, i_in, i_w = ptr_indices[0], ptr_indices[1], ptr_indices[2]
        g = cv
        if g.get('multicore'):
          # Each tinygrad thread handles one batch item; core_id selects the slice.
          in_stride  = g['Cin'] * g['IH'] * g['IW'] * 2   # bytes per batch item in input
          out_stride = g['Cout'] * g['OH'] * g['OW'] * 2  # bytes per batch item in output
          body = [
            f"  npu_conv_fp16(npu_fd, "
            f"(void*)((char*){bufs[i_out][0]} + (long long)core_id * {out_stride}), "
            f"dma_{i_out} + (unsigned long long)core_id * {out_stride}, obj_{i_out}, "
            f"(const void*)((char*){bufs[i_in][0]} + (long long)core_id * {in_stride}), "
            f"dma_{i_in} + (unsigned long long)core_id * {in_stride}, "
            f"(const void*){bufs[i_w][0]}, dma_{i_w}, "
            f"1, {g['Cin']}, {g['IH']}, {g['IW']}, {g['Cout']}, "
            f"{g['KH']}, {g['KW']}, {g['sh']}, {g['sw']}, 1);"]
        else:
          body = [f"  npu_conv_fp16(npu_fd, "
                  f"(void*){bufs[i_out][0]}, dma_{i_out}, obj_{i_out}, "
                  f"(const void*){bufs[i_in][0]}, dma_{i_in}, "
                  f"(const void*){bufs[i_w][0]}, dma_{i_w}, "
                  f"{g['N']}, {g['Cin']}, {g['IH']}, {g['IW']}, {g['Cout']}, "
                  f"{g['KH']}, {g['KW']}, {g['sh']}, {g['sw']}, 1);"]
          self._npu_kernel_names.add(_strip_ansi(name))
        return self.render_kernel(name, body, bufs, uops)

    # *** Batched matmul fast path (attention q@kᵀ / attn@v, [B,H,T,·]) ***
    # Bh>1 branch of the schedule-level match stashed by the get_runner hook (_sched_ann_current).
    # The Bh slices are contiguous, so one npu_matmul_fp16_batched() builds+submits all tiles in a
    # single chained, multicore ioctl. (No post-linearize matcher: the schedule AST gives Bh/M/K/N/
    # layout directly — see the _sched_match_matmul block below.)
    bmm = _sched_ann_current if (_sched_ann_current is not None and _sched_ann_current.get('kind') == 'matmul'
                                 and _sched_ann_current['Bh'] > 1) else None
    if bmm is not None:
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      if len(ptr_indices) == 3:
        i_out, i_A, i_B = ptr_indices[0], ptr_indices[1], ptr_indices[2]
        Bh, M, K, N = bmm['Bh'], bmm['M'], bmm['K'], bmm['N']
        weight_t = 1 if bmm['layout'] == 'nk' else 0   # nk (q@kᵀ) -> pre-transposed pack
        # One call into npu_matmul_fp16_batched: it builds all Bh*tiles and submits them in a single
        # chained, multicore ioctl (the slices are contiguous, base ptr + internal s*M*K/K*N/M*N
        # offsets). Force threads=1 (the whole batch is one C call). The earlier serial-loop and the
        # tinygrad-thread multicore are both superseded — see the batched-matmul-recovery memory note.
        body = [f"  npu_matmul_fp16_batched(npu_fd, "
                f"(void*){bufs[i_out][0]}, dma_{i_out}, obj_{i_out}, "
                f"(const void*){bufs[i_A][0]}, dma_{i_A}, "
                f"(const void*){bufs[i_B][0]}, dma_{i_B}, "
                f"{Bh}, {M}, {K}, {N}, {weight_t}, 0);"]
        self._npu_kernel_names.add(_strip_ansi(name))
        return self.render_kernel(name, body, bufs, uops)

    # *** Matmul fast path ***
    # Bh==1 branch of the schedule-level match (_sched_ann_current). Emits one npu_matmul_<dtype>()
    # for the whole kernel: (M,K,N), dtype/fn, fused relu/silu, and a per-channel bias (a@b+bias[n],
    # e.g. nn.Linear) folded via pcbias — all recovered by _sched_match_matmul from the schedule AST.
    # No annotation (not a matmul, or shape the matcher rejects) -> fall through to reduce/EW/CPU.
    a = _sched_ann_current
    mm = (a['M'], a['N'], a['K'], a['fn'], a['relu'], a['bias_pidx'], a.get('scale', 1.0), a.get('bias_const', 0.0)) \
         if (a is not None and a.get('kind') == 'matmul' and a['Bh'] == 1) else None
    if mm is not None:
      M, N, K, fn, relu, bias_pidx, scale, bias_const = mm
      name, _kernel, bufs = self._render(uops)
      ptr_indices = [i for i, (_, (dtype, _)) in enumerate(bufs) if isinstance(dtype, PtrDType)]
      # PARAMs emit to bufs in arg order: 0=output, 1=A, 2=B, (3=per-channel bias) (verified via probe).
      n_ptr = 4 if bias_pidx is not None else 3
      if len(ptr_indices) == n_ptr:
        i_out, i_A, i_B = ptr_indices[0], ptr_indices[1], ptr_indices[2]
        pre = []
        # pcbias: the wrapper reads a host fp32 array[N] during weight packing. The bias PARAM is fp16
        # (validated in _sched_match_matmul), so upcast it into a stack temp here (N is a compile-time literal).
        if bias_pidx is not None:
          i_bias = ptr_indices[3]
          pre = [f"  float npu_pcbias[{N}];",
                 f"  for (int _i = 0; _i < {N}; _i++) "
                 f"npu_pcbias[_i] = (float)((const __fp16*){bufs[i_bias][0]})[_i];"]
        bias_arg = "npu_pcbias" if bias_pidx is not None else "(const float*)0"
        # A scalar bias-add (a@b + c) fuses into the wrapper's `float bias` slot -> DPU BS ALU (proven).
        npu_call = (f"{fn}(npu_fd, "
                f"(void*){bufs[i_out][0]}, dma_{i_out}, obj_{i_out}, "
                f"(const void*){bufs[i_A][0]}, dma_{i_A}, "
                f"(const void*){bufs[i_B][0]}, dma_{i_B}, "
                f"{M}, {K}, {N}, {relu}, {bias_const!r}f, {bias_arg})")
        # A scalar output-scale ((a@b)*c) is applied as a cheap post-multiply over the [M,N] fp16
        # output (the DPU BS/BN MUL stages don't do an fp32 float multiply here). The point of peeling
        # it in the matcher is to keep the M*K*N matmul on the NPU instead of dumping the whole kernel
        # to the CPU; the O(M*N) scale is the same pass tinygrad would have run anyway. relu is fused
        # pre-output, so relu(x)*c == relu(x*c) for the c>0 the matcher accepts.
        post = ([f"  for (int _s = 0; _s < {M*N}; _s++) ((__fp16*){bufs[i_out][0]})[_s] *= {scale!r}f;"]
                if scale != 1.0 else [])
        # For multicore kernels (global_size>1 → core_id param), the runner calls the
        # C function once per thread. The NPU computes the full result in one call, so
        # guard with core_id==0 to avoid redundant/overwriting launches.
        has_core_id = any(bname == 'core_id' for bname, _ in bufs)
        if has_core_id:
          body = pre + [f"  if (core_id == 0) {{ {npu_call}; {' '.join(s.strip() for s in post)} }}"]
        else:
          body = pre + [f"  {npu_call};"] + post
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
      'void npu_div(int fd, unsigned long long dst_dma, unsigned long long dst_obj, unsigned long long srcA_dma, unsigned long long srcB_dma, int elements);',
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
      'void npu_matmul_fp16_silu(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_matmul_fp16_bt(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_matmul_int8_bt(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int M, int K, int N, int relu, float bias, const float *pcbias);',
      'void npu_matmul_fp16_batched(int fd, void *dst_va, unsigned long long dst_dma, unsigned long long dst_obj, const void *a_va, unsigned long long a_dma, const void *b_va, unsigned long long b_dma, int batch, int M, int K, int N, int weight_t, int relu);',
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
    """Run a kernel: VA pointers + zero dma args + device fd. tinygrad buffers are plain CPU RAM
    (dma=0). matmul/conv/reduce ignore the dma arg (they pack from the VA into their own scratch) and
    every other op runs on the CPU, so there is no dma to resolve and no cache maintenance to do. The
    baked dma_args (args[bufs:bufs+n_dma]) are ignored; zero dma args are passed to match the C
    signature. Under HCQGraph the VA is variable-patched at submit, which is all the C op needs."""
    import platform
    raw_va = args[:bufs]
    vals = list(args[bufs+n_dma:])
    if 'core_id' in prg.runtimevars: vals[prg.runtimevars['core_id']] = tid
    va_args = list(map(ctypes.c_uint64, raw_va))
    vals_mapped = list(map(ctypes.c_int64 if platform.machine() == "arm64" else ctypes.c_int32, vals))
    prg.fxn(*va_args, *vals_mapped, *([ctypes.c_uint64(0)] * (2 * bufs)), ctypes.c_int(dev_fd))
    # This kernel just wrote its output buffer (raw_va[0] = the STORE target, for any kernel — NPU
    # matmul/conv/reduce or a CPU op like an in-place `W = W - lr*grad`). If that VA was a cached
    # matmul weight, its packed copy is now stale, so drop it. With _copyin (host writes) and _do_free
    # (release), this covers every way a buffer's bytes can change -> the weight cache stays sound.
    if bufs: _lib.mm_wcache_drop(dev_fd, ctypes.c_void_p(int(raw_va[0])))

  def exec(self, prg, args_state:HCQArgsState, global_size, local_size):
    dev_fd = prg.dev.fd
    # No dma args: tinygrad buffers are plain RAM (dma=0) and the NPU op wrappers ignore the caller
    # dma (they use the VA + their own scratch), so _rknpu_exec passes zero dma args itself (n_dma=0).
    # NPU fast-path kernels (matmul/conv/reduce) emit a single npu_*() over the whole buffer and must
    # run ONCE; tinygrad may split a kernel into global_size>1 data-parallel threads, which for an NPU
    # kernel would re-submit the FULL op per thread and serialize on the NPU FIFO. Force threads=1.
    is_npu = _strip_ansi(prg.name) in getattr(prg.dev.renderer, "_npu_kernel_names", ())
    threads = 1 if is_npu else (global_size or (1,))[0]
    return self.cmd(self._rknpu_exec, prg, dev_fd, len(args_state.bufs), 0,
                    *[x.va_addr for x in args_state.bufs], *args_state.vals, threads=threads)

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


# ============================================================================
# Schedule-level matmul dispatch (get_runner hook)
#
# The schedule matcher below recovers (M,K,N,layout) cleanly from the pre-opt AST, unlike the
# POST-linearize uop list by byte-size archaeology (isqrt over PARAM sizes, affine-coeff permute
# proofs) because tinygrad's opt passes have destroyed the loop structure by render time. This hook
# matches the SAME matmuls one stage earlier — on the schedule-form AST, where a matmul is still
# REDUCE(MUL(A.index, B.index), Krange, ADD) with tiny affine coeffs, so M/K/N are range extents and
# layout is a coeff read. No isqrt, no square-EW/permuted-A guards. It intercepts get_runner (which
# exec_kernel and jit both call to turn a kernel AST into a Runner) and returns an NpuMatmulRunner
# that calls npu_matmul_*() directly (VA + zero dma, exactly like RKNPUComputeQueue._rknpu_exec).
# 2D + batched, fp16/int8, plain/relu/silu/per-channel-bias. Anything it can't prove -> None ->
# old post-linearize byte-size reconstruction (isqrt/permute proofs), now deleted.
#
# ANNOTATE -> EMIT (JIT-safe): the hook does NOT return a bespoke runner (that broke TinyJit/HCQGraph
# — a plain Runner can't be graph-captured/VA-patched). Instead it STASHES the schedule match in
# `_sched_ann_current` and lets the normal compile path run; RkRenderer.render() reads the stash and
# emits its usual C npu_matmul() call from those params, skipping the render-time archaeology. The
# result is the SAME CompiledRunner render always produced -> HCQGraph captures it, JIT replay is
# faithful. The clean schedule match is just a better SOURCE of (M,K,N,layout,epilogue,bias) than the
# post-linearize isqrt/permute reconstruction. Kill-switch: NPU_NO_SCHED_MATMUL=1.
# ============================================================================
import tinygrad.engine.realize as _realize

_SILU_C_SCHED = -1.4426950408889634   # -log2(e)
_sched_ann_current = None             # schedule match for the kernel currently being rendered (or None)

def _sched_range_ext(r): return r.src[0].arg if r.src and r.src[0].op is Ops.CONST else None
def _sched_is_reduce(r): return r.op is Ops.RANGE and len(r.arg) > 1 and r.arg[-1] is AxisType.REDUCE

def _sched_peel_epilogue(val):
  """(epilogue, bias_index_node, scale, bias_const, reduce_node) from a matmul store value, or all-None.
  Peels relu (max/where) -> silu (x*recip(1+exp2(x*C))) -> scalar output SCALE (reduce*CONST, fused via
  the DPU BS MUL stage, e.g. attention's 1/sqrt(d)) -> per-channel bias ADD (INDEX PARAM) OR scalar bias
  ADD (reduce+CONST, fused via BS ALU), all structurally on the clean schedule nodes. reduce_node is the
  REDUCE feeding the accumulator. `scale`/`bias_const` are Python floats (None = absent)."""
  epi, bias, scale, bias_const, v = 'plain', None, None, None, _strip_cast(val)
  zc = lambda u: u.op is Ops.CONST and u.arg == 0
  if v.op is Ops.MAX and len(v.src) == 2 and any(zc(s) for s in v.src):
    epi = 'relu'; v = _strip_cast(next(s for s in v.src if not zc(s)))
  elif v.op is Ops.WHERE and len(v.src) == 3 and v.src[0].op is Ops.CMPLT \
       and zc(v.src[0].src[0]) and v.src[1] is v.src[0].src[1] and zc(v.src[2]):
    epi = 'relu'; v = _strip_cast(v.src[1])
  if v.op is Ops.MUL and len(v.src) == 2:
    x, rc = v.src
    if rc.op is not Ops.RECIPROCAL: x, rc = rc, x
    if rc.op is Ops.RECIPROCAL and rc.src[0].op is Ops.ADD:
      add = rc.src[0]
      ones = [s for s in add.src if s.op is Ops.CONST and abs(s.arg - 1.0) < 1e-6]
      exps = [s for s in add.src if s.op is Ops.EXP2]
      if len(ones) == 1 and len(exps) == 1 and exps[0].src[0].op is Ops.MUL:
        mul = exps[0].src[0]
        cs = [s for s in mul.src if s.op is Ops.CONST and abs(s.arg - _SILU_C_SCHED) < 1e-3]
        xs = [s for s in mul.src if not (s.op is Ops.CONST and abs(s.arg - _SILU_C_SCHED) < 1e-3)]
        if len(cs) == 1 and len(xs) == 1 and _strip_cast(xs[0]) is _strip_cast(x):
          if epi == 'relu': return None, None, None, None, None
          epi = 'silu'; v = _strip_cast(x)
  # scalar output scale: MUL(reduce, CONST). Not a silu (that consumed the MUL above). The const is on
  # the accumulator, folded into BS MUL. relu(reduce*scale) is fine (BS order MUL -> ALU -> ReLU).
  if epi != 'silu' and v.op is Ops.MUL and len(v.src) == 2:
    cs = [s for s in v.src if s.op is Ops.CONST]
    xs = [s for s in v.src if s.op is not Ops.CONST]
    if len(cs) == 1 and len(xs) == 1 and _strip_cast(xs[0]).op is Ops.REDUCE:
      scale = float(cs[0].arg); v = _strip_cast(xs[0])
  if v.op is Ops.ADD and len(v.src) == 2:
    reds = [s for s in v.src if _strip_cast(s).op is Ops.REDUCE]
    others = [s for s in v.src if _strip_cast(s).op is not Ops.REDUCE]
    if len(reds) == 1 and len(others) == 1:
      o = others[0]
      if o.op is Ops.INDEX and o.src[0].op is Ops.PARAM: bias = o; v = _strip_cast(reds[0])
      elif o.op is Ops.CONST: bias_const = float(o.arg); v = _strip_cast(reds[0])
  if v.op is not Ops.REDUCE: return None, None, None, None, None
  return epi, bias, scale, bias_const, v

def _sched_match_matmul(sink):
  """Return dict(Bh,M,K,N,fn,relu,weight_t,bias_pidx,order,in_dt) if `sink` is a schedule-form
  matmul the runtime can run, else None. Mirrors the render-path dtype routing + CBUF feasibility."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return None
  out_idx, val = stores[0].src[0], stores[0].src[1]
  epi, bias, scale, bias_const, red = _sched_peel_epilogue(val)
  if epi is None or red.arg is not Ops.ADD: return None
  red_ranges = [s for s in red.src[1:] if s.op is Ops.RANGE]
  if not red_ranges: return None
  body = _strip_cast(red.src[0])
  if body.op is not Ops.MUL or len(body.src) != 2: return None
  ia, ib = body.src
  if ia.op is not Ops.INDEX or ib.op is not Ops.INDEX: return None
  if ia.src[0].op is not Ops.PARAM or ib.src[0].op is not Ops.PARAM: return None
  a_p, b_p = ia.src[0], ib.src[0]
  ra, rb, ro = _affine_offset(ia.src[1]), _affine_offset(ib.src[1]), _affine_offset(out_idx.src[1])
  if ra is None or rb is None or ro is None: return None
  ac, bc, oc = ra[1], rb[1], ro[1]
  loops = [r for r in set(ac) | set(bc) if not _sched_is_reduce(r)]
  m_axes = [r for r in loops if ac.get(r, 0) != 0 and bc.get(r, 0) == 0]
  n_axes = [r for r in loops if bc.get(r, 0) != 0 and ac.get(r, 0) == 0]
  batch  = [r for r in loops if ac.get(r, 0) != 0 and bc.get(r, 0) != 0]
  if len(m_axes) != 1 or len(n_axes) != 1: return None
  mr, nr = m_axes[0], n_axes[0]
  M, N = _sched_range_ext(mr), _sched_range_ext(nr)
  if M is None or N is None: return None
  Bh = 1
  for r in batch:
    e = _sched_range_ext(r)
    if e is None or e < 1: return None
    Bh *= e
  # K may be a SINGLE reduce axis (plain a@b) or SEVERAL (a multi-axis contraction, e.g. einsum
  # 'mkl,kln->mn'). Either way the runtime needs one flattened K, which is valid iff the reduce axes
  # form a CONTIGUOUS run in A (innermost stride 1, each next = product of previous extents) — then
  # K = product of extents. Order the axes by their A-stride and verify the flatten.
  ks = sorted(red_ranges, key=lambda r: ac.get(r, 0))
  K, stride = 1, 1
  for r in ks:
    e = _sched_range_ext(r)
    if e is None or e < 1 or ac.get(r, 0) != stride: return None                # non-contiguous K in A
    K *= e; stride *= e
  if ac.get(mr) != K: return None                                              # A must be packed [M,K]
  # B: kn -> every k-stride is N× its A-stride and N is innermost (bc[n]==1);
  #    nk -> every k-stride equals its A-stride (contiguous [.,K]) and bc[n]==K.
  if   bc.get(nr) == 1 and all(bc.get(r, 0) == N * ac.get(r, 0) for r in ks): layout = 'kn'
  elif bc.get(nr) == K and all(bc.get(r, 0) == ac.get(r, 0) for r in ks):     layout = 'nk'
  else: return None
  for r in batch:                                                             # contiguous batch slices
    if ac.get(r, 0) % (M * K) or bc.get(r, 0) % (K * N) or oc.get(r, 0) % (M * N): return None

  out_dt, in_dt = out_idx.src[0].dtype.base, a_p.dtype.base
  if b_p.dtype.base is not in_dt: return None
  fn = _NPU_MATMUL.get((in_dt, out_dt))
  if fn is None: return None
  relu = 1 if epi == 'relu' else 0
  # CBUF feasibility: K_pad column must fit one bank + room for one Mt slab (else the wrapper can't tile)
  elem_bytes = 2 if in_dt in (dtypes.half, dtypes.bfloat16) else 1
  N_align, K_pad = (16 if elem_bytes == 2 else 32), ((K + 31) // 32) * 32
  CBUF_BANK, USABLE = 32768, 11
  if K_pad * elem_bytes > CBUF_BANK: return None
  wb_min = (K_pad * N_align * elem_bytes + CBUF_BANK - 1) // CBUF_BANK
  avail = USABLE - wb_min
  if avail < 1: return None
  if (1 if M == 1 else 4) * K_pad * elem_bytes > avail * CBUF_BANK: return None

  if Bh > 1:
    if fn != 'npu_matmul_fp16' or bias is not None or epi == 'silu' \
       or scale is not None or bias_const is not None: return None   # batched: fp16 plain/relu
  else:
    if layout == 'nk':
      fn = {'npu_matmul_fp16': 'npu_matmul_fp16_bt', 'npu_matmul_int8': 'npu_matmul_int8_bt'}.get(fn)
      if fn is None: return None
  bias_pidx = None
  if bias is not None:
    if fn not in ('npu_matmul_fp16', 'npu_matmul_fp16_bt') or Bh > 1: return None
    rbias = _affine_offset(bias.src[1])
    if rbias is None or rbias[0] != 0: return None
    if any(c != 0 and r is not nr for r, c in rbias[1].items()) or rbias[1].get(nr, 0) != 1: return None
    if bias.src[0].dtype.base is not dtypes.half or bias.src[0].dtype.size != N: return None
    bias_pidx = bias.src[0].arg
  # Scalar output scale (BS MUL) and scalar bias (BS ALU register) are fp16-only. Keep them mutually
  # exclusive with each other and with per-channel bias -- each alone is validated; combined BS
  # MUL+ALU ordering is untested, so a compound epilogue falls back to CPU.
  if scale is not None and (fn not in ('npu_matmul_fp16', 'npu_matmul_fp16_bt') or bias_pidx is not None
                            or bias_const is not None or (relu and scale <= 0)): return None
  if bias_const is not None and (fn not in ('npu_matmul_fp16', 'npu_matmul_fp16_bt') or bias_pidx is not None):
    return None
  if epi == 'silu':
    if fn != 'npu_matmul_fp16' or bias_pidx is not None or (M * N) % 8 != 0 or Bh > 1: return None
    fn = 'npu_matmul_fp16_silu'
  return dict(kind='matmul', Bh=Bh, M=M, N=N, K=K, fn=fn, relu=relu, bias_pidx=bias_pidx, layout=layout,
              scale=(1.0 if scale is None else scale), bias_const=(0.0 if bias_const is None else bias_const))

def _sched_match_conv(sink):
  """Schedule-form conv2d match -> dict(kind='conv', N,Cin,IH,IW,Cout,KH,KW,sh,sw) or None.
  NCHW geometry is direct affine-coeff reads (IW=coeff_in(KH), sw=coeff_in(OW), sh=coeff_in(OH)/IW,
  IH=coeff_in(Cin)/IW; reduce axes sorted by input-stride -> KW,KH,Cin; loops by output-stride ->
  OW,OH,N,Cout). fp16 only, dilation=1, no padding/groups, KH*KW>1 (1x1 -> matmul), no fused
  epilogue (npu_conv_fp16 has no relu/bias). Emitted params match the render conv fast path's `g`."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return None
  out_idx, val = stores[0].src[0], stores[0].src[1]
  v = _strip_cast(val)
  if v.op is not Ops.REDUCE or v.arg is not Ops.ADD: return None       # no fused epilogue for conv
  red = [s for s in v.src[1:] if s.op is Ops.RANGE]
  if len(red) != 3: return None                                        # Cin,KH,KW (1x1->matmul; dil/groups differ)
  body = _strip_cast(v.src[0])
  if body.op is not Ops.MUL or len(body.src) != 2: return None         # a padding mask breaks the pure MUL
  ia, ib = body.src
  if ia.op is not Ops.INDEX or ib.op is not Ops.INDEX: return None
  if ia.src[0].op is not Ops.PARAM or ib.src[0].op is not Ops.PARAM: return None
  if any(x.dtype.base is not dtypes.half for x in (out_idx.src[0], ia.src[0], ib.src[0])): return None
  ra, rb, ro = _affine_offset(ia.src[1]), _affine_offset(ib.src[1]), _affine_offset(out_idx.src[1])
  if ra is None or rb is None or ro is None: return None
  ac, bc, oc = ra[1], rb[1], ro[1]
  loops = [r for r in set(ac) | set(bc) if not _sched_is_reduce(r)]
  a_loops, b_loops = [r for r in loops if ac.get(r, 0)], [r for r in loops if bc.get(r, 0)]
  if   len(a_loops) == 1 and len(b_loops) > 1: w_i, in_i, wc, inc, cout_r = ia, ib, ac, bc, a_loops[0]
  elif len(b_loops) == 1 and len(a_loops) > 1: w_i, in_i, wc, inc, cout_r = ib, ia, bc, ac, b_loops[0]
  else: return None                                                    # weight = operand indexed by 1 loop (Cout)
  in_loops = [r for r in loops if r is not cout_r]
  ow_r = next((r for r in in_loops if oc.get(r) == 1), None)           # OW: output stride 1
  if ow_r is None: return None
  OW = _sched_range_ext(ow_r)
  oh_r = next((r for r in in_loops if oc.get(r) == OW), None)          # OH: output stride OW
  if oh_r is None: return None
  OH, Cout = _sched_range_ext(oh_r), _sched_range_ext(cout_r)
  if OW is None or OH is None or Cout is None: return None
  n_r = next((r for r in in_loops if oc.get(r) == Cout * OH * OW), None)   # N (batch): optional
  N = _sched_range_ext(n_r) if n_r is not None else 1
  if oc.get(cout_r) != OH * OW: return None
  if {ow_r, oh_r} | ({n_r} if n_r is not None else set()) != set(in_loops): return None  # no stray loop
  kw_r, kh_r, cin_r = sorted(red, key=lambda r: inc.get(r, 0))         # by input stride: KW(1),KH(IW),Cin(IH*IW)
  IW, Cin, KW, KH = inc.get(kh_r), _sched_range_ext(cin_r), _sched_range_ext(kw_r), _sched_range_ext(kh_r)
  if IW is None or IW < 1 or Cin is None or KW is None or KH is None: return None
  if inc.get(kw_r) != 1 or inc.get(cin_r) % IW: return None            # dilation_w!=1 / bad Cin stride
  IH = inc.get(cin_r) // IW
  sw, sh_num = inc.get(ow_r), inc.get(oh_r)
  if sw is None or sh_num is None or sh_num % IW: return None
  sh = sh_num // IW
  # soundness: weight is [Cout,Cin,KH,KW] and the remaining input strides are consistent
  if wc.get(kw_r) != 1 or wc.get(kh_r) != KW or wc.get(cin_r) != KH * KW or wc.get(cout_r) != Cin * KH * KW: return None
  if inc.get(oh_r) != sh * IW or inc.get(cin_r) != IH * IW: return None
  if n_r is not None and inc.get(n_r) != Cin * IH * IW: return None
  if KH * KW <= 1: return None                                         # 1x1 is a matmul
  return dict(kind='conv', N=N, Cin=Cin, IH=IH, IW=IW, Cout=Cout, KH=KH, KW=KW, sh=sh, sw=sw)

_sched_cache: dict = {}
_sched_installed = [False]
def _install_sched_matmul():
  if _sched_installed[0] or os.environ.get('NPU_NO_SCHED_MATMUL'): return
  _sched_installed[0] = True
  _orig_get_runner = _realize.get_runner
  def _hooked_get_runner(device, ast):
    # Compute the schedule-level match here (only place with the pre-opt AST), stash it, then let the
    # normal compile+render path build the CompiledRunner — render() reads _sched_ann_current.
    global _sched_ann_current
    if device.split(":")[0] == "RKNPU":
      sink = ast.src[0] if ast.op is Ops.BEAM else ast
      if sink.op is Ops.SINK:
        ckey = (device, sink.key)
        if ckey not in _sched_cache: _sched_cache[ckey] = _sched_match_matmul(sink) or _sched_match_conv(sink)
        _sched_ann_current = _sched_cache[ckey]
        try: return _orig_get_runner(device, ast)     # render() consumes _sched_ann_current on cache-miss
        finally: _sched_ann_current = None
    return _orig_get_runner(device, ast)
  _realize.get_runner = _hooked_get_runner


class RKNPUDevice(HCQCompiled):
  _shared_fd: int = -1
  _fd_refcount: int = 0
  _fd_lock: threading.Lock = threading.Lock()

  # No VA->dma registry: tinygrad buffers are plain CPU RAM (dma=0). The NPU op wrappers
  # (matmul/conv/reduce) allocate their own DMA scratch internally and only read/write the caller
  # buffer via its VA, so the device never needs to map a tinygrad VA to a dma address. Under
  # HCQGraph the VA is variable-patched at submit, which is all the C ops consume.

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
    _install_sched_matmul()   # intercept matmul kernels at schedule time (get_runner hook)

  def finalize(self):
    super().finalize()
    with RKNPUDevice._fd_lock:
      RKNPUDevice._fd_refcount -= 1
      fd_to_close = RKNPUDevice._shared_fd if RKNPUDevice._fd_refcount == 0 else -1
      if fd_to_close >= 0:
        RKNPUDevice._shared_fd = -1
    if fd_to_close >= 0:
      _lib.npu_close(fd_to_close)
