"""Phase-1 validation for the batched-matmul matcher (_try_match_batched_matmul).

Runs in RECOVER-AND-LOG mode: every dispatch matcher is patched to None so realize() falls back to
the CPU path (ZERO NPU submission — safe for the watchdog-less board). We then call the REAL
_try_match_batched_matmul on each rendered kernel's uops and check the recovered (Bh,M,K,N,layout)
against ground truth. Run: tinygrad/.venv/bin/python -P test_batched_matmul_match.py
"""
import os
os.environ["RKNPU"] = "1"
from tinygrad import Tensor, dtypes, Device
import tinygrad.runtime.ops_rknpu as R
from tinygrad.uop.ops import Ops

bmm = R._try_match_batched_matmul                       # the matcher under test (real)
# SAFETY: disable all DISPATCH matchers -> pure CPU fallback -> no NPU jobs.
R._try_match_matmul = lambda u: None; R._try_match_conv = lambda u: None
R._try_match_reduce = lambda u: None; R._conv_signature = lambda u: False

_caps = []
_orig = R.RkRenderer.render
def _cap(self, uops):
  if sum(1 for u in uops if u.op is Ops.PARAM) >= 2: _caps.append(list(uops))
  return _orig(self, uops)
R.RkRenderer.render = _cap
Device["RKNPU"]                                         # bring the device up

H  = lambda *s: Tensor.randn(*s, dtype=dtypes.half,  device="RKNPU").realize()
HF = lambda *s: Tensor.randn(*s, dtype=dtypes.float, device="RKNPU").realize()
_fails = 0
def check(label, fn, expect):
  global _fails
  _caps.clear()
  try: fn().realize()
  except Exception as e:
    print(f"XX {label}: EXCEPTION {type(e).__name__}: {e}"); _fails += 1; return
  got = next((r for r in (bmm(u) for u in _caps) if r), None)
  if expect is None:
    ok = got is None
  else:
    ok = got is not None and all(got[k] == v for k, v in expect.items())
  if not ok: _fails += 1
  shown = {k: got[k] for k in ('Bh','M','K','N','layout')} if got else None
  print(f"{'OK ' if ok else 'XX '}{label}: got={shown} want={expect}")

# --- q@kᵀ (transpose-view) : M=N=T, K=d, layout nk (B slice is [N,K]) ---
check("q@kT  B1H4 T16  d16",  lambda: H(1,4,16,16)  @ H(1,4,16,16).transpose(-2,-1),  dict(Bh=4,  M=16,  K=16, N=16,  layout='nk'))
check("q@kT  B1H8 T64  d64",  lambda: H(1,8,64,64)  @ H(1,8,64,64).transpose(-2,-1),  dict(Bh=8,  M=64,  K=64, N=64,  layout='nk'))
check("q@kT  B1H8 T128 d64",  lambda: H(1,8,128,64) @ H(1,8,128,64).transpose(-2,-1), dict(Bh=8,  M=128, K=64, N=128, layout='nk'))
check("q@kT  B1H8 T256 d64",  lambda: H(1,8,256,64) @ H(1,8,256,64).transpose(-2,-1), dict(Bh=8,  M=256, K=64, N=256, layout='nk'))
check("q@kT  B2H4 T32  d16",  lambda: H(2,4,32,16)  @ H(2,4,32,16).transpose(-2,-1),  dict(Bh=8,  M=32,  K=16, N=32,  layout='nk'))
check("q@kT  B2H8 T128 d64",  lambda: H(2,8,128,64) @ H(2,8,128,64).transpose(-2,-1), dict(Bh=16, M=128, K=64, N=128, layout='nk'))
# --- attn@v : aw[T,T] @ v[T,d] -> M=T, K=T, N=d, layout kn (B slice is [K,N]) ---
check("attn@v B1H4 T16  d16",  lambda: H(1,4,16,16)  @ H(1,4,16,16),  dict(Bh=4, M=16,  K=16,  N=16, layout='kn'))
check("attn@v B1H8 T64  d64",  lambda: H(1,8,64,64)  @ H(1,8,64,64),  dict(Bh=8, M=64,  K=64,  N=64, layout='kn'))
check("attn@v B1H8 T256 d64",  lambda: H(1,8,256,256)@ H(1,8,256,64), dict(Bh=8, M=256, K=256, N=64, layout='kn'))
# --- must REJECT (-> CPU): not plain fp16, or not a contraction ---
check("REJECT relu(q@kT)",  lambda: (H(1,8,64,64) @ H(1,8,64,64).transpose(-2,-1)).relu(), None)
check("REJECT bf16 batched", lambda: H(1,4,32,32).cast(dtypes.bfloat16) @ H(1,4,32,32).cast(dtypes.bfloat16).transpose(-2,-1), None)
check("REJECT fp32 batched", lambda: HF(1,4,32,32) @ HF(1,4,32,32).transpose(-2,-1), None)
check("REJECT batched add",  lambda: H(2,4,32,32) + H(2,4,32,32), None)
check("REJECT batched mul",  lambda: H(2,4,16,16) * H(2,4,16,16), None)
check("REJECT plain 2D mm",  lambda: H(32,32) @ H(32,32), None)

print(f"\n{'ALL PASS' if _fails == 0 else str(_fails) + ' FAILURES'}")
raise SystemExit(1 if _fails else 0)
