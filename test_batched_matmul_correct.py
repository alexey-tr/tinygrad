"""On-device correctness for batched-matmul DISPATCH (attention q@kᵀ / attn@v).

Unlike test_batched_matmul_match.py (matcher-only, no submission), this ACTUALLY RUNS the batched
matmul on the NPU and compares to the CPU reference. It submits real NPU jobs — start small (the
board has no watchdog). Run: tinygrad/.venv/bin/python -P test_batched_matmul_correct.py
"""
import os, numpy as np
os.environ["RKNPU"] = "1"
from tinygrad import Tensor, dtypes, Device
np.random.seed(0)
mk = lambda *s: np.random.randn(*s).astype(np.float16)

def cmp(label, inputs, fn, tol=0.02):
  run = lambda dev: fn(*[Tensor(a, device=dev).realize() for a in inputs]).realize().numpy()
  out_npu, out_cpu = run("RKNPU"), run("CPU")
  err = np.abs(out_npu.astype(np.float32) - out_cpu.astype(np.float32)).max()
  rel = err / (np.abs(out_cpu.astype(np.float32)).max() + 1e-6)
  ok = rel < tol and np.isfinite(out_npu).all()
  print(f"{'OK ' if ok else 'XX '}{label}: shape={out_npu.shape} max_err={err:.4f} rel={rel:.5f}")
  return ok

qkt   = lambda B,H,T,d: ([mk(B,H,T,d), mk(B,H,T,d)], lambda q,k: q @ k.transpose(-2,-1))
attnv = lambda B,H,T,d: ([mk(B,H,T,T), mk(B,H,T,d)], lambda a,v: a @ v)

cases = [
  ("q@kT  1x2x16x16",  *qkt(1,2,16,16)),
  ("q@kT  1x4x32x32",  *qkt(1,4,32,32)),
  ("q@kT  1x8x64x64",  *qkt(1,8,64,64)),
  ("q@kT  1x8x128x64", *qkt(1,8,128,64)),
  ("q@kT  2x8x128x64", *qkt(2,8,128,64)),   # Bh=16 (RANGE x core_id) -> serial loop
  ("q@kT  1x8x256x64", *qkt(1,8,256,64)),
  ("attn@v 1x2x16x16", *attnv(1,2,16,16)),
  ("attn@v 1x8x64x64", *attnv(1,8,64,64)),
  ("attn@v 1x8x256x64",*attnv(1,8,256,64)),
]
fails = 0
for label, inputs, fn in cases:
  if not cmp(label, inputs, fn): fails += 1; print("  -> stopping on failure"); break
print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURES'}")
raise SystemExit(1 if fails else 0)
