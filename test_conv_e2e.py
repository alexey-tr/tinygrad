"""E2E safety sweep: Tensor.conv2d on RKNPU must be CORRECT for every shape (NPU-matched or CPU
fallback) — never garbage. Compares to a numpy fp32 reference (fp16-rounded)."""
import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import dtypes

def npconv(x, w, sh, sw):
  N,Cin,IH,IW = x.shape; Cout,_,KH,KW = w.shape
  OH=(IH-KH)//sh+1; OW=(IW-KW)//sw+1
  y=np.zeros((N,Cout,OH,OW),np.float32)
  for n in range(N):
    for oc in range(Cout):
      for oh in range(OH):
        for ow in range(OW):
          y[n,oc,oh,ow]=np.sum(x[n,:,oh*sh:oh*sh+KH,ow*sw:ow*sw+KW].astype(np.float32)*w[oc].astype(np.float32))
  return y

def run(N,Cin,IH,IW,Cout,KH,KW,sh=1,sw=1):
  rng=np.random.default_rng(0)
  xn=(rng.standard_normal((N,Cin,IH,IW))*0.5).astype(np.float16)
  wn=(rng.standard_normal((Cout,Cin,KH,KW))*0.25).astype(np.float16)
  got=Tensor(xn,device="RKNPU").conv2d(Tensor(wn,device="RKNPU"),stride=(sh,sw)).numpy().astype(np.float32)
  ref=npconv(xn,wn,sh,sw).astype(np.float16).astype(np.float32)
  if got.shape!=ref.shape:
    print(f"BAD  N{N} Cin{Cin} {IH}x{IW} Cout{Cout} k{KH}x{KW} s{sh} shape {got.shape}!={ref.shape}"); return False
  rel=np.abs(got-ref)/(np.abs(ref)+1e-3)
  ok=rel.max()<2e-2
  print(f"{'OK ' if ok else 'BAD'} N{N} Cin{Cin} {IH}x{IW} Cout{Cout} k{KH}x{KW} s{sh} max_rel={rel.max():.2e}")
  return ok

if __name__=="__main__":
  shapes=[(1,8,8,8,16,3,3,1),(1,16,8,8,32,3,3,1),(1,16,8,8,24,3,3,1),(1,32,8,8,32,3,3,1),
          (1,64,10,10,16,3,3,1),(1,128,8,8,64,3,3,1),(1,16,12,12,48,3,3,1),(1,64,7,7,32,3,3,2),
          (1,8,7,9,8,3,5,1),(2,16,6,6,16,3,3,1),(1,16,6,6,16,3,3,1),(1,4,8,8,8,3,3,2),
          (1,32,12,12,32,5,5,2),(1,16,16,16,32,3,3,1),(1,8,5,5,16,3,3,1),(1,16,9,9,16,3,3,1)]
  # NOTE: 1x1-OUTPUT convs (KH==IH) are matmul-shaped -> handled by the matmul matcher, which has a
  # pre-existing M=1/odd-K gap (out of scope here); the conv matcher correctly returns None for them.
  r=[run(*s) for s in shapes]
  print(f"\n{sum(r)}/{len(r)} correct  ({len(r)-sum(r)} WRONG)")
