"""Test battery for _try_match_conv (the RKNPU conv / pool-view matcher) in ops_rknpu.

Lowers real tinygrad conv2d/matmul/EW kernels with the RKNPU renderer and checks the matcher
recovers exact geometry for in-scope convs, delegates KH=KW=1 to the matmul matcher, and refuses
(returns None) out-of-scope convs and non-conv kernels — so a wrong geometry can never mis-dispatch
(an OOB DMA wedges this SoC).

Run:  .venv/bin/python test_conv_match.py
"""
import os
os.environ["DEBUG"] = "0"
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops
from tinygrad.codegen import get_program
from tinygrad.runtime.ops_rknpu import RkRenderer, _try_match_conv, _try_match_matmul

ren = RkRenderer(Target.parse("RKNPU"))
class _Done(Exception):
  def __init__(self, uops): self.uops = uops
RkRenderer.render = lambda self, uops: (_ for _ in ()).throw(_Done(uops))

def lower(t):
  for si in t.schedule():
    if si.ast.op is not Ops.SINK: continue
    try: get_program(si.ast, ren)
    except _Done as d:
      if any(u.op is Ops.RANGE for u in d.uops): return d.uops
  return None

H = dtypes.half
def conv(n, cin, ih, iw, cout, kh, kw, s):
  return Tensor.empty(n, cin, ih, iw, dtype=H).conv2d(Tensor.empty(cout, cin, kh, kw, dtype=H), stride=s)

def expect_conv(n, cin, ih, iw, cout, kh, kw, s):
  got = _try_match_conv(lower(conv(n, cin, ih, iw, cout, kh, kw, s)))
  oh, ow = (ih - kh) // s + 1, (iw - kw) // s + 1
  exp = dict(N=n, Cin=cin, IH=ih, IW=iw, Cout=cout, KH=kh, KW=kw, OH=oh, OW=ow, sh=s, sw=s)
  ok = got == exp
  print(f"{'OK ' if ok else 'BAD'} conv N{n} Cin{cin} {ih}x{iw} Cout{cout} k{kh}x{kw} s{s}"
        + ("" if ok else f"\n    exp={exp}\n    got={got}"))
  return ok

def expect_none(label, t):
  got = _try_match_conv(lower(t))
  print(f"{'OK ' if got is None else 'BAD'} (None) {label}: got={got}")
  return got is None

def expect_matmul(label, t):
  u = lower(t); conv_g, mm = _try_match_conv(u), _try_match_matmul(u)
  ok = conv_g is None and mm is not None
  print(f"{'OK ' if ok else 'BAD'} (matmul,not conv) {label}: conv={conv_g} matmul={mm}")
  return ok

if __name__ == "__main__":
  r = []
  # in-scope convs: Cin/Cout/spatial/stride/batch, symmetric + asymmetric kernels
  r.append(expect_conv(1, 8, 6, 6, 16, 3, 3, 1))
  r.append(expect_conv(1, 4, 8, 8, 8, 3, 3, 2))
  r.append(expect_conv(1, 16, 10, 10, 8, 3, 3, 1))
  r.append(expect_conv(1, 8, 7, 7, 8, 3, 3, 2))
  r.append(expect_conv(2, 8, 6, 6, 8, 3, 3, 1))
  r.append(expect_conv(1, 16, 16, 16, 32, 3, 3, 1))
  r.append(expect_conv(1, 8, 7, 9, 8, 3, 5, 1))     # asymmetric kernel (KH kept as reduce range)
  r.append(expect_conv(1, 8, 9, 7, 16, 3, 3, 2))    # rectangular spatial, stride 2
  # KH=KW=1 (and 1x1-output) collapse to the matmul lowering -> matmul matcher owns them
  r.append(expect_matmul("1x1 conv 1x64x8x8*32x64x1x1", conv(1, 64, 8, 8, 32, 1, 1, 1)))
  r.append(expect_matmul("k5 on 5x5 -> 1x1 out", conv(1, 32, 5, 5, 16, 5, 5, 1)))
  # out-of-scope convs MUST fall back, never mis-dispatch
  r.append(expect_none("dilation2", Tensor.empty(1,8,10,10,dtype=H).conv2d(Tensor.empty(8,8,3,3,dtype=H), dilation=2)))
  r.append(expect_none("padding1",  Tensor.empty(1,8,8,8,dtype=H).conv2d(Tensor.empty(8,8,3,3,dtype=H), padding=1)))
  r.append(expect_none("groups2",   Tensor.empty(1,8,8,8,dtype=H).conv2d(Tensor.empty(8,4,3,3,dtype=H), groups=2)))
  # non-conv kernels
  r.append(expect_none("matmul", Tensor.empty(8,32,dtype=H) @ Tensor.empty(32,16,dtype=H)))
  r.append(expect_none("ew-mul", Tensor.empty(64,64,dtype=H) * Tensor.empty(64,64,dtype=H)))
  r.append(expect_none("rowsum", Tensor.empty(32,64,dtype=H).sum(axis=1)))
  print(f"\n{sum(r)}/{len(r)} passed")
