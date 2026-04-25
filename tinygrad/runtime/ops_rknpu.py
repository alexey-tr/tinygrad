from __future__ import annotations
import ctypes, functools, mmap, queue, threading
from tinygrad.helpers import to_mv, mv_address
from tinygrad.device import BufferSpec
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, MMIOInterface
from tinygrad.runtime.ops_cpu import CPUSignal, CPUWorker, CPUComputeQueue, CPUProgram
from tinygrad.renderer.cstyle import ClangJITRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer

# *** ctypes bindings for librk3588-npu.so ***

_LIB_PATH = "/home/alexey/src/hack/rk3588-npu/build/librk3588-npu.so"

try:
  _lib = ctypes.CDLL(_LIB_PATH)
except OSError as e:
  raise RuntimeError(f"Failed to load librk3588-npu.so from {_LIB_PATH}: {e}") from e

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
    print (f"alloc {size}, {options}")
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
      [ClangJITRenderer, CPULLVMRenderer],
      functools.partial(CPUProgram, self),
      CPUSignal,
      CPUComputeQueue,
    )

  def finalize(self):
    super().finalize()
    _lib.npu_close(self.fd)
