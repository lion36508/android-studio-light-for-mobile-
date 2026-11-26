# OxCore — standalone native core for Q-Mini Add/Sub

OxCore lives **inside** the new Q-Mini Add/Sub model package. It is not a new
model family, repository, tokenizer, checkpoint format, or training dataset.
The old Q-Mini Light Q2-v2 implementation remains separate and unchanged.

## Current v0.2 scope

The first release is a small C ABI library with a Python `ctypes` bridge:

- C++20 tensor-view metadata with storage-bound and signed-stride checks.
- Caller-owned, fixed-capacity arena with 64-byte default alignment, explicit
  marks/rewinds, and high-water accounting. The native core never calls
  `malloc`/`free` or owns the arena buffer.
- Mirror-63 balanced-base-5 triplet encoding: 250 valid code/octave pairs and
  exactly six reserved byte values. `0x3f` is the zero-block skip sentinel;
  the other five escape values are rejected.
- Mirror-63 int8 dot accumulator and row-wise integer linear projection.
  They use additions, subtractions and small-coefficient additions for the
  five integer q symbols. The linear operator supports batches, multiple
  output rows, partial final triplets and returns the exact integer numerator;
  the caller applies the final scale and division by two once.
- Causal depthwise convolution for contiguous time-major int8 input and
  per-channel Mirror-63 kernels. Tap zero is oldest, the final tap is current,
  and missing left context is exact zero padding. It returns int64 numerators
  without intermediate shifts.
- Q15 diagonal SSM scan over int32 state and injection. The decay scale is
  implemented with binary shift-add accumulation (no integer multiply in that
  scaling kernel), nearest-even rounding, a caller-supplied initial cache,
  per-token states and a returned final cache. A conservative int32 overflow
  preflight runs before output writes.
- In-place, unnormalized integer FWHT with overflow preflight.
- Stable 4x4 Sinkhorn projection and float32 L2 normalization.
- Python bridge with no ML framework dependency and no import-time compilation.

Build the shared library:

```bash
bash scripts/build_oxcore.sh /tmp/libqmini_oxcore.so
export QMINI_OXCORE_LIBRARY=/tmp/libqmini_oxcore.so
```

Run the native AddressSanitizer/UndefinedBehaviorSanitizer checks:

```bash
bash scripts/test_oxcore_sanitizers.sh
```

Then use it explicitly:

```python
from qmini_addsub.oxcore import OxCoreNative, StaticArena

ox = OxCoreNative.load()
print(ox.version)
packed = ox.mirror63_pack((2, -1, 0), octave=1)
decoded = ox.mirror63_unpack(packed)
numerator = ox.mirror63_dot_i8([packed], [3, -4, 9])
arena = StaticArena(ox, 64 * 1024)
scratch_address = arena.alloc(4096)
```

The current Add/Sub PyTorch trainer and reference generation path do not
automatically switch to OxCore. This is deliberate: the native format and
kernels must pass parity, numeric-stability, and measured benchmark gates
before replacing any existing training or inference operation.

## Architecture boundaries

1. **Native CPU ABI (implemented v0.2):** tensor descriptors, fixed arena,
   Mirror-63 dot/linear/causal-depthwise operators, Q15 SSM scan, FWHT, Sinkhorn and L2 normalization.
2. **Python facade (implemented v0.1):** explicit library loading and typed
   wrappers. The shared library is built separately; importing this package
   does not import PyTorch or compile code.
3. **Graph/autograd (not implemented):** lazy IR, kernel fusion, reverse-mode
   differentiation, checkpoint/recompute policies, hysteresis STE and Muon.
4. **Accelerator codegen (not implemented):** ARM NEON/AVX2 kernels, CUDA PTX,
   OpenCL, Vulkan or Triton. ARM64 cross-compilation and real-device runtime
   parity are separate gates.

## Non-claims

- OxCore is not yet a replacement for PyTorch training or the Add/Sub model.
- It does not yet implement a full tensor algebra, graph compiler, optimizer,
  recurrent-thinking engine or end-to-end model runtime. The Q15 SSM primitive
  accepts precomputed decay and injection tensors; it does not yet replace the
  model's drive/decay/gate projections or full GatedDiagonalStateSpace block.
- It does not establish a speedup, lower training memory, stronger reasoning,
  or improved English quality. Those require matched, reproducible benchmarks.
- The packed Mirror-63 primitive is experimental and does not change the
  existing five-symbol QAT/checkpoint format.
