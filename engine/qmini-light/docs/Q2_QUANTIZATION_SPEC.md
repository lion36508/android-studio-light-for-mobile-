# Q-Mini production Q2 quantization contract

**Status:** normative for all newly exported Q2-v2 artifacts. The older Q2-v1 artifact format remains readable for compatibility; new exports must not emit v1.

## 1. Scope

Q2 applies to production matrix weights with two bits per symbol. It does not describe activation quantization or non-matrix vectors:

- rank-2+ matrix weights: Q2, four symbols per byte;
- activations: signed INT8 with per-token AbsMax scales;
- biases and normalization/vector parameters: FP16 in packed inference archives;
- row scales: FP32;
- accumulator: INT64 in the native projection kernel, converted to FP32 for scale/bias application.

Ternary, INT4 and rowwise INT8 weight quantizers are experimental diagnostic helpers only. They must not be substituted into the production QAT/export/runtime path by default.

## 2. Q2-v2 normalized codebook

Two-bit symbols are packed least-significant pair first within each byte:

| Bits | Normalized value |
|---|---:|
| 00 | -1.0 |
| 01 | -0.5 |
| 10 | +0.5 |
| 11 | +1.0 |

The effective reconstructed matrix weight is:

\[
\widehat{w}_{r,i}=c_{r,i}s_r
\]

where c is the normalized codebook value and the per-output-row scale is:

\[
s_r=2\cdot\operatorname{mean}_i(|w_{r,i}|)
\]

This means:
- negative/positive larger-magnitude entries map to -1/+1;
- smaller-magnitude entries map to -0.5/+0.5;
- exact zero entries in a non-zero row map to +0.5 times the row scale because the production codebook contains no zero symbol;
- an entirely zero row has scale exactly zero and reconstructs to exact zeros regardless of its stored symbols.

The normalized fake quantizer uses the threshold |w|/s >= 0.75 to select magnitude 1, otherwise 0.5. Since s=2*mean(|w|), this is the same decision boundary as the legacy v1 |w|/mean(|w|)>=1.5, but the symbols are normalized and the scale is doubled. For threshold decisions only, the QAT path and packer clamp the divisor to 1e-8; the stored row scale remains the exact 2*mean value, including zero or a tiny non-zero value. The resulting effective QAT weights are unchanged for non-zero rows while the packed representation is explicit and consistently normalized.

Non-finite matrix weights are rejected. For a zero row, no division by zero is performed.

## 3. Legacy Q2-v1 compatibility

Existing qmini-packed-v1 archives remain readable using their original codebook and scale semantics:

| Bits | Q2-v1 legacy value |
|---|---:|
| 00 | -2 |
| 01 | -1 |
| 10 | +1 |
| 11 | +2 |

Legacy v1 row scales are mean(abs(original row)), clamped to a positive epsilon; they must remain strictly positive. Do not reinterpret v1 bytes using the v2 normalized table or v2 scale formula.

All newly written archives use the qmini-packed-v2 metadata format and the Q2-v2 contract below. The loader dispatches on the metadata version rather than guessing from scale magnitudes.

## 4. Q2-v2 archive metadata

Every new archive declares:

    {
      "format": "qmini-packed-v2",
      "q2_contract": {
        "version": 2,
        "codebook": {
          "00": -1.0,
          "01": -0.5,
          "10": 0.5,
          "11": 1.0
        },
        "row_scale": "2 * mean(abs(original_output_row))",
        "all_zero_row_scale": 0.0
      }
    }

The metadata is normative. A v2 archive with missing, altered or contradictory contract metadata must fail closed. The loader must validate packed shape, row-scale dtype/shape, finite scales and non-negative v2 scales before installing buffers.

## 5. Native projection contract

Native projection receives:
- INT8 activations;
- UINT8 two-bit packed weight bytes;
- FP32 row scales;
- FP32 activation scales;
- optional FP32 bias;
- an explicit Q2 format version (1 or 2).

The inner product uses integer half-units. For Q2-v2, decoded integer values are [-2,-1,+1,+2], then multiplied by 0.5 * activation_scale * row_scale. For Q2-v1, decoded integer values are [-4,-2,+2,+4], also multiplied by 0.5, reproducing the legacy [-2,-1,+1,+2] values. V2 scale zero is valid and yields a zero weighted contribution. Activation scales must remain finite and positive.

The versioned exported C ABI symbol is qmini_linear_i8_v2; older libraries lacking that symbol must be rejected with a rebuild instruction. Both scalar and ARM NEON paths use the same codebook/version semantics and must agree with the Python dequantized reference.

## 6. Required regression coverage

- exact four-symbol v2 codebook and normalized reconstruction;
- row scale equals 2 * mean(abs(original row));
- zero entries in a non-zero row use the positive half-symbol;
- all-zero rows use scale zero and reconstruct exactly zero;
- v1 archives decode with v1 table/scales;
- v2 archives declare and validate the exact metadata contract;
- full export/reconstruct/load parity, including embeddings and depthwise convolution;
- native projection parity for v1 and v2, including zero-scale v2 rows;
- scalar and NEON kernels share the versioned ABI.

Q2 file size or tensor reconstruction tests do not prove assistant quality. Use an independently trained Q-Mini checkpoint for held-out NLL/task-quality/runtime ablations before claiming that Q2 is a suitable 5B or handset format.
