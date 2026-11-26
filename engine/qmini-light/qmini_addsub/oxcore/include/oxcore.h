#ifndef QMINI_ADDSUB_OXCORE_H
#define QMINI_ADDSUB_OXCORE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define OXCORE_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define OXCORE_API __attribute__((visibility("default")))
#else
#define OXCORE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OXCORE_ABI_VERSION 1u
#define OXCORE_MAX_TENSOR_RANK 8u
#define OXCORE_DEFAULT_ALIGNMENT 64u
#define OXCORE_MIRROR63_SKIP_CODE 0x3fu

typedef enum ox_status {
    OX_OK = 0,
    OX_SKIP = 1,
    OX_EINVAL = 2,
    OX_ECAPACITY = 3,
    OX_EOVERFLOW = 4,
    OX_ERESERVED = 5,
    OX_ENUMERIC = 6
} ox_status;

typedef enum ox_dtype {
    OX_DTYPE_F32 = 1,
    OX_DTYPE_I8 = 2,
    OX_DTYPE_I32 = 3,
    OX_DTYPE_I64 = 4,
    OX_DTYPE_U8 = 5
} ox_dtype;

typedef struct ox_tensor_view {
    void *storage;
    size_t storage_bytes;
    size_t offset_bytes;
    uint32_t dtype;
    uint32_t rank;
    uint64_t shape[OXCORE_MAX_TENSOR_RANK];
    int64_t strides_bytes[OXCORE_MAX_TENSOR_RANK];
} ox_tensor_view;

typedef struct ox_arena {
    uint8_t *base;
    size_t capacity;
    size_t used;
    size_t high_water;
} ox_arena;

OXCORE_API uint32_t oxcore_abi_version(void);
OXCORE_API const char *oxcore_version(void);
OXCORE_API const char *ox_status_name(ox_status status);

OXCORE_API ox_status ox_tensor_view_validate(const ox_tensor_view *view);

OXCORE_API ox_status ox_arena_init(ox_arena *arena, void *buffer, size_t capacity);
OXCORE_API void ox_arena_reset(ox_arena *arena);
OXCORE_API size_t ox_arena_mark(const ox_arena *arena);
OXCORE_API ox_status ox_arena_rewind(ox_arena *arena, size_t mark);
OXCORE_API ox_status ox_arena_alloc(
    ox_arena *arena, size_t size, size_t alignment, void **out
);

OXCORE_API ox_status ox_mirror63_pack_triplet(
    int8_t q0, int8_t q1, int8_t q2, uint8_t octave, uint8_t *out_code
);
OXCORE_API ox_status ox_mirror63_unpack_triplet(
    uint8_t code, int8_t out_q[3], uint8_t *out_octave
);
OXCORE_API ox_status ox_mirror63_dot_i8(
    const uint8_t *packed_blocks,
    const int8_t *activations,
    size_t block_count,
    int64_t *out_scaled_accumulator
);

/* Input: [batch, in_features]. Packed rows: [out_features, ceil(in_features/3)].
 * Output: [batch, out_features]. Returns the unscaled integer numerator. */
OXCORE_API ox_status ox_linear_mirror63_acc_i8(
    const int8_t *input,
    const uint8_t *packed_weights,
    size_t batch,
    size_t in_features,
    size_t out_features,
    int64_t *output
);

/* Causal depthwise input/output: contiguous [time_steps, channels].
 * Weight bytes: [channels, ceil(kernel_size/3)]. Tap 0 is oldest; final tap
 * multiplies the current time step. Missing left-context is exact zero pad. */
OXCORE_API ox_status ox_causal_depthwise_conv_mirror63_acc_i8(
    const int8_t *input,
    size_t time_steps,
    size_t channels,
    const uint8_t *packed_weights,
    size_t kernel_size,
    int64_t *output
);

/* Q15 diagonal SSM recurrence:
 * state[t] = round_to_nearest_even(decay_q15[t] * state[t-1] / 32768)
 *            + injection[t].
 * All buffers are contiguous [time,batch,channels], except initial/final
 * state [batch,channels]. Decay coefficients must be in [0,32768].
 * Scaling uses shift-add only; no integer multiply is used by this operator.
 */
OXCORE_API ox_status ox_ssm_scan_q15_i32(
    const int32_t *initial_state,
    const int32_t *injection,
    const uint16_t *decay_q15,
    size_t time_steps,
    size_t batch,
    size_t channels,
    int32_t *all_states,
    int32_t *final_state
);

/* Generic activation/gate LUT lookup. One uint8 index maps to one int16 value. */
OXCORE_API ox_status ox_lut_u8_i16(
    const uint8_t *indices, const int16_t table[256],
    size_t length, int16_t *output
);

/* Stable Top-1: scores are [batch, experts], ties select the lowest expert id. */
OXCORE_API ox_status ox_top1_i16(
    const int16_t *scores,
    size_t batch,
    size_t experts,
    uint16_t *selected_expert,
    int16_t *selected_score
);

OXCORE_API ox_status ox_fwht_i64(int64_t *values, size_t length);
OXCORE_API ox_status ox_sinkhorn4(
    const double logits[16], double out_matrix[16], uint32_t iterations
);
OXCORE_API ox_status ox_normalize_f32(
    const float *input, float *output, size_t length, float epsilon
);

#ifdef __cplusplus
}
#endif

#endif
