#include "oxcore.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr size_t kMaxArenaAlignment = 4096u;
constexpr uint8_t kOctaveMask = 0x80u;
constexpr uint8_t kSignMask = 0x40u;
constexpr uint8_t kMagnitudeMask = 0x3fu;
constexpr uint64_t kMirror63MaxAbsAccumulatorPerBlock = 1536u;

bool is_power_of_two(size_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case OX_DTYPE_F32:
            return sizeof(float);
        case OX_DTYPE_I8:
        case OX_DTYPE_U8:
            return sizeof(uint8_t);
        case OX_DTYPE_I32:
            return sizeof(int32_t);
        case OX_DTYPE_I64:
            return sizeof(int64_t);
        default:
            return 0u;
    }
}

constexpr int balanced_digit(int value) {
    int remainder = value % 5;
    if (remainder > 2) {
        remainder -= 5;
    } else if (remainder < -2) {
        remainder += 5;
    }
    return remainder;
}

constexpr auto make_balanced_triplet_lookup() {
    std::array<std::array<int8_t, 3>, 125> table{};
    for (int value = -62; value <= 62; ++value) {
        const int q2 = balanced_digit(value);
        const int quotient2 = (value - q2) / 5;
        const int q1 = balanced_digit(quotient2);
        const int q0 = (quotient2 - q1) / 5;
        table[static_cast<size_t>(value + 62)] = {
            static_cast<int8_t>(q0),
            static_cast<int8_t>(q1),
            static_cast<int8_t>(q2)
        };
    }
    return table;
}

constexpr auto kBalancedTripletLookup = make_balanced_triplet_lookup();

int add_five(int value) {
    return value + value + value + value + value;
}

ox_status decode_mirror63(uint8_t code, int q[3], uint8_t *octave) {
    if (code == static_cast<uint8_t>(OXCORE_MIRROR63_SKIP_CODE)) {
        q[0] = 0;
        q[1] = 0;
        q[2] = 0;
        *octave = 0u;
        return OX_SKIP;
    }

    const int magnitude = static_cast<int>(code & kMagnitudeMask);
    const bool negative = (code & kSignMask) != 0u;
    if (magnitude == 63 || (negative && magnitude == 0)) {
        return OX_ERESERVED;
    }

    const int signed_value = negative ? -magnitude : magnitude;
    const auto &triplet =
        kBalancedTripletLookup[static_cast<size_t>(signed_value + 62)];
    q[0] = static_cast<int>(triplet[0]);
    q[1] = static_cast<int>(triplet[1]);
    q[2] = static_cast<int>(triplet[2]);
    *octave = (code & kOctaveMask) != 0u ? 1u : 0u;
    return OX_OK;
}

int64_t scale_by_small_integer_additive(int value, int8_t activation) {
    const int64_t x = static_cast<int64_t>(activation);
    switch (value) {
        case -2:
            return -(x + x);
        case -1:
            return -x;
        case 0:
            return 0;
        case 1:
            return x;
        case 2:
            return x + x;
        default:
            return 0;
    }
}

int64_t dot_decoded_triplet(
    const int q[3], uint8_t octave, const int8_t activations[3]
) {
    int64_t sum = 0;
    sum += scale_by_small_integer_additive(q[0], activations[0]);
    sum += scale_by_small_integer_additive(q[1], activations[1]);
    sum += scale_by_small_integer_additive(q[2], activations[2]);
    if (octave != 0u) {
        sum += sum;
    }
    return sum;
}

bool size_product_overflows(size_t left, size_t right) {
    return right != 0u && left > std::numeric_limits<size_t>::max() / right;
}

uint64_t absolute_i32(int32_t value) {
    const int64_t wide = static_cast<int64_t>(value);
    return wide < 0 ? static_cast<uint64_t>(-wide)
                    : static_cast<uint64_t>(wide);
}

ox_status scale_q15_shift_add_round_even(
    int32_t value, uint16_t coefficient, int32_t *out
) {
    if (out == nullptr || coefficient > 32768u) {
        return OX_EINVAL;
    }

    const bool negative = value < 0;
    const uint64_t magnitude = absolute_i32(value);
    uint64_t accumulator = 0u;
    for (uint32_t bit = 0u; bit <= 15u; ++bit) {
        if ((coefficient & static_cast<uint16_t>(1u << bit)) != 0u) {
            accumulator += magnitude << bit;
        }
    }

    uint64_t quotient = accumulator >> 15u;
    const uint64_t remainder = accumulator & 0x7fffu;
    constexpr uint64_t half = 0x4000u;
    if (remainder > half || (remainder == half && (quotient & 1u) != 0u)) {
        ++quotient;
    }

    if (quotient > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return OX_EOVERFLOW;
    }
    const int64_t signed_value = negative
        ? -static_cast<int64_t>(quotient)
        : static_cast<int64_t>(quotient);
    if (signed_value < std::numeric_limits<int32_t>::min() ||
        signed_value > std::numeric_limits<int32_t>::max()) {
        return OX_EOVERFLOW;
    }
    *out = static_cast<int32_t>(signed_value);
    return OX_OK;
}

ox_status validate_mirror63_weight_array(
    const uint8_t *packed_weights, size_t count
) {
    if (count != 0u && packed_weights == nullptr) {
        return OX_EINVAL;
    }
    for (size_t index = 0u; index < count; ++index) {
        int q[3] = {0, 0, 0};
        uint8_t octave = 0u;
        const ox_status status =
            decode_mirror63(packed_weights[index], q, &octave);
        if (status != OX_OK && status != OX_SKIP) {
            return status;
        }
    }
    return OX_OK;
}

}  // namespace

extern "C" {

uint32_t oxcore_abi_version(void) {
    return OXCORE_ABI_VERSION;
}

const char *oxcore_version(void) {
    return "OxCore 0.1.0";
}

const char *ox_status_name(ox_status status) {
    switch (status) {
        case OX_OK:
            return "ok";
        case OX_SKIP:
            return "skip";
        case OX_EINVAL:
            return "invalid argument";
        case OX_ECAPACITY:
            return "arena capacity exceeded";
        case OX_EOVERFLOW:
            return "integer or size overflow";
        case OX_ERESERVED:
            return "reserved Mirror-63 code";
        case OX_ENUMERIC:
            return "numeric failure";
        default:
            return "unknown OxCore status";
    }
}

ox_status ox_tensor_view_validate(const ox_tensor_view *view) {
    if (view == nullptr || view->rank > OXCORE_MAX_TENSOR_RANK) {
        return OX_EINVAL;
    }
    const size_t item_size = dtype_size(view->dtype);
    if (item_size == 0u || view->offset_bytes > view->storage_bytes) {
        return OX_EINVAL;
    }

    uint64_t element_count = 1u;
    for (uint32_t axis = 0u; axis < view->rank; ++axis) {
        const uint64_t dimension = view->shape[axis];
        if (dimension != 0u &&
            element_count > std::numeric_limits<uint64_t>::max() / dimension) {
            return OX_EOVERFLOW;
        }
        element_count *= dimension;
    }
    if (element_count == 0u) {
        return OX_OK;
    }
    if (view->storage == nullptr || view->storage_bytes == 0u ||
        view->offset_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return OX_EINVAL;
    }

    int64_t minimum_offset = static_cast<int64_t>(view->offset_bytes);
    int64_t maximum_offset = minimum_offset;
    for (uint32_t axis = 0u; axis < view->rank; ++axis) {
        const uint64_t dimension = view->shape[axis];
        const int64_t stride = view->strides_bytes[axis];
        if (dimension <= 1u) {
            continue;
        }
        const uint64_t stride_magnitude = stride < 0
            ? static_cast<uint64_t>(-(stride + 1)) + 1u
            : static_cast<uint64_t>(stride);
        if (stride_magnitude % static_cast<uint64_t>(item_size) != 0u) {
            return OX_EINVAL;
        }
        const uint64_t steps = dimension - 1u;
        if (stride_magnitude != 0u &&
            steps > std::numeric_limits<uint64_t>::max() / stride_magnitude) {
            return OX_EOVERFLOW;
        }
        const uint64_t span = steps * stride_magnitude;
        if (span > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return OX_EOVERFLOW;
        }
        if (stride < 0) {
            if (span > static_cast<uint64_t>(minimum_offset)) {
                return OX_EINVAL;
            }
            minimum_offset -= static_cast<int64_t>(span);
        } else {
            if (span > static_cast<uint64_t>(
                           std::numeric_limits<int64_t>::max() - maximum_offset)) {
                return OX_EOVERFLOW;
            }
            maximum_offset += static_cast<int64_t>(span);
        }
    }

    if (minimum_offset < 0 || maximum_offset < minimum_offset) {
        return OX_EINVAL;
    }
    const uint64_t end_offset =
        static_cast<uint64_t>(maximum_offset) + static_cast<uint64_t>(item_size);
    if (end_offset > static_cast<uint64_t>(view->storage_bytes)) {
        return OX_ECAPACITY;
    }
    return OX_OK;
}

ox_status ox_arena_init(ox_arena *arena, void *buffer, size_t capacity) {
    if (arena == nullptr || buffer == nullptr || capacity == 0u) {
        return OX_EINVAL;
    }
    arena->base = static_cast<uint8_t *>(buffer);
    arena->capacity = capacity;
    arena->used = 0u;
    arena->high_water = 0u;
    return OX_OK;
}

void ox_arena_reset(ox_arena *arena) {
    if (arena == nullptr) {
        return;
    }
    arena->used = 0u;
}

size_t ox_arena_mark(const ox_arena *arena) {
    return arena == nullptr ? 0u : arena->used;
}

ox_status ox_arena_rewind(ox_arena *arena, size_t mark) {
    if (arena == nullptr || arena->base == nullptr || mark > arena->used) {
        return OX_EINVAL;
    }
    arena->used = mark;
    return OX_OK;
}

ox_status ox_arena_alloc(
    ox_arena *arena, size_t size, size_t alignment, void **out
) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (arena == nullptr || arena->base == nullptr || out == nullptr ||
        size == 0u || !is_power_of_two(alignment) ||
        alignment > kMaxArenaAlignment) {
        return OX_EINVAL;
    }

    const uintptr_t base_address = reinterpret_cast<uintptr_t>(arena->base);
    if (arena->used > arena->capacity ||
        base_address > std::numeric_limits<uintptr_t>::max() - arena->used) {
        return OX_EOVERFLOW;
    }
    const uintptr_t current_address = base_address + arena->used;
    if (current_address >
        std::numeric_limits<uintptr_t>::max() - (alignment - 1u)) {
        return OX_EOVERFLOW;
    }
    const uintptr_t aligned_address =
        (current_address + alignment - 1u) &
        ~static_cast<uintptr_t>(alignment - 1u);
    const size_t offset = static_cast<size_t>(aligned_address - base_address);
    if (offset > arena->capacity || size > arena->capacity - offset) {
        return OX_ECAPACITY;
    }

    const size_t new_used = offset + size;
    arena->used = new_used;
    if (new_used > arena->high_water) {
        arena->high_water = new_used;
    }
    *out = reinterpret_cast<void *>(aligned_address);
    return OX_OK;
}

ox_status ox_mirror63_pack_triplet(
    int8_t q0, int8_t q1, int8_t q2, uint8_t octave, uint8_t *out_code
) {
    if (out_code == nullptr || octave > 1u ||
        q0 < -2 || q0 > 2 || q1 < -2 || q1 > 2 || q2 < -2 || q2 > 2) {
        return OX_EINVAL;
    }

    const int q0_value = static_cast<int>(q0);
    const int q1_value = static_cast<int>(q1);
    const int value =
        add_five(add_five(q0_value)) + add_five(q1_value) +
        static_cast<int>(q2);
    const int magnitude = value < 0 ? -value : value;
    uint8_t code = static_cast<uint8_t>(magnitude);
    if (value < 0) {
        code = static_cast<uint8_t>(code | kSignMask);
    }
    if (octave != 0u) {
        code = static_cast<uint8_t>(code | kOctaveMask);
    }
    *out_code = code;
    return OX_OK;
}

ox_status ox_mirror63_unpack_triplet(
    uint8_t code, int8_t out_q[3], uint8_t *out_octave
) {
    if (out_q == nullptr || out_octave == nullptr) {
        return OX_EINVAL;
    }
    int q[3] = {0, 0, 0};
    uint8_t octave = 0u;
    const ox_status status = decode_mirror63(code, q, &octave);
    if (status == OX_OK || status == OX_SKIP) {
        out_q[0] = static_cast<int8_t>(q[0]);
        out_q[1] = static_cast<int8_t>(q[1]);
        out_q[2] = static_cast<int8_t>(q[2]);
        *out_octave = octave;
    }
    return status;
}

ox_status ox_mirror63_dot_i8(
    const uint8_t *packed_blocks,
    const int8_t *activations,
    size_t block_count,
    int64_t *out_scaled_accumulator
) {
    if (out_scaled_accumulator == nullptr ||
        (block_count != 0u && (packed_blocks == nullptr || activations == nullptr))) {
        return OX_EINVAL;
    }
    if (block_count > std::numeric_limits<size_t>::max() / 3u ||
        block_count >
            static_cast<size_t>(std::numeric_limits<int64_t>::max() /
                                static_cast<int64_t>(kMirror63MaxAbsAccumulatorPerBlock))) {
        return OX_EOVERFLOW;
    }

    int64_t accumulator = 0;
    for (size_t block = 0u; block < block_count; ++block) {
        int q[3] = {0, 0, 0};
        uint8_t octave = 0u;
        const ox_status status =
            decode_mirror63(packed_blocks[block], q, &octave);
        if (status == OX_SKIP) {
            continue;
        }
        if (status != OX_OK) {
            return status;
        }

        int64_t contribution = 0;
        const size_t offset = block * 3u;
        contribution += scale_by_small_integer_additive(q[0], activations[offset]);
        contribution += scale_by_small_integer_additive(q[1], activations[offset + 1u]);
        contribution += scale_by_small_integer_additive(q[2], activations[offset + 2u]);
        if (octave != 0u) {
            contribution += contribution;
        }
        accumulator += contribution;
    }

    *out_scaled_accumulator = accumulator;
    return OX_OK;
}

ox_status ox_linear_mirror63_acc_i8(
    const int8_t *input,
    const uint8_t *packed_weights,
    size_t batch,
    size_t in_features,
    size_t out_features,
    int64_t *output
) {
    if (batch == 0u || in_features == 0u || out_features == 0u ||
        input == nullptr || packed_weights == nullptr || output == nullptr) {
        return OX_EINVAL;
    }
    if (in_features > std::numeric_limits<size_t>::max() - 2u) {
        return OX_EOVERFLOW;
    }
    const size_t blocks_per_row = (in_features + 2u) / 3u;
    if (size_product_overflows(batch, in_features) ||
        size_product_overflows(out_features, blocks_per_row) ||
        size_product_overflows(batch, out_features) ||
        in_features > static_cast<size_t>(
            std::numeric_limits<int64_t>::max() / 512)) {
        return OX_EOVERFLOW;
    }

    const size_t weight_count = out_features * blocks_per_row;
    const ox_status validation =
        validate_mirror63_weight_array(packed_weights, weight_count);
    if (validation != OX_OK) {
        return validation;
    }

    for (size_t row = 0u; row < batch; ++row) {
        const size_t input_offset = row * in_features;
        const size_t output_offset = row * out_features;
        for (size_t output_index = 0u; output_index < out_features; ++output_index) {
            int64_t accumulator = 0;
            const size_t weight_offset = output_index * blocks_per_row;
            for (size_t block = 0u; block < blocks_per_row; ++block) {
                int q[3] = {0, 0, 0};
                uint8_t octave = 0u;
                const ox_status status = decode_mirror63(
                    packed_weights[weight_offset + block], q, &octave
                );
                if (status == OX_SKIP) {
                    continue;
                }
                if (status != OX_OK) {
                    return status;
                }

                int8_t block_input[3] = {0, 0, 0};
                const size_t feature_offset = block * 3u;
                for (size_t lane = 0u; lane < 3u; ++lane) {
                    const size_t feature = feature_offset + lane;
                    if (feature < in_features) {
                        block_input[lane] = input[input_offset + feature];
                    }
                }
                accumulator += dot_decoded_triplet(q, octave, block_input);
            }
            output[output_offset + output_index] = accumulator;
        }
    }
    return OX_OK;
}

ox_status ox_causal_depthwise_conv_mirror63_acc_i8(
    const int8_t *input,
    size_t time_steps,
    size_t channels,
    const uint8_t *packed_weights,
    size_t kernel_size,
    int64_t *output
) {
    if (time_steps == 0u || channels == 0u || kernel_size == 0u ||
        input == nullptr || packed_weights == nullptr || output == nullptr) {
        return OX_EINVAL;
    }
    if (kernel_size > std::numeric_limits<size_t>::max() - 2u) {
        return OX_EOVERFLOW;
    }
    const size_t blocks_per_channel = (kernel_size + 2u) / 3u;
    if (size_product_overflows(time_steps, channels) ||
        size_product_overflows(channels, blocks_per_channel) ||
        kernel_size > static_cast<size_t>(
            std::numeric_limits<int64_t>::max() / 512)) {
        return OX_EOVERFLOW;
    }

    const size_t weight_count = channels * blocks_per_channel;
    const ox_status validation =
        validate_mirror63_weight_array(packed_weights, weight_count);
    if (validation != OX_OK) {
        return validation;
    }

    for (size_t time = 0u; time < time_steps; ++time) {
        const size_t output_offset = time * channels;
        for (size_t channel = 0u; channel < channels; ++channel) {
            int64_t accumulator = 0;
            const size_t weight_offset = channel * blocks_per_channel;
            for (size_t block = 0u; block < blocks_per_channel; ++block) {
                int q[3] = {0, 0, 0};
                uint8_t octave = 0u;
                const ox_status status = decode_mirror63(
                    packed_weights[weight_offset + block], q, &octave
                );
                if (status == OX_SKIP) {
                    continue;
                }
                if (status != OX_OK) {
                    return status;
                }

                int8_t block_input[3] = {0, 0, 0};
                const size_t tap_offset = block * 3u;
                for (size_t lane = 0u; lane < 3u; ++lane) {
                    const size_t tap = tap_offset + lane;
                    if (tap >= kernel_size) {
                        continue;
                    }
                    const size_t left_context = kernel_size - 1u - tap;
                    if (time >= left_context) {
                        const size_t source_time = time - left_context;
                        block_input[lane] =
                            input[source_time * channels + channel];
                    }
                }
                accumulator += dot_decoded_triplet(q, octave, block_input);
            }
            output[output_offset + channel] = accumulator;
        }
    }
    return OX_OK;
}

ox_status ox_ssm_scan_q15_i32(
    const int32_t *initial_state,
    const int32_t *injection,
    const uint16_t *decay_q15,
    size_t time_steps,
    size_t batch,
    size_t channels,
    int32_t *all_states,
    int32_t *final_state
) {
    if (time_steps == 0u || batch == 0u || channels == 0u ||
        initial_state == nullptr || injection == nullptr ||
        decay_q15 == nullptr || all_states == nullptr || final_state == nullptr) {
        return OX_EINVAL;
    }
    if (size_product_overflows(batch, channels) ||
        size_product_overflows(time_steps, batch * channels)) {
        return OX_EOVERFLOW;
    }

    const size_t state_count = batch * channels;
    const size_t value_count = time_steps * state_count;

    uint64_t max_initial = 0u;
    for (size_t index = 0u; index < state_count; ++index) {
        const uint64_t magnitude = absolute_i32(initial_state[index]);
        if (magnitude > max_initial) {
            max_initial = magnitude;
        }
    }

    uint64_t max_injection = 0u;
    for (size_t index = 0u; index < value_count; ++index) {
        if (decay_q15[index] > 32768u) {
            return OX_EINVAL;
        }
        const uint64_t magnitude = absolute_i32(injection[index]);
        if (magnitude > max_injection) {
            max_injection = magnitude;
        }
    }

    /* Conservative bound: |decay| <= 1 and nearest-even rounding of a
       nonnegative scale in [0,1] cannot increase the absolute integer state.
       Reject before writing any output. */
    const uint64_t int32_limit =
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    if (max_initial > int32_limit) {
        return OX_EOVERFLOW;
    }
    if (max_injection != 0u &&
        static_cast<uint64_t>(time_steps) >
            (int32_limit - max_initial) / max_injection) {
        return OX_EOVERFLOW;
    }

    for (size_t index = 0u; index < state_count; ++index) {
        final_state[index] = initial_state[index];
    }

    for (size_t time = 0u; time < time_steps; ++time) {
        const size_t time_offset = time * state_count;
        for (size_t index = 0u; index < state_count; ++index) {
            int32_t decayed = 0;
            const ox_status status = scale_q15_shift_add_round_even(
                final_state[index],
                decay_q15[time_offset + index],
                &decayed
            );
            if (status != OX_OK) {
                return status;
            }
            const int64_t updated =
                static_cast<int64_t>(decayed) +
                static_cast<int64_t>(injection[time_offset + index]);
            if (updated < std::numeric_limits<int32_t>::min() ||
                updated > std::numeric_limits<int32_t>::max()) {
                return OX_EOVERFLOW;
            }
            final_state[index] = static_cast<int32_t>(updated);
            all_states[time_offset + index] = final_state[index];
        }
    }
    return OX_OK;
}

ox_status ox_lut_u8_i16(
    const uint8_t *indices, const int16_t table[256],
    size_t length, int16_t *output
) {
    if (indices == nullptr || table == nullptr || output == nullptr ||
        length == 0u) {
        return OX_EINVAL;
    }
    for (size_t index = 0u; index < length; ++index) {
        output[index] = table[indices[index]];
    }
    return OX_OK;
}

ox_status ox_top1_i16(
    const int16_t *scores,
    size_t batch,
    size_t experts,
    uint16_t *selected_expert,
    int16_t *selected_score
) {
    if (scores == nullptr || selected_expert == nullptr ||
        selected_score == nullptr || batch == 0u || experts == 0u ||
        experts > 65536u || size_product_overflows(batch, experts)) {
        return OX_EINVAL;
    }

    for (size_t row = 0u; row < batch; ++row) {
        const size_t offset = row * experts;
        size_t best_index = 0u;
        int16_t best_score = scores[offset];
        for (size_t expert = 1u; expert < experts; ++expert) {
            const int16_t candidate = scores[offset + expert];
            if (candidate > best_score) {
                best_score = candidate;
                best_index = expert;
            }
        }
        selected_expert[row] = static_cast<uint16_t>(best_index);
        selected_score[row] = best_score;
    }
    return OX_OK;
}

ox_status ox_fwht_i64(int64_t *values, size_t length) {
    if (values == nullptr || length == 0u || !is_power_of_two(length)) {
        return OX_EINVAL;
    }
    if (length >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return OX_EOVERFLOW;
    }

    uint64_t max_abs = 0u;
    for (size_t index = 0u; index < length; ++index) {
        const int64_t value = values[index];
        const uint64_t magnitude = value < 0
            ? static_cast<uint64_t>(-(value + 1)) + 1u
            : static_cast<uint64_t>(value);
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }
    const uint64_t safe_limit =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
        static_cast<uint64_t>(length);
    if (max_abs > safe_limit) {
        return OX_EOVERFLOW;
    }

    for (size_t span = 1u; span < length; span *= 2u) {
        const size_t group_size = span * 2u;
        for (size_t base = 0u; base < length; base += group_size) {
            for (size_t offset = 0u; offset < span; ++offset) {
                const int64_t left = values[base + offset];
                const int64_t right = values[base + span + offset];
                values[base + offset] = left + right;
                values[base + span + offset] = left - right;
            }
        }
    }
    return OX_OK;
}

ox_status ox_sinkhorn4(
    const double logits[16], double out_matrix[16], uint32_t iterations
) {
    if (logits == nullptr || out_matrix == nullptr ||
        iterations == 0u || iterations > 10000u) {
        return OX_EINVAL;
    }

    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t index = 0u; index < 16u; ++index) {
        if (!std::isfinite(logits[index])) {
            return OX_ENUMERIC;
        }
        if (logits[index] > maximum) {
            maximum = logits[index];
        }
    }

    double matrix[16] = {};
    for (size_t index = 0u; index < 16u; ++index) {
        matrix[index] = std::exp(logits[index] - maximum);
        if (!std::isfinite(matrix[index])) {
            return OX_ENUMERIC;
        }
    }

    for (uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        for (size_t row = 0u; row < 4u; ++row) {
            double sum = 0.0;
            for (size_t column = 0u; column < 4u; ++column) {
                sum += matrix[row * 4u + column];
            }
            if (!std::isfinite(sum) || sum <= 0.0) {
                return OX_ENUMERIC;
            }
            for (size_t column = 0u; column < 4u; ++column) {
                matrix[row * 4u + column] /= sum;
            }
        }
        for (size_t column = 0u; column < 4u; ++column) {
            double sum = 0.0;
            for (size_t row = 0u; row < 4u; ++row) {
                sum += matrix[row * 4u + column];
            }
            if (!std::isfinite(sum) || sum <= 0.0) {
                return OX_ENUMERIC;
            }
            for (size_t row = 0u; row < 4u; ++row) {
                matrix[row * 4u + column] /= sum;
            }
        }
    }

    for (size_t index = 0u; index < 16u; ++index) {
        out_matrix[index] = matrix[index];
    }
    return OX_OK;
}

ox_status ox_normalize_f32(
    const float *input, float *output, size_t length, float epsilon
) {
    if (input == nullptr || output == nullptr || length == 0u ||
        !std::isfinite(epsilon) || epsilon <= 0.0f) {
        return OX_EINVAL;
    }

    double sum_squares = 0.0;
    for (size_t index = 0u; index < length; ++index) {
        if (!std::isfinite(input[index])) {
            return OX_ENUMERIC;
        }
        const double value = static_cast<double>(input[index]);
        sum_squares += value * value;
    }
    const double denominator = std::sqrt(sum_squares) +
                               static_cast<double>(epsilon);
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        return OX_ENUMERIC;
    }

    for (size_t index = 0u; index < length; ++index) {
        output[index] = static_cast<float>(
            static_cast<double>(input[index]) / denominator
        );
    }
    return OX_OK;
}

}  // extern "C"
