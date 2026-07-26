#pragma once

#include "engine/framework/sampling/torch_random.h"

#include <cstddef>
#include <cstdint>

namespace engine::sampling::detail {

void fill_torch_hip_tensor_iterator_randn_hip(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision);

}  // namespace engine::sampling::detail

// GPU argmax: returns index of max value in logits array on device.
// logits_device: device pointer to float array of size vocab_size.
// result_host: host pointer to receive the argmax index (int32_t).
void gpu_argmax_hip(const float * logits_device, int vocab_size, int32_t * result_host);
