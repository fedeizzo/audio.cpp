#include "torch_random_hip_runtime.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace engine::sampling::detail {
namespace {

constexpr uint32_t kPhiloxM0 = 0xD2511F53U;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57U;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9U;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85U;
constexpr float kInvTwoPow32 = 2.3283064365386963e-10F;
constexpr float kInvTwoPow32TwoPi = 1.4629180792671596e-09F;

struct Philox4 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

void check_hip(hipError_t status, const char * label) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(label) + ": " + hipGetErrorString(status));
    }
}

uint64_t tensor_iterator_stride(uint64_t total_elements, const TorchCudaSamplingPolicy & policy) {
    constexpr uint64_t block_size = 256;
    uint64_t grid = (total_elements + block_size - 1) / block_size;
    uint64_t blocks_per_sm = static_cast<uint64_t>(policy.max_threads_per_multiprocessor) / block_size;
    if (blocks_per_sm == 0) {
        blocks_per_sm = 1;
    }
    const uint64_t grid_cap = static_cast<uint64_t>(policy.multiprocessor_count) * blocks_per_sm;
    grid = std::max<uint64_t>(1, std::min(grid_cap, grid));
    return block_size * grid;
}

__device__ void mul_hi_lo(uint32_t lhs, uint32_t rhs, uint32_t & hi, uint32_t & lo) {
    const uint64_t product = static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs);
    lo = static_cast<uint32_t>(product);
    hi = static_cast<uint32_t>(product >> 32U);
}

__device__ Philox4 philox_round(Philox4 counter, uint32_t key0, uint32_t key1) {
    uint32_t hi0 = 0;
    uint32_t lo0 = 0;
    uint32_t hi1 = 0;
    uint32_t lo1 = 0;
    mul_hi_lo(kPhiloxM0, counter.x, hi0, lo0);
    mul_hi_lo(kPhiloxM1, counter.z, hi1, lo1);
    return Philox4{
        hi1 ^ counter.y ^ key0,
        lo1,
        hi0 ^ counter.w ^ key1,
        lo0,
    };
}

__device__ Philox4 philox_4x32_10(Philox4 counter, uint64_t seed) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
    for (int round = 0; round < 10; ++round) {
        counter = philox_round(counter, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

__device__ void box_muller(uint32_t uniform0, uint32_t uniform1, float & normal0, float & normal1) {
    const float radius_input =
        static_cast<float>(uniform0) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
    const float angle =
        static_cast<float>(uniform1) * kInvTwoPow32TwoPi + (kInvTwoPow32TwoPi * 0.5F);
    const float radius = sqrtf(-2.0F * logf(radius_input));
    sincosf(angle, &normal0, &normal1);
    normal0 *= radius;
    normal1 *= radius;
}

__device__ float round_to_bfloat16(float value) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = value;
    cvt.u += 0x7FFFU + ((cvt.u >> 16U) & 1U);
    cvt.u &= 0xFFFF0000U;
    return cvt.f;
}

__global__ void fill_tensor_iterator_randn_kernel(
    float * output,
    uint64_t total,
    uint64_t seed,
    uint64_t offset_blocks,
    uint64_t stride,
    uint64_t loop_count,
    int precision) {
    constexpr uint64_t unroll_factor = 4;
    const uint64_t sequence = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence >= stride) {
        return;
    }
    for (uint64_t loop_index = 0; loop_index < loop_count; ++loop_index) {
        const uint64_t first_index = loop_index * unroll_factor * stride + sequence;
        if (first_index >= total) {
            continue;
        }
        const Philox4 random = philox_4x32_10(
            Philox4{
                static_cast<uint32_t>(offset_blocks + loop_index),
                static_cast<uint32_t>((offset_blocks + loop_index) >> 32U),
                static_cast<uint32_t>(sequence),
                static_cast<uint32_t>(sequence >> 32U),
            },
            seed);
        float normal0 = 0.0F;
        float normal1 = 0.0F;
        float normal2 = 0.0F;
        float normal3 = 0.0F;
        box_muller(random.x, random.y, normal0, normal1);
        box_muller(random.z, random.w, normal2, normal3);
        if (precision == static_cast<int>(TorchRandnPrecision::BFloat16)) {
            normal0 = round_to_bfloat16(normal0);
            normal1 = round_to_bfloat16(normal1);
            normal2 = round_to_bfloat16(normal2);
            normal3 = round_to_bfloat16(normal3);
        }
        output[static_cast<size_t>(first_index)] = normal0;
        const uint64_t second_index = first_index + stride;
        if (second_index < total) {
            output[static_cast<size_t>(second_index)] = normal1;
        }
        const uint64_t third_index = second_index + stride;
        if (third_index < total) {
            output[static_cast<size_t>(third_index)] = normal2;
        }
        const uint64_t fourth_index = third_index + stride;
        if (fourth_index < total) {
            output[static_cast<size_t>(fourth_index)] = normal3;
        }
    }
}

}  // namespace

void fill_torch_hip_tensor_iterator_randn_hip(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision) {
    if (policy.multiprocessor_count <= 0 || policy.max_threads_per_multiprocessor <= 0) {
        throw std::invalid_argument("torch HIP TensorIterator randn fast path requires HIP device properties");
    }
    const uint64_t total = static_cast<uint64_t>(count);
    const uint64_t stride = tensor_iterator_stride(total, policy);
    const uint64_t loop_count = (total + stride * 4 - 1) / (stride * 4);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((stride + threads - 1) / threads);

    check_hip(hipSetDevice(policy.cuda_device_index), "hipSetDevice");

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");

    float * device_output = nullptr;
    check_hip(hipMalloc(&device_output, count * sizeof(float)), "hipMalloc torch random output");
    (void)hipHostRegister(output, count * sizeof(float), hipHostRegisterMapped);

    try {
        hipLaunchKernelGGL(
            fill_tensor_iterator_randn_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            stream,
            device_output,
            total,
            seed,
            offset_blocks,
            stride,
            loop_count,
            static_cast<int>(precision));
        check_hip(hipGetLastError(), "fill_tensor_iterator_randn_kernel");
        check_hip(hipMemcpyAsync(output, device_output, count * sizeof(float), hipMemcpyDeviceToHost, stream), "hipMemcpyAsync torch random output");
        check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize stream");
    } catch (...) {
        (void)hipHostUnregister(output);
        (void)hipFree(device_output);
        (void)hipStreamDestroy(stream);
        throw;
    }
    (void)hipHostUnregister(output);
    (void)hipFree(device_output);
    (void)hipStreamDestroy(stream);
}


// Single-block argmax: each thread strides through the array,
// then shared-memory tree reduction produces the global argmax.
__global__ void gpu_argmax_kernel(const float * __restrict__ logits, int vocab_size, int32_t * __restrict__ result) {
    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    float best_val = -1e30f;
    int   best_idx = -1;
    for (int i = tid; i < vocab_size; i += block_size) {
        float v = logits[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();
    
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[0] = s_idx[0];
    }
}

void gpu_argmax_hip(const float * logits_device, int vocab_size, int32_t * result_host) {
    if (logits_device == nullptr || vocab_size <= 0 || result_host == nullptr) {
        throw std::runtime_error("gpu_argmax_hip: invalid arguments");
    }
    
    hipStream_t stream = 0;
    const int block_size = 1024;
    
    int32_t * result_dev = nullptr;
    check_hip(hipMalloc(&result_dev, sizeof(int32_t)), "gpu_argmax hipMalloc");
    
    hipLaunchKernelGGL(gpu_argmax_kernel, dim3(1), dim3(block_size), 0, stream,
                       logits_device, vocab_size, result_dev);
    check_hip(hipGetLastError(), "gpu_argmax_kernel");
    
    check_hip(hipMemcpy(result_host, result_dev, sizeof(int32_t), hipMemcpyDeviceToHost), "gpu_argmax hipMemcpy result");
    check_hip(hipFree(result_dev), "gpu_argmax hipFree");
}

}  // namespace engine::sampling::detail
