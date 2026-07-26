#include "istft_hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::audio::detail {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void check_hip(hipError_t status, const char * label) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(label) + ": " + hipGetErrorString(status));
    }
}

void check_hipfft(hipfftResult status, const char * label) {
    if (status != HIPFFT_SUCCESS) {
        throw std::runtime_error(std::string(label) + ": hipfft status " + std::to_string(static_cast<int>(status)));
    }
}

struct HipTimer {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    hipStream_t stream = nullptr;

    explicit HipTimer(hipStream_t st = nullptr) : stream(st) {
        check_hip(hipEventCreate(&start), "hipEventCreate start");
        check_hip(hipEventCreate(&stop), "hipEventCreate stop");
    }

    ~HipTimer() {
        if (start != nullptr) {
            (void)hipEventDestroy(start);
        }
        if (stop != nullptr) {
            (void)hipEventDestroy(stop);
        }
    }

    void begin() {
        check_hip(hipEventRecord(start, stream), "hipEventRecord start");
    }

    double end_ms(const char * label) {
        check_hip(hipEventRecord(stop, stream), "hipEventRecord stop");
        check_hip(hipEventSynchronize(stop), label);
        float ms = 0.0F;
        check_hip(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
        return static_cast<double>(ms);
    }
};

__global__ void build_spectrum_kernel(
    const float * head,
    hipfftComplex * spectrum,
    int frames,
    int out_dim,
    int freq_bins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = frames * freq_bins;
    if (index >= total) {
        return;
    }
    const int frame = index / freq_bins;
    const int freq = index - frame * freq_bins;
    const float * row = head + frame * out_dim;
    const float mag = fminf(expf(row[freq]), 100.0F);
    const float phase = row[freq_bins + freq];
    float sin_value = 0.0F;
    float cos_value = 0.0F;
    sincosf(phase, &sin_value, &cos_value);
    spectrum[index] = make_hipFloatComplex(mag * cos_value, mag * sin_value);
}

__global__ void fused_overlap_add_normalize_kernel(
    const float * framed,
    const float * window,
    float * audio,
    int frames,
    int n_fft,
    int hop_length,
    int samples,
    int pad) {
    const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= samples) {
        return;
    }

    const int target_pos = sample_idx + pad;
    float accumulated_signal = 0.0F;
    float accumulated_window_sq = 0.0F;

    // Determine the range of overlapping FFT frames for this target_pos sample
    const int min_frame = max(0, (target_pos - n_fft + hop_length) / hop_length);
    const int max_frame = min(frames - 1, target_pos / hop_length);

    for (int frame = min_frame; frame <= max_frame; ++frame) {
        const int frame_offset = target_pos - frame * hop_length;
        if (frame_offset >= 0 && frame_offset < n_fft) {
            const float w = window[frame_offset];
            const float val = framed[frame * n_fft + frame_offset];
            accumulated_signal += val * w;
            accumulated_window_sq += w * w;
        }
    }

    const float norm_factor = static_cast<float>(n_fft);
    if (accumulated_window_sq > 1e-8F) {
        audio[sample_idx] = (accumulated_signal / norm_factor) / accumulated_window_sq;
    } else {
        audio[sample_idx] = 0.0F;
    }
}

int blocks_for(int count) {
    constexpr int kThreads = 256;
    return (count + kThreads - 1) / kThreads;
}

}  // namespace

class HipIstftRuntime::Impl {
public:
    explicit Impl(const HipIstftRuntimeConfig & config)
        : config_(config),
          freq_bins_(config.n_fft / 2 + 1),
          pad_((config.n_fft - config.hop_length) / 2),
          output_size_((config.frames - 1) * config.hop_length + config.n_fft),
          samples_(output_size_ - 2 * pad_) {
        if (config_.frames <= 0 || config_.n_fft <= 0 || config_.hop_length <= 0) {
            throw std::runtime_error("HIP ISTFT requires positive shape parameters");
        }
        if (config_.out_dim != freq_bins_ * 2) {
            throw std::runtime_error("HIP ISTFT expects log-magnitude and phase halves");
        }
        if (samples_ <= 0) {
            throw std::runtime_error("HIP ISTFT produced non-positive sample count");
        }

        check_hip(hipSetDevice(config_.device), "hipSetDevice");
        check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreateWithFlags");

        check_hip(hipMalloc(&head_, bytes(config_.frames * config_.out_dim)), "hipMalloc head");
        check_hip(hipMalloc(&window_, bytes(config_.n_fft)), "hipMalloc window");
        check_hip(hipMalloc(&spectrum_, sizeof(hipfftComplex) * static_cast<size_t>(config_.frames * freq_bins_)), "hipMalloc spectrum");
        check_hip(hipMalloc(&framed_, bytes(config_.frames * config_.n_fft)), "hipMalloc framed");
        check_hip(hipMalloc(&audio_, bytes(samples_)), "hipMalloc audio");

        int n[] = {static_cast<int>(config_.n_fft)};
        int inembed[] = {static_cast<int>(freq_bins_)};
        int onembed[] = {static_cast<int>(config_.n_fft)};
        check_hipfft(
            hipfftPlanMany(
                &plan_,
                1,
                n,
                inembed,
                1,
                static_cast<int>(freq_bins_),
                onembed,
                1,
                static_cast<int>(config_.n_fft),
                HIPFFT_C2R,
                static_cast<int>(config_.frames)),
            "hipfftPlanMany");

        check_hipfft(hipfftSetStream(plan_, stream_), "hipfftSetStream");
    }

    ~Impl() {
        if (plan_ != 0) {
            (void)hipfftDestroy(plan_);
        }
        (void)hipFree(audio_);
        (void)hipFree(framed_);
        (void)hipFree(spectrum_);
        (void)hipFree(window_);
        (void)hipFree(head_);
        if (stream_ != nullptr) {
            (void)hipStreamDestroy(stream_);
        }
    }

    HipIstftRuntimeResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window) {
        if (static_cast<int64_t>(log_magnitude_phase.size()) != config_.frames * config_.out_dim) {
            throw std::runtime_error("HIP ISTFT input size mismatch");
        }
        if (static_cast<int64_t>(window.size()) != config_.n_fft) {
            throw std::runtime_error("HIP ISTFT window size mismatch");
        }

        HipIstftRuntimeResult result;
        auto & timing = result.timing;
        const auto total_start = Clock::now();
        check_hip(hipSetDevice(config_.device), "hipSetDevice");

        auto timing_start = Clock::now();
        if (window_cache_ != window) {
            check_hip(hipMemcpyAsync(window_, window.data(), bytes(config_.n_fft), hipMemcpyHostToDevice, stream_), "hipMemcpyAsync window");
            window_cache_ = window;
        }
        check_hip(
            hipMemcpyAsync(head_, log_magnitude_phase.data(), bytes(config_.frames * config_.out_dim), hipMemcpyHostToDevice, stream_),
            "hipMemcpyAsync head");
        timing.input_upload_ms = elapsed_ms(timing_start, Clock::now());

        HipTimer timer(stream_);
        timer.begin();
        hipLaunchKernelGGL(
            build_spectrum_kernel,
            dim3(blocks_for(static_cast<int>(config_.frames * freq_bins_))),
            dim3(256),
            0,
            stream_,
            head_,
            spectrum_,
            static_cast<int>(config_.frames),
            static_cast<int>(config_.out_dim),
            static_cast<int>(freq_bins_));
        check_hip(hipGetLastError(), "build_spectrum_kernel");
        timing.spectrum_kernel_ms = timer.end_ms("build_spectrum_kernel synchronize");

        timer.begin();
        check_hipfft(hipfftExecC2R(plan_, spectrum_, framed_), "hipfftExecC2R");
        timing.fft_inverse_ms = timer.end_ms("hipfftExecC2R synchronize");

        timer.begin();
        hipLaunchKernelGGL(
            fused_overlap_add_normalize_kernel,
            dim3(blocks_for(static_cast<int>(samples_))),
            dim3(256),
            0,
            stream_,
            framed_,
            window_,
            audio_,
            static_cast<int>(config_.frames),
            static_cast<int>(config_.n_fft),
            static_cast<int>(config_.hop_length),
            static_cast<int>(samples_),
            static_cast<int>(pad_));
        check_hip(hipGetLastError(), "fused_overlap_add_normalize_kernel");
        timing.overlap_add_ms = timer.end_ms("fused_overlap_add_normalize_kernel synchronize");
        timing.normalize_ms = 0.0; // Fused into single kernel launch

        timing_start = Clock::now();
        result.audio.resize(static_cast<size_t>(samples_));
        (void)hipHostRegister(result.audio.data(), bytes(samples_), hipHostRegisterMapped);
        check_hip(hipMemcpyAsync(result.audio.data(), audio_, bytes(samples_), hipMemcpyDeviceToHost, stream_), "hipMemcpyAsync audio");
        check_hip(hipStreamSynchronize(stream_), "hipStreamSynchronize audio");
        (void)hipHostUnregister(result.audio.data());
        timing.audio_read_ms = elapsed_ms(timing_start, Clock::now());
        timing.total_ms = elapsed_ms(total_start, Clock::now());
        return result;
    }

private:
    static size_t bytes(int64_t elements) {
        return sizeof(float) * static_cast<size_t>(elements);
    }

    HipIstftRuntimeConfig config_;
    int64_t freq_bins_ = 0;
    int64_t pad_ = 0;
    int64_t output_size_ = 0;
    int64_t samples_ = 0;
    float * head_ = nullptr;
    float * window_ = nullptr;
    hipfftComplex * spectrum_ = nullptr;
    float * framed_ = nullptr;
    float * audio_ = nullptr;
    hipfftHandle plan_ = 0;
    hipStream_t stream_ = nullptr;
    std::vector<float> window_cache_;
};

HipIstftRuntime::HipIstftRuntime(const HipIstftRuntimeConfig & config)
    : impl_(std::make_unique<Impl>(config)) {}

HipIstftRuntime::~HipIstftRuntime() = default;

HipIstftRuntime::HipIstftRuntime(HipIstftRuntime &&) noexcept = default;

HipIstftRuntime & HipIstftRuntime::operator=(HipIstftRuntime &&) noexcept = default;

HipIstftRuntimeResult HipIstftRuntime::compute(
    const std::vector<float> & log_magnitude_phase,
    const std::vector<float> & window) {
    return impl_->compute(log_magnitude_phase, window);
}

}  // namespace engine::audio::detail
