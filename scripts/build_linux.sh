#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV=""
BUILD_DIR=""
BUILD_TYPE="RelWithDebInfo"
CUDA_MODE="auto"
HIP_MODE="off"
VULKAN_MODE="off"
WITH_TESTS="OFF"
WITH_EXAMPLES="OFF"
WITH_WARMBENCH="OFF"
AUDIOCPP_DEPLOYMENT_BUILD="OFF"
NATIVE_CPU="ON"
LLAMAFILE="ON"
TARGETS=()
JOBS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --backend)
            if [[ "$2" == "cuda" ]]; then
                CUDA_MODE="on"
                HIP_MODE="off"
                VULKAN_MODE="off"
            elif [[ "$2" == "hip" || "$2" == "rocm" ]]; then
                CUDA_MODE="off"
                HIP_MODE="on"
                VULKAN_MODE="off"
            elif [[ "$2" == "cpu" ]]; then
                CUDA_MODE="off"
                HIP_MODE="off"
                VULKAN_MODE="off"
            elif [[ "$2" == "vulkan" ]]; then
                CUDA_MODE="off"
                HIP_MODE="off"
                VULKAN_MODE="on"
            else
                echo "Unsupported legacy --backend value: $2" >&2
                exit 1
            fi
            shift 2
            ;;
        --cuda)
            CUDA_MODE="$2"
            shift 2
            ;;
        --hip|--rocm)
            HIP_MODE="$2"
            shift 2
            ;;
        --vulkan)
            VULKAN_MODE="$2"
            shift 2
            ;;
        --with-tests)
            WITH_TESTS="ON"
            shift
            ;;
        --with-examples)
            WITH_EXAMPLES="ON"
            shift
            ;;
        --with-warmbench)
            WITH_WARMBENCH="ON"
            shift
            ;;
        --deployment-build)
            AUDIOCPP_DEPLOYMENT_BUILD="ON"
            shift
            ;;
        --native-cpu)
            case "$2" in
                ON|on|On|1|true|TRUE|yes|YES)
                    NATIVE_CPU="ON"
                    ;;
                OFF|off|Off|0|false|FALSE|no|NO)
                    NATIVE_CPU="OFF"
                    ;;
                *)
                    echo "--native-cpu must be ON or OFF" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --llamafile)
            case "$2" in
                ON|on|On|1|true|TRUE|yes|YES)
                    LLAMAFILE="ON"
                    ;;
                OFF|off|Off|0|false|FALSE|no|NO)
                    LLAMAFILE="OFF"
                    ;;
                *)
                    echo "--llamafile must be ON or OFF" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --target)
            TARGETS+=("$2")
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

ENGINE_ENABLE_CUDA="OFF"
ENGINE_ENABLE_HIP="OFF"
ENGINE_ENABLE_VULKAN="OFF"
case "$CUDA_MODE" in
    on)
        ENGINE_ENABLE_CUDA="ON"
        ;;
    off)
        ENGINE_ENABLE_CUDA="OFF"
        ;;
    auto)
        if command -v nvcc >/dev/null 2>&1; then
            ENGINE_ENABLE_CUDA="ON"
        fi
        ;;
    *)
        echo "Unknown --cuda mode: $CUDA_MODE (expected on, off, or auto)" >&2
        exit 1
        ;;
esac

case "$HIP_MODE" in
    on)
        ENGINE_ENABLE_HIP="ON"
        if ! command -v hipcc >/dev/null 2>&1 && [[ -z "${ROCM_PATH:-}" ]]; then
            echo "Error: ROCm / HIP compiler (hipcc) not found in PATH and ROCM_PATH is not set." >&2
            echo "When using Nix, run with the ROCm devShell:" >&2
            echo "  nix develop .#rocm -c bash ./scripts/build_linux.sh --backend hip" >&2
            exit 1
        fi
        ;;
    off)
        ENGINE_ENABLE_HIP="OFF"
        ;;
    auto)
        if command -v hipcc >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1; then
            ENGINE_ENABLE_HIP="ON"
        fi
        ;;
    *)
        echo "Unknown --hip mode: $HIP_MODE (expected on, off, or auto)" >&2
        exit 1
        ;;
esac

case "$VULKAN_MODE" in
    on)
        ENGINE_ENABLE_VULKAN="ON"
        ;;
    off)
        ENGINE_ENABLE_VULKAN="OFF"
        ;;
    *)
        echo "Unknown --vulkan mode: $VULKAN_MODE (expected on or off)" >&2
        exit 1
        ;;
esac

GPU_BACKENDS_ENABLED=0
[[ "$ENGINE_ENABLE_CUDA" == "ON" ]] && GPU_BACKENDS_ENABLED=$((GPU_BACKENDS_ENABLED + 1))
[[ "$ENGINE_ENABLE_HIP" == "ON" ]] && GPU_BACKENDS_ENABLED=$((GPU_BACKENDS_ENABLED + 1))
[[ "$ENGINE_ENABLE_VULKAN" == "ON" ]] && GPU_BACKENDS_ENABLED=$((GPU_BACKENDS_ENABLED + 1))

if [[ $GPU_BACKENDS_ENABLED -gt 1 ]]; then
    echo "CUDA, HIP, and Vulkan backends cannot be simultaneously enabled by this script" >&2
    exit 1
fi

BACKEND_NAME="cpu"
if [[ "$ENGINE_ENABLE_CUDA" == "ON" ]]; then
    BACKEND_NAME="cuda"
elif [[ "$ENGINE_ENABLE_HIP" == "ON" ]]; then
    BACKEND_NAME="hip"
elif [[ "$ENGINE_ENABLE_VULKAN" == "ON" ]]; then
    BACKEND_NAME="vulkan"
fi

BUILD_TYPE_NAME="$(printf '%s' "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
if [[ "$BUILD_TYPE_NAME" == "relwithdebinfo" ]]; then
    BUILD_TYPE_NAME="release"
fi

if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="build/linux-${BACKEND_NAME}-${BUILD_TYPE_NAME}"
fi

if [[ -z "$JOBS" ]]; then
    JOBS="8"
fi

RUNNER=()
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required on PATH when --conda-env is used" >&2
        exit 1
    fi
    if ! conda info --envs | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
        echo "Conda environment '$CONDA_ENV' was not found" >&2
        exit 1
    fi
    RUNNER=(conda run -n "$CONDA_ENV")
fi

if [[ -n "$CONDA_ENV" ]]; then
    echo "Using conda env: $CONDA_ENV"
fi
echo "Using generator: $GENERATOR"
echo "Using build dir: $BUILD_DIR"
echo "Including CUDA backend: $ENGINE_ENABLE_CUDA"
echo "Including HIP backend: $ENGINE_ENABLE_HIP"
echo "Including Vulkan backend: $ENGINE_ENABLE_VULKAN"
echo "Native CPU optimization: $NATIVE_CPU"
echo "llamafile SGEMM: $LLAMAFILE"
echo "Building examples: $WITH_EXAMPLES"
echo "Building tests: $WITH_TESTS"
echo "Building warmbench: $WITH_WARMBENCH"
echo "Deployment build: $AUDIOCPP_DEPLOYMENT_BUILD"

HIP_EXTRA_ARGS=()
if [[ "$ENGINE_ENABLE_HIP" == "ON" ]]; then
    HIP_EXTRA_ARGS=(
        "-DCMAKE_HIP_ARCHITECTURES=gfx1151"
        "-DGGML_CUDA_FORCE_MMQ=ON"
        "-DGGML_CUDA_FA_ALL_QUANTS=ON"
        "-DGGML_HIP_UMA=ON"
        "-DGGML_ROCM_USE_HIPBLASLT=ON"
    )
fi

"${RUNNER[@]}" cmake \
    -S . \
    -B "$BUILD_DIR" \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENGINE_ENABLE_CUDA="$ENGINE_ENABLE_CUDA" \
    -DENGINE_ENABLE_HIP="$ENGINE_ENABLE_HIP" \
    -DENGINE_ENABLE_VULKAN="$ENGINE_ENABLE_VULKAN" \
    -DENGINE_ENABLE_NATIVE_CPU="$NATIVE_CPU" \
    -DENGINE_ENABLE_LLAMAFILE="$LLAMAFILE" \
    -DENGINE_BUILD_EXAMPLES="$WITH_EXAMPLES" \
    -DENGINE_BUILD_TESTS="$WITH_TESTS" \
    -DENGINE_BUILD_WARMBENCH="$WITH_WARMBENCH" \
    -DAUDIOCPP_DEPLOYMENT_BUILD="$AUDIOCPP_DEPLOYMENT_BUILD" \
    "${HIP_EXTRA_ARGS[@]}"

BUILD_CMD=("${RUNNER[@]}" cmake --build "$BUILD_DIR" --parallel "$JOBS")
for target in "${TARGETS[@]}"; do
    BUILD_CMD+=(--target "$target")
done

"${BUILD_CMD[@]}"
