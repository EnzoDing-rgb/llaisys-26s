#include "runtime_api.hpp"

namespace llaisys::device {

int getDeviceCount() {
    return 0;
}

void setDevice(int) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void deviceSynchronize() {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

llaisysStream_t createStream() {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void destroyStream(llaisysStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
void streamSynchronize(llaisysStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void *mallocDevice(size_t size) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void freeDevice(void *ptr) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void *mallocHost(size_t size) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void freeHost(void *ptr) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

static const LlaisysRuntimeAPI NOOP_RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI() {
    return &NOOP_RUNTIME_API;
}

const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type) {
    // Implement for all device types
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::device::cpu::getRuntimeAPI();
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return llaisys::device::nvidia::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    case LLAISYS_DEVICE_ILUVATAR:
#ifdef ENABLE_ILUVATAR_API
        return llaisys::device::iluvatar::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}

#ifdef ENABLE_ILUVATAR_API
namespace iluvatar {
// 天数（Iluvatar）复用同一套 CUDA Runtime 实现。
// COREX 提供与 CUDA Runtime 兼容的接口（cudaMalloc/cudaMemcpy/...），
// 因此直接把 nvidia 的函数表原样返回，无需复制一份 runtime 代码。
// 注意：本别名依赖 ENABLE_NVIDIA_API 同时打开（xmake.lua 在 --nv-gpu=y 时
// 恒定义 ENABLE_NVIDIA_API，天数构建再额外定义 ENABLE_ILUVATAR_API）。
const LlaisysRuntimeAPI *getRuntimeAPI() {
    return llaisys::device::nvidia::getRuntimeAPI();
}
} // namespace iluvatar
#endif
} // namespace llaisys::device
