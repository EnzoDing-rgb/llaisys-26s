#include "../runtime_api.hpp"
#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    // 返回值是 cudaError_t；设备个数写在输出参数 count 里
    // 签名：cudaError_t cudaGetDeviceCount(int *count)
    ASSERT(cudaGetDeviceCount(&count) == cudaSuccess, "cudaGetDeviceCount failed");
    return count;
}

void setDevice(int device_id) {
    // device_id：当前线程绑定的 GPU 序号（0, 1, 2, ...）
    ASSERT(cudaSetDevice(device_id) == cudaSuccess, "cudaSetDevice failed");
}

void deviceSynchronize() {
    // 等当前 GPU 上所有 stream 的已提交工作完成
    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize failed");
}

llaisysStream_t createStream() {
    // CUDA 用 cudaStream_t；llaisys 对外是 void* 句柄 llaisysStream_t
    cudaStream_t stream = nullptr;
    // 输出参数写法：新建的 stream 写进 &stream
    // 这里显式创建非默认流，供异步拷贝 / 算子重叠使用
    ASSERT(cudaStreamCreate(&stream) == cudaSuccess, "cudaStreamCreate failed");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    // 与 createStream 成对，释放驱动侧 stream 资源
    ASSERT(cudaStreamDestroy(cuda_stream) == cudaSuccess, "cudaStreamDestroy failed");
}

void streamSynchronize(llaisysStream_t stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    // 阻塞当前 host 线程，直到这一条 stream 上的已提交工作完成
    ASSERT(cudaStreamSynchronize(cuda_stream) == cudaSuccess, "cudaStreamSynchronize failed");
}

// llaisys 拷贝方向 → CUDA cudaMemcpyKind（显式映射，便于对照文档）
static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default:
        ASSERT(false, "unknown llaisysMemcpyKind_t");
        return cudaMemcpyDefault;
    }
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    // 在当前 GPU（cudaSetDevice 绑定的那块）上分配显存
    ASSERT(cudaMalloc(&ptr, size) == cudaSuccess, "cudaMalloc failed");
    return ptr;
}

void freeDevice(void *ptr) {
    // 与 cudaMalloc 成对
    ASSERT(cudaFree(ptr) == cudaSuccess, "cudaFree failed");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    // pinned host 内存：页锁定，GPU DMA 可直接访问；适合异步 H2D/D2H
    ASSERT(cudaMallocHost(&ptr, size) == cudaSuccess, "cudaMallocHost failed");
    return ptr;
}

void freeHost(void *ptr) {
    // 与 cudaMallocHost 成对
    ASSERT(cudaFreeHost(ptr) == cudaSuccess, "cudaFreeHost failed");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    // 函数返回时拷贝已完成
    ASSERT(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)) == cudaSuccess, "cudaMemcpy failed");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    // 把拷贝提交到指定 stream 后立即返回；完成时机由该 stream 进度决定
    ASSERT(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), cuda_stream) == cudaSuccess,
           "cudaMemcpyAsync failed");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
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

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
