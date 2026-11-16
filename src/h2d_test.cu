// Per-GPU H2D / D2H bandwidth test
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)
int main() {
    int n = 0; CK(cudaGetDeviceCount(&n));
    const size_t SZ = 512ull << 20;
    void *host; CK(cudaMallocHost(&host, SZ));
    printf("GPU  H2D_GBps  D2H_GBps\n");
    for (int i = 0; i < n; i++) {
        CK(cudaSetDevice(i));
        void *dev; CK(cudaMalloc(&dev, SZ));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (int k = 0; k < 3; k++) CK(cudaMemcpyAsync(dev, host, SZ, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int k = 0; k < 5; k++) CK(cudaMemcpyAsync(dev, host, SZ, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double h2d = (double)SZ * 5 / (ms / 1e3) / 1e9;
        for (int k = 0; k < 3; k++) CK(cudaMemcpyAsync(host, dev, SZ, cudaMemcpyDeviceToHost));
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int k = 0; k < 5; k++) CK(cudaMemcpyAsync(host, dev, SZ, cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        double d2h = (double)SZ * 5 / (ms / 1e3) / 1e9;
        printf("%3d  %8.2f  %8.2f\n", i, h2d, d2h);
        cudaFree(dev);
    }
    return 0;
}
