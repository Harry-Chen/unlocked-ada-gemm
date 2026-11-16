// Minimal P2P bandwidth/latency test for multi-GPU (cuda-samples style)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

int main(int argc, char **argv) {
    int n = 0; CK(cudaGetDeviceCount(&n));
    printf("GPUs: %d\n", n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, i));
        printf("Device %d: %s, pci %04x:%02x:%02x\n", i, p.name, p.pciDomainID, p.pciBusID, p.pciDeviceID);
    }

    // connectivity matrix
    printf("\nP2P Connectivity Matrix (1=can access)\n     D\\D");
    for (int j = 0; j < n; j++) printf("%6d", j);
    printf("\n");
    int can[16][16] = {};
    for (int i = 0; i < n; i++) {
        printf("%6d", i);
        for (int j = 0; j < n; j++) {
            if (i == j) { can[i][j] = 1; printf("%6d", 1); continue; }
            int c = 0; CK(cudaDeviceCanAccessPeer(&c, i, j));
            can[i][j] = c; printf("%6d", c);
        }
        printf("\n");
    }

    const size_t SZ = 256ull << 20; // 256 MB
    void *buf[16] = {};
    cudaEvent_t ev0[16], ev1[16];
    for (int i = 0; i < n; i++) {
        CK(cudaSetDevice(i));
        CK(cudaMalloc(&buf[i], SZ));
        CK(cudaMemset(buf[i], 0, SZ));
        CK(cudaEventCreate(&ev0[i])); CK(cudaEventCreate(&ev1[i]));
    }

    for (int pass = 0; pass < 2; pass++) { // pass0: P2P disabled (staged), pass1: enabled
        // (re)set peer access state
        for (int i = 0; i < n; i++) {
            CK(cudaSetDevice(i));
            for (int j = 0; j < n; j++) {
                if (i == j || !can[i][j]) continue;
                if (pass == 1) cudaDeviceEnablePeerAccess(j, 0);
                else cudaDeviceDisablePeerAccess(j);
            }
        }
        printf("\nUnidirectional P2P=%s Bandwidth Matrix (GB/s)\n     D\\D", pass ? "Enabled " : "Disabled");
        for (int j = 0; j < n; j++) printf("%9d", j);
        printf("\n");
        for (int i = 0; i < n; i++) {
            printf("%6d", i);
            for (int j = 0; j < n; j++) {
                if (i == j) { printf("%9s", "-"); continue; }
                CK(cudaSetDevice(i));
                // warmup
                for (int k = 0; k < 3; k++) CK(cudaMemcpyPeerAsync(buf[i], i, buf[j], j, SZ));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(ev0[i]));
                int iters = 10;
                for (int k = 0; k < iters; k++) CK(cudaMemcpyPeerAsync(buf[i], i, buf[j], j, SZ));
                CK(cudaEventRecord(ev1[i]));
                CK(cudaEventSynchronize(ev1[i]));
                float ms = 0; CK(cudaEventElapsedTime(&ms, ev0[i], ev1[i]));
                double gbs = (double)SZ * iters / (ms / 1000.0) / 1e9;
                printf("%9.2f", gbs);
            }
            printf("\n");
        }
    }

    // latency: 8-byte copies, peer enabled
    for (int i = 0; i < n; i++) { CK(cudaSetDevice(i));
        for (int j = 0; j < n; j++) if (i != j && can[i][j]) cudaDeviceEnablePeerAccess(j, 0); }
    printf("\nP2P Latency Matrix (us, 8B copy)\n     D\\D");
    for (int j = 0; j < n; j++) printf("%9d", j);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%6d", i);
        for (int j = 0; j < n; j++) {
            if (i == j) { printf("%9s", "-"); continue; }
            CK(cudaSetDevice(i));
            for (int k = 0; k < 10; k++) CK(cudaMemcpyPeerAsync(buf[i], i, buf[j], j, 8));
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ev0[i]));
            int iters = 1000;
            for (int k = 0; k < iters; k++) CK(cudaMemcpyPeerAsync(buf[i], i, buf[j], j, 8));
            CK(cudaEventRecord(ev1[i])); CK(cudaEventSynchronize(ev1[i]));
            float ms = 0; CK(cudaEventElapsedTime(&ms, ev0[i], ev1[i]));
            printf("%9.2f", ms * 1000.0 / iters);
        }
        printf("\n");
    }
    for (int i = 0; i < n; i++) { CK(cudaSetDevice(i)); cudaFree(buf[i]); }
    printf("\nDONE\n");
    return 0;
}
