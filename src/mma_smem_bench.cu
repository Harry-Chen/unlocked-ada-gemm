// SMEM-fed MMA sustained-rate microbench (fp16 m16n8k16, fp32 acc).
// Mode R: per iteration load one A frag (ldmatrix.x4) + one B frag (ldmatrix.x2),
// then issue R independent-accumulator MMAs reusing those fragments.
// SMEM traffic per MMA = 768B / R. R=0 -> pure register reuse (no loads in loop).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

#define WARPS_PER_BLOCK 8
#define BLOCKS 1056
#define MAXCHAIN 8
#define A_SLOTS 4
#define B_SLOTS 4
// per-warp: A slots 4*512B + B slots 4*256B = 3KB
#define WARP_BYTES (A_SLOTS*512 + B_SLOTS*256)

__device__ __forceinline__ void ldsm_x4(unsigned &r0,unsigned &r1,unsigned &r2,unsigned &r3,unsigned addr){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(unsigned &r0,unsigned &r1,unsigned addr){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0),"=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void mma_f16(float &d0,float &d1,float &d2,float &d3,
                                        unsigned a0,unsigned a1,unsigned a2,unsigned a3,
                                        unsigned b0,unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

template<int R>
__global__ void __launch_bounds__(WARPS_PER_BLOCK*32) bench(unsigned long long iters, float* sink){
    __shared__ unsigned char smem_raw[WARPS_PER_BLOCK * WARP_BYTES];
    int warp = threadIdx.x/32, lane = threadIdx.x%32;
    unsigned char* base = smem_raw + warp*WARP_BYTES;
    for (int i=threadIdx.x; i<WARPS_PER_BLOCK*WARP_BYTES/4; i+=WARPS_PER_BLOCK*32)
        ((unsigned*)smem_raw)[i] = 0x3c003c00u;
    __syncthreads();

    // ldmatrix lane->address mapping (row of an 8x8 fp16 matrix = 16B)
    unsigned a_off = (lane/8)*128 + (lane%8)*16;   // within a 512B A tile (x4)
    unsigned b_off = ((lane%16)/8)*128 + (lane%8)*16; // within a 256B B tile (x2)
    unsigned a_base = (unsigned)__cvta_generic_to_shared(base);
    unsigned b_base = (unsigned)__cvta_generic_to_shared(base + A_SLOTS*512);

    unsigned a0,a1,a2,a3,b0,b1;
    float acc[MAXCHAIN][4];
    for (int j=0;j<MAXCHAIN;j++) acc[j][0]=acc[j][1]=acc[j][2]=acc[j][3]=0.0f;
    ldsm_x4(a0,a1,a2,a3, a_base + a_off);
    ldsm_x2(b0,b1, b_base + b_off);

    for (unsigned long long i=0;i<iters;i++){
        if (R>0){
            unsigned aslot = i & (A_SLOTS-1), bslot = i & (B_SLOTS-1);
            ldsm_x4(a0,a1,a2,a3, a_base + aslot*512 + a_off);
            ldsm_x2(b0,b1, b_base + bslot*256 + b_off);
        }
        if (R==0){
#pragma unroll
            for (int j=0;j<MAXCHAIN;j++) mma_f16(acc[j][0],acc[j][1],acc[j][2],acc[j][3],a0,a1,a2,a3,b0,b1);
        } else {
#pragma unroll
            for (int j=0;j<R;j++) mma_f16(acc[j][0],acc[j][1],acc[j][2],acc[j][3],a0,a1,a2,a3,b0,b1);
        }
    }
    float carry=0;
    for (int j=0;j<MAXCHAIN;j++) carry += acc[j][0]+acc[j][1]+acc[j][2]+acc[j][3];
    if (carry == 1234567.0f) sink[threadIdx.x]=carry;
}

template<int R>
double run_mode(unsigned long long iters, float* sink){
    dim3 g(BLOCKS), b(WARPS_PER_BLOCK*32);
    bench<R><<<g,b>>>(iters,sink); CKC(cudaGetLastError()); CKC(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    bench<R><<<g,b>>>(iters,sink);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    int nmma = (R==0)?MAXCHAIN:R;
    double flops = (double)BLOCKS*WARPS_PER_BLOCK*iters*nmma*8192.0*2/2; // m16n8k16 = 8192 FLOP (2*16*8*16)
    flops = (double)BLOCKS*WARPS_PER_BLOCK*iters*nmma*(2.0*16*8*16);
    return flops/(ms/1e3)/1e12;
}

int main(int argc, char**argv){
    unsigned long long iters = argc>1?strtoull(argv[1],0,10):20000;
    float* sink; CKC(cudaMalloc(&sink, 4096));
    printf("smem-fed mma bench: blocks=%d warps/blk=%d iters=%llu\n", BLOCKS, WARPS_PER_BLOCK, iters);
    double t0 = run_mode<0>(iters,sink); printf("R=0 (regs only,   0B/mma): %8.1f TF/s\n", t0);
    double t1 = run_mode<1>(iters,sink); printf("R=1 (no reuse,  768B/mma): %8.1f TF/s\n", t1);
    double t2 = run_mode<2>(iters,sink); printf("R=2 (          384B/mma): %8.1f TF/s\n", t2);
    double t4 = run_mode<4>(iters,sink); printf("R=4 (          192B/mma): %8.1f TF/s\n", t4);
    double t8 = run_mode<8>(iters,sink); printf("R=8 (           96B/mma): %8.1f TF/s\n", t8);
    return 0;
}
