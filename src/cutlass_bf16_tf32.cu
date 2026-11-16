// CUTLASS 2.x device GEMM bench: bf16 and tf32 TN, fp32 accumulate.
// A: row-major MxK (lda=K), B: col-major KxN (ldb=N), C: row-major MxN (ldc=N).
// Sweep: TB {128x128x32, 128x256x32, 256x128x32} x stages {3,4} x swizzle {1,4}.
// TF32 gets bonus stages=2 runs for the 256-wide tiles (st>=3 exceed 99KB SMEM opt-in).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/bfloat16.h>
#include <cutlass/tfloat32.h>

#ifndef RUN_BF16
#define RUN_BF16 1
#endif
#ifndef RUN_TF32
#define RUN_TF32 1
#endif

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

template<typename ElemT, int INST_K, int TBM,int TBN,int TBK,int WM,int WN,int STAGES,int SWZ>
double bench_one(int M,int N,int K, void*Av, void*Bv, void*Cv){
    ElemT *A=(ElemT*)Av, *B=(ElemT*)Bv, *C=(ElemT*)Cv;
    constexpr int EPI = 16/(int)sizeof(ElemT); // 128-bit epilogue access
    using Gemm = cutlass::gemm::device::Gemm<
        ElemT, cutlass::layout::RowMajor,
        ElemT, cutlass::layout::ColumnMajor,
        ElemT, cutlass::layout::RowMajor,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM,TBN,TBK>,
        cutlass::gemm::GemmShape<WM,WN,TBK>,
        cutlass::gemm::GemmShape<16,8,INST_K>,
        cutlass::epilogue::thread::LinearCombination<ElemT, EPI, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SWZ>,
        STAGES>;
    typename Gemm::Arguments args({M,N,K}, {A,K},{B,N},{C,N},{C,N},{1.0f,0.0f});
    Gemm gemm;
    cutlass::Status st = gemm.initialize(args);
    if (st != cutlass::Status::kSuccess) return -1.0;
    st = gemm(); if (st != cutlass::Status::kSuccess) return -2.0;
    CKC(cudaDeviceSynchronize());
    for (int i=0;i<2;i++) gemm();
    CKC(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    int iters = (M>=16384)?4:8;
    for (int i=0;i<iters;i++) gemm();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 2.0*M*N*K*iters/(ms/1e3)/1e12;
}

int main(int argc, char**argv){
    int S = argc>1?atoi(argv[1]):8192;
    int M=S,N=S,K=S;
    void *A,*B,*C;
    // allocate 4 bytes/elem (tf32 worst case); bf16 uses half
    CKC(cudaMalloc(&A,(size_t)M*K*4)); CKC(cudaMalloc(&B,(size_t)K*N*4)); CKC(cudaMalloc(&C,(size_t)M*N*4));
    CKC(cudaMemset(A,0x11,(size_t)M*K*4)); CKC(cudaMemset(B,0x11,(size_t)K*N*4));
#define RUNP(tag,TBM,TBN,TBK,ST,SWZ,t) \
    printf("%s TB %3dx%3dx%2d warp 64x64 st%d swz%d : ",tag,TBM,TBN,TBK,ST,SWZ); \
    if(t<0) printf("FAILED (%.0f)\n",-t); else printf("%8.1f TF/s\n",t);
#if RUN_BF16
    printf("=== bf16 TN %d^3 (fp32 acc, bf16 out) ===\n", S);
#define RUN_BF(TBM,TBN,TBK,ST,SWZ) { double t=bench_one<cutlass::bfloat16_t,16,TBM,TBN,TBK,64,64,ST,SWZ>(M,N,K,A,B,C); RUNP("bf16",TBM,TBN,TBK,ST,SWZ,t) }
    RUN_BF(128,128,32,3,1)
    RUN_BF(128,128,32,3,4)
    RUN_BF(128,128,32,4,1)
    RUN_BF(128,128,32,4,4)
    RUN_BF(128,256,32,3,1)
    RUN_BF(128,256,32,3,4)
    RUN_BF(128,256,32,4,1)
    RUN_BF(128,256,32,4,4)
    RUN_BF(256,128,32,3,1)
    RUN_BF(256,128,32,3,4)
    RUN_BF(256,128,32,4,1)
    RUN_BF(256,128,32,4,4)
#endif
#if RUN_TF32
    printf("=== tf32 TN %d^3 (fp32 acc, tf32 out) ===\n", S);
#define RUN_TF(TBM,TBN,TBK,ST,SWZ) { double t=bench_one<cutlass::tfloat32_t,8,TBM,TBN,TBK,64,64,ST,SWZ>(M,N,K,A,B,C); RUNP("tf32",TBM,TBN,TBK,ST,SWZ,t) }
    RUN_TF(128,128,32,3,1)
    RUN_TF(128,128,32,3,4)
    RUN_TF(128,128,32,4,1)
    RUN_TF(128,128,32,4,4)
    RUN_TF(128,256,32,3,1)
    RUN_TF(128,256,32,3,4)
    RUN_TF(128,256,32,4,1)
    RUN_TF(128,256,32,4,4)
    RUN_TF(256,128,32,3,1)
    RUN_TF(256,128,32,3,4)
    RUN_TF(256,128,32,4,1)
    RUN_TF(256,128,32,4,4)
    // bonus: stages=2 so 256-wide tf32 tiles fit in 99KB SMEM
    RUN_TF(128,256,32,2,1)
    RUN_TF(128,256,32,2,4)
    RUN_TF(256,128,32,2,1)
    RUN_TF(256,128,32,2,4)
#endif
    return 0;
}
