// CUTLASS 2.x device GEMM bench: fp16 TN, fp32 accumulate, fp16 out.
// Tries several threadblock/warp tile shapes and stage counts.
#include <cstdio>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/half.h>

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

template<int TBM,int TBN,int TBK,int WM,int WN,int STAGES,int SWZ=1>
double bench_one(int M,int N,int K, cutlass::half_t*A, cutlass::half_t*B, cutlass::half_t*C){
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM,TBN,TBK>,
        cutlass::gemm::GemmShape<WM,WN,TBK>,
        cutlass::gemm::GemmShape<16,8,16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
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
    cutlass::half_t *A,*B,*C;
    CKC(cudaMalloc(&A,(size_t)M*K*2)); CKC(cudaMalloc(&B,(size_t)K*N*2)); CKC(cudaMalloc(&C,(size_t)M*N*2));
    CKC(cudaMemset(A,0x11,(size_t)M*K*2)); CKC(cudaMemset(B,0x11,(size_t)K*N*2));
    printf("CUTLASS fp16 TN %d^3 (fp32 acc)\n", S);
#define RUN(TBM,TBN,TBK,WM,WN,ST) { double t=bench_one<TBM,TBN,TBK,WM,WN,ST>(M,N,K,A,B,C); \
    printf("TB %3dx%3dx%2d warp %3dx%3d st%d : ",TBM,TBN,TBK,WM,WN,ST); \
    if(t<0) printf("FAILED (%.0f)\n",-t); else printf("%8.1f TF/s\n",t); }
    RUN(128,128,32,64,64,3)
    RUN(128,256,32,64,64,3)
    RUN(256,128,32,64,64,3)
    RUN(128,128,64,64,64,3)
    RUN(128,256,64,64,64,3)
    RUN(128,256,64,64,64,4)
    RUN(256,256,32,128,64,3)
    RUN(256,128,64,64,64,4)
    RUN(64,256,64,64,64,4)
    RUN(128,128,32,64,64,5)
    { double t=bench_one<128,256,32,64,64,3,2>(M,N,K,A,B,C); printf("TB 128x256x32 warp 64x64 st3 swz2 : %8.1f TF/s\n", t); }
    { double t=bench_one<128,256,32,64,64,3,3>(M,N,K,A,B,C); printf("TB 128x256x32 warp 64x64 st3 swz3 : %8.1f TF/s\n", t); }
    { double t=bench_one<128,256,32,64,64,3,4>(M,N,K,A,B,C); printf("TB 128x256x32 warp 64x64 st3 swz4 : %8.1f TF/s\n", t); }
    { double t=bench_one<256,128,32,64,64,3,3>(M,N,K,A,B,C); printf("TB 256x128x32 warp 64x64 st3 swz3 : %8.1f TF/s\n", t); }
    return 0;
}
