// v7: swizzle deep-dive for the two champions across all shapes.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/half.h>
#include <cutlass/float8.h>

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
using e4m3 = cutlass::float_e4m3_t;

template<int TBM,int TBN,int TBK,int WM,int WN,int STAGES,int SWZ=1, typename OP=cutlass::arch::OpMultiplyAdd>
double bench_one(int M,int N,int K, e4m3*A, e4m3*B, cutlass::half_t*C){
    using Gemm = cutlass::gemm::device::Gemm<
        e4m3, cutlass::layout::RowMajor,
        e4m3, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<TBM,TBN,TBK>,
        cutlass::gemm::GemmShape<WM,WN,TBK>,
        cutlass::gemm::GemmShape<16,8,32>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SWZ>,
        STAGES, 16, 16, false, OP>;
    typename Gemm::Arguments args({M,N,K}, {A,K},{B,K},{C,N},{C,N},{1.0f,0.0f});
    Gemm gemm;
    cutlass::Status st = gemm.initialize(args);
    if (st != cutlass::Status::kSuccess) return -1.0;
    st = gemm(); if (st != cutlass::Status::kSuccess) return -2.0;
    CKC(cudaDeviceSynchronize());
    cudaEvent_t r0,r1; cudaEventCreate(&r0); cudaEventCreate(&r1);
    cudaEventRecord(r0);
    do { for(int i=0;i<32;i++) gemm(); cudaEventRecord(r1); cudaEventSynchronize(r1);
         float ms; cudaEventElapsedTime(&ms,r0,r1); if (ms>300.f) break; } while(true);
    int iters = (M>=16384)?10:((M>=8192)?20:50);
    double best=0;
    for(int rep=0;rep<3;rep++){
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int i=0;i<iters;i++) gemm();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms,e0,e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        double tf = 2.0*M*N*K*iters/(ms/1e3)/1e12;
        if (tf>best) best=tf;
    }
    cudaEventDestroy(r0); cudaEventDestroy(r1);
    return best;
}

int main(int argc, char**argv){
    using OpMA = cutlass::arch::OpMultiplyAdd;
    using OpFA = cutlass::arch::OpMultiplyAddFastAccum;
    for (int Si=1; Si<argc; Si++){
        int S=atoi(argv[Si]); int M=S,N=S,K=S;
        e4m3 *A,*B; cutlass::half_t *C;
        CKC(cudaMalloc(&A,(size_t)M*K)); CKC(cudaMalloc(&B,(size_t)K*N)); CKC(cudaMalloc(&C,(size_t)M*N*2));
        CKC(cudaMemset(A,0x11,(size_t)M*K)); CKC(cudaMemset(B,0x11,(size_t)K*N));
        printf("== fp8 e4m3 TN %d^3 ==\n", S);
#define RUN(TBM,TBN,TBK,WM,WN,ST,SW,OP,ONM) { double t=bench_one<TBM,TBN,TBK,WM,WN,ST,SW,OP>(M,N,K,A,B,C); \
        printf("%-8s TB %3dx%3dx%3d warp %3dx%3d st%d swz%d : ",ONM,TBM,TBN,TBK,WM,WN,ST,SW); \
        if(t<0) printf("FAILED (%.0f)\n",-t); else printf("%8.1f TF/s\n",t); }
        RUN(128,128,64,32,32,3,3,OpMA,"staged")
        RUN(128,128,64,32,32,3,4,OpMA,"staged")
        RUN(128,128,64,32,32,3,5,OpMA,"staged")
        RUN(128,128,64,32,32,3,6,OpMA,"staged")
        RUN(128,128,64,64,32,4,4,OpMA,"staged")
        RUN(128,128,64,64,32,4,5,OpMA,"staged")
        RUN(128,128,64,32,32,4,4,OpMA,"staged")
        RUN(128,256,64,64,64,3,3,OpFA,"fastacc")
        RUN(128,256,64,64,64,3,4,OpFA,"fastacc")
        RUN(128,256,64,64,64,3,5,OpFA,"fastacc")
        RUN(128,256,64,64,64,3,6,OpFA,"fastacc")
        RUN(256,128,64,64,64,4,4,OpFA,"fastacc")
        cudaFree(A);cudaFree(B);cudaFree(C);
    }
    return 0;
}
