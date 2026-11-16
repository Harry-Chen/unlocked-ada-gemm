// autotune_demo: default heuristic pick vs LtAutotuner, per shape. fp16 + fp8.
#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "cublaslt_autotune.hpp"

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
#define CKL(x) do { cublasStatus_t s=(x); if(s){printf("cuBLAS err %d @%d\n",(int)s,__LINE__); exit(1);} } while(0)

static size_t esz(cudaDataType_t t){ switch(t){case CUDA_R_16F:case CUDA_R_16BF:return 2;case CUDA_R_8F_E4M3:return 1;default:return 4;} }

struct Prob { cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t Ad,Bd,Cd,Dd; void *A,*B,*C,*D; int M,N,K; };

Prob setup(cublasComputeType_t comp, cudaDataType_t io, cudaDataType_t outt, int M,int N,int K){
    Prob p{}; p.M=M;p.N=N;p.K=K;
    CKC(cudaMalloc(&p.A,(size_t)K*M*esz(io))); CKC(cudaMalloc(&p.B,(size_t)K*N*esz(io)));
    CKC(cudaMalloc(&p.C,(size_t)M*N*esz(outt))); CKC(cudaMalloc(&p.D,(size_t)M*N*esz(outt)));
    CKC(cudaMemset(p.A,0x11,(size_t)K*M*esz(io))); CKC(cudaMemset(p.B,0x11,(size_t)K*N*esz(io)));
    CKL(cublasLtMatmulDescCreate(&p.desc, comp, CUDA_R_32F));
    cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT));
    cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN));
    CKL(cublasLtMatrixLayoutCreate(&p.Ad,io,K,M,K));
    CKL(cublasLtMatrixLayoutCreate(&p.Bd,io,K,N,K));
    CKL(cublasLtMatrixLayoutCreate(&p.Cd,outt,M,N,M));
    CKL(cublasLtMatrixLayoutCreate(&p.Dd,outt,M,N,M));
    return p;
}

double time_call(cublasLtHandle_t lt, Prob&p, float alpha, float beta, void*ws, size_t wsSize,
                 cudaStream_t st, ltat::Autotuner* tuner, int iters){
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto call = [&](){
        if (tuner) CKL(tuner->run(lt,p.desc,&alpha,p.A,p.Ad,p.B,p.Bd,&beta,p.C,p.Cd,p.D,p.Dd,ws,wsSize,st));
        else CKL(cublasLtMatmul(lt,p.desc,&alpha,p.A,p.Ad,p.B,p.Bd,&beta,p.C,p.Cd,p.D,p.Dd,nullptr,ws,wsSize,st));
    };
    call(); CKC(cudaStreamSynchronize(st)); // warm
    cudaEventRecord(e0,st);
    for(int i=0;i<iters;i++) call();
    cudaEventRecord(e1,st); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 2.0*p.M*p.N*p.K*iters/(ms/1e3)/1e12;
}

void run_case(ltat::Autotuner& tuner, cublasLtHandle_t lt, const char* name,
              cublasComputeType_t comp, cudaDataType_t io, cudaDataType_t outt,
              int M,int N,int K, void*ws, size_t wsSize, cudaStream_t st){
    Prob p = setup(comp, io, outt, M,N,K);
    float alpha=1.0f, beta=0.0f;
    double flops=2.0*M*N*K;
    int iters = (int)fmax(2.0, fmin(16.0, 6e10/flops));
    tuner.run(lt,p.desc,&alpha,p.A,p.Ad,p.B,p.Bd,&beta,p.C,p.Cd,p.D,p.Dd,ws,wsSize,st);
    // interleave default/tuned blocks, take best of each: immune to clock drift
    double tDef=0, tTun=0;
    for (int r=0;r<3;r++){
        tDef = fmax(tDef, time_call(lt,p,alpha,beta,ws,wsSize,st,nullptr,iters));
        tTun = fmax(tTun, time_call(lt,p,alpha,beta,ws,wsSize,st,&tuner,iters));
    }
    printf("%-8s M=%-6d N=%-6d K=%-6d  default %7.1f  tuned %7.1f  speedup %5.2fx\n",
           name,M,N,K,tDef,tTun,tTun/tDef);
    // numeric check: default into D2, tuned is in p.D; compare (different algos -> fp order differs)
    {
        size_t n=(size_t)p.M*p.N;
        void* D2; CKC(cudaMalloc(&D2, n*esz(outt)));
        cublasLtMatrixLayout_t D2d; CKL(cublasLtMatrixLayoutCreate(&D2d,outt,p.M,p.N,p.M));
        CKL(cublasLtMatmul(lt,p.desc,&alpha,p.A,p.Ad,p.B,p.Bd,&beta,p.C,p.Cd,D2,D2d,nullptr,ws,wsSize,st));
        CKC(cudaStreamSynchronize(st));
        // compare on host as fp16 bit patterns decoded via half
        size_t es=esz(outt); std::vector<char> h1(n*es),h2(n*es);
        CKC(cudaMemcpy(h1.data(),p.D,n*es,cudaMemcpyDeviceToHost));
        CKC(cudaMemcpy(h2.data(),D2,n*es,cudaMemcpyDeviceToHost));
        double maxrel=0; int cnt=0;
        auto rd=[&](char*p_,int i)->double{ uint16_t u=((uint16_t*)p_)[i]; __half_raw hr;hr.x=u;return (double)__half2float(*(__half*)&hr); };
        if(es==2){ for(size_t i=0;i<n;i+=997){ double a=rd(h1.data(),i),b=rd(h2.data(),i); double d=fabs(a-b)/fmax(1e-6,fabs(b)); maxrel=fmax(maxrel,d); cnt++; } }
        printf("      numeric check: %zu sampled, max rel diff vs default = %.3g\n", cnt, maxrel);
        cudaFree(D2); cublasLtMatrixLayoutDestroy(D2d);
    }
    cudaFree(p.A);cudaFree(p.B);cudaFree(p.C);cudaFree(p.D);
    cublasLtMatmulDescDestroy(p.desc);
    cublasLtMatrixLayoutDestroy(p.Ad);cublasLtMatrixLayoutDestroy(p.Bd);
    cublasLtMatrixLayoutDestroy(p.Cd);cublasLtMatrixLayoutDestroy(p.Dd);
}

int main(){
    cublasLtHandle_t lt; CKL(cublasLtCreate(&lt));
    ltat::Options o; o.verbose = getenv("LTAT_VERBOSE")!=nullptr;
    cudaStream_t st; CKC(cudaStreamCreate(&st));
    void* ws; CKC(cudaMalloc(&ws, 256u<<20));
    ltat::Autotuner tuner(o);
    run_case(tuner, lt, "fp16", CUBLAS_COMPUTE_32F, CUDA_R_16F, CUDA_R_16F, 8192,8192,8192, ws, 256u<<20, st);
    run_case(tuner, lt, "fp16", CUBLAS_COMPUTE_32F, CUDA_R_16F, CUDA_R_16F, 4096,4096,4096, ws, 256u<<20, st);
    run_case(tuner, lt, "fp16", CUBLAS_COMPUTE_32F, CUDA_R_16F, CUDA_R_16F, 16384,16384,16384, ws, 256u<<20, st);
    run_case(tuner, lt, "fp16", CUBLAS_COMPUTE_32F, CUDA_R_16F, CUDA_R_16F, 8192,14336,4096, ws, 256u<<20, st);
    run_case(tuner, lt, "fp16", CUBLAS_COMPUTE_32F, CUDA_R_16F, CUDA_R_16F, 64,4096,4096, ws, 256u<<20, st);
    run_case(tuner, lt, "fp8 ", CUBLAS_COMPUTE_32F, CUDA_R_8F_E4M3, CUDA_R_16F, 8192,8192,8192, ws, 256u<<20, st);
    run_case(tuner, lt, "bf16", CUBLAS_COMPUTE_32F, CUDA_R_16BF, CUDA_R_16BF, 8192,8192,8192, ws, 256u<<20, st);
    run_case(tuner, lt, "tf32", CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, CUDA_R_32F, 8192,8192,8192, ws, 256u<<20, st);
    return 0;
}
