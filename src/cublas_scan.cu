// Enumerate cublasLt algos for a given precision/shape, time each, report all.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

static size_t esz(cudaDataType_t t){ switch(t){case CUDA_R_16F: case CUDA_R_16BF: return 2; case CUDA_R_8F_E4M3: return 1; default: return 4;} }

int main(int argc, char** argv){
    int S = argc>1 ? atoi(argv[1]) : 8192;
    const char* prec = argc>2 ? argv[2] : "fp16";
    cudaDataType_t io = CUDA_R_16F;
    cublasComputeType_t comp = CUBLAS_COMPUTE_32F;
    if (!strcmp(prec,"bf16")) io = CUDA_R_16BF;
    else if (!strcmp(prec,"fp8")) { io = CUDA_R_8F_E4M3; }
    else if (!strcmp(prec,"tf32")) { io = CUDA_R_32F; comp = CUBLAS_COMPUTE_32F_FAST_TF32; }
    int M=S,N=S,K=S;
    cudaDataType_t outt = CUDA_R_16F;
    if (!strcmp(prec,"bf16")) outt = CUDA_R_16BF;
    else if (!strcmp(prec,"tf32")) outt = CUDA_R_32F;
    printf("scan: %s %d^3\n", prec, S);

    cublasLtHandle_t lt; CKC((cudaError_t)cublasLtCreate(&lt));
    cudaStream_t stream; CKC(cudaStreamCreate(&stream));
    void* workspace; CKC(cudaMalloc(&workspace, 256u<<20));

    void *A,*B,*C,*D;
    CKC(cudaMalloc(&A,(size_t)K*M*esz(io))); CKC(cudaMalloc(&B,(size_t)K*N*esz(io)));
    CKC(cudaMalloc(&C,(size_t)M*N*esz(outt))); CKC(cudaMalloc(&D,(size_t)M*N*esz(outt)));
    CKC(cudaMemset(A,0x11,(size_t)K*M*esz(io))); CKC(cudaMemset(B,0x11,(size_t)K*N*esz(io)));

    cublasLtMatmulDesc_t desc;
    CKC((cudaError_t)cublasLtMatmulDescCreate(&desc, comp, CUDA_R_32F));
    cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
    CKC((cudaError_t)cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT)));
    CKC((cudaError_t)cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN)));
    cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
    CKC((cudaError_t)cublasLtMatrixLayoutCreate(&Ad,io,K,M,K));
    CKC((cudaError_t)cublasLtMatrixLayoutCreate(&Bd,io,K,N,K));
    CKC((cudaError_t)cublasLtMatrixLayoutCreate(&Cd,outt,M,N,M));
    CKC((cudaError_t)cublasLtMatrixLayoutCreate(&Dd,outt,M,N,M));
    cublasLtMatmulPreference_t pref;
    CKC((cudaError_t)cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 64u<<20;
    CKC((cudaError_t)cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));

    const int MAXA=32;
    cublasLtMatmulHeuristicResult_t heur[MAXA];
    int nres=0;
    { cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt,desc,Ad,Bd,Cd,Dd,pref,MAXA,heur,&nres);
      printf("heuristic status=%d nres=%d\n", (int)hs, nres); fflush(stdout);
      if (hs!=CUBLAS_STATUS_SUCCESS) return 1; }
    printf("heuristic returned %d algos\n", nres);

    float alpha=1.0f,beta=0.0f;
    struct R { int idx; double tf; int ok; } res[MAXA];
    for (int i=0;i<nres;i++){
        res[i].idx=i; res[i].ok=1; res[i].tf=0;
        cudaError_t pre = cudaGetLastError();
        cublasStatus_t st = cublasLtMatmul(lt,desc,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&heur[i].algo,workspace,ws,stream);
        CKC(cudaStreamSynchronize(stream));
        if (st!=CUBLAS_STATUS_SUCCESS || cudaGetLastError()!=cudaSuccess){ res[i].ok=0; printf("algo %2d: run failed (status %d)\n", i, (int)st); continue; }
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0,stream);
        int iters = (S>=16384)?4:6;
        for(int k=0;k<iters;k++) cublasLtMatmul(lt,desc,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&heur[i].algo,workspace,ws,stream);
        cudaEventRecord(e1,stream); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms,e0,e1);
        res[i].tf = 2.0*M*N*K*iters/(ms/1e3)/1e12;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    std::sort(res, res+nres, [](const R&a,const R&b){return a.tf>b.tf;});
    for (int i=0;i<nres && i<12;i++)
        if (res[i].ok) printf("rank by speed %2d  heuristic#%2d  %8.1f TF/s%s\n", i, res[i].idx, res[i].tf, res[i].idx==0?"  <- heuristic pick":"");
    return 0;
}
