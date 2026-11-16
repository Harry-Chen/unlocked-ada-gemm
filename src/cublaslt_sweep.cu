// cuBLASLt full sweep: per (dtype, M,N,K), enumerate heuristic algos, decode
// tile/stage/splitk config, measure TF/s. Output: CSV to stdout.
// Usage: cublaslt_sweep <dtype:fp16|bf16|tf32|fp8> [maxAlgos]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "lt_enum_names.h"

#define CKC(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

static size_t esz(cudaDataType_t t){ switch(t){case CUDA_R_16F:case CUDA_R_16BF:return 2;case CUDA_R_8F_E4M3:return 1;default:return 4;} }

struct Shape { int M,N,K; };

int main(int argc, char** argv){
    if (argc<2){fprintf(stderr,"need dtype\n");return 1;}
    const char* prec = argv[1];
    int maxAlgos = argc>2?atoi(argv[2]):32;
    cudaDataType_t io, outt; cublasComputeType_t comp;
    if (!strcmp(prec,"fp16")){io=CUDA_R_16F;outt=CUDA_R_16F;comp=CUBLAS_COMPUTE_32F;}
    else if (!strcmp(prec,"bf16")){io=CUDA_R_16BF;outt=CUDA_R_16BF;comp=CUBLAS_COMPUTE_32F;}
    else if (!strcmp(prec,"tf32")){io=CUDA_R_32F;outt=CUDA_R_32F;comp=CUBLAS_COMPUTE_32F_FAST_TF32;}
    else if (!strcmp(prec,"fp8")){io=CUDA_R_8F_E4M3;outt=CUDA_R_16F;comp=CUBLAS_COMPUTE_32F;}
    else {fprintf(stderr,"bad dtype\n");return 1;}

    std::vector<Shape> shapes = {
        {2048,2048,2048},{4096,4096,4096},{8192,8192,8192},{16384,16384,16384},
        {8192,14336,4096},{8192,4096,14336},{2048,14336,4096},
        {64,4096,4096},{256,4096,4096},{1024,4096,4096},
        {64,8192,8192},{1024,8192,8192},
    };

    cublasLtHandle_t lt; CKC((cudaError_t)cublasLtCreate(&lt));
    cudaStream_t stream; CKC(cudaStreamCreate(&stream));
    void* workspace; CKC(cudaMalloc(&workspace, 256u<<20));

    printf("dtype,M,N,K,heurRank,algoCfgId,tile,stages,splitK,TFps\n");
    for (auto& s : shapes) {
        int M=s.M,N=s.N,K=s.K;
        void *A,*B,*C,*D;
        CKC(cudaMalloc(&A,(size_t)K*M*esz(io))); CKC(cudaMalloc(&B,(size_t)K*N*esz(io)));
        CKC(cudaMalloc(&C,(size_t)M*N*esz(outt))); CKC(cudaMalloc(&D,(size_t)M*N*esz(outt)));
        CKC(cudaMemset(A,0x11,(size_t)K*M*esz(io))); CKC(cudaMemset(B,0x11,(size_t)K*N*esz(io)));
        cublasLtMatmulDesc_t desc;
        CKC((cudaError_t)cublasLtMatmulDescCreate(&desc, comp, CUDA_R_32F));
        cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN));
        cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
        cublasLtMatrixLayoutCreate(&Ad,io,K,M,K);
        cublasLtMatrixLayoutCreate(&Bd,io,K,N,K);
        cublasLtMatrixLayoutCreate(&Cd,outt,M,N,M);
        cublasLtMatrixLayoutCreate(&Dd,outt,M,N,M);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t ws=256u<<20;
        cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws));
        std::vector<cublasLtMatmulHeuristicResult_t> heur(maxAlgos);
        int nres=0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt,desc,Ad,Bd,Cd,Dd,pref,maxAlgos,heur.data(),&nres);
        if (hs!=CUBLAS_STATUS_SUCCESS || nres==0){
            printf("%s,%d,%d,%d,ERR status=%d\n", prec,M,N,K,(int)hs);
        }
        float alpha=1.0f,beta=0.0f;
        for (int i=0;i<nres;i++){
            int cfgId=-1,tileId=-1,stagesId=-1,splitk=-1; size_t wr;
            cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_ID,&cfgId,sizeof(int),&wr);
            cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tileId,sizeof(int),&wr);
            cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stagesId,sizeof(int),&wr);
            cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&splitk,sizeof(int),&wr);
            // warmup + validate
            cudaGetLastError();
            cublasStatus_t st = cublasLtMatmul(lt,desc,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&heur[i].algo,workspace,ws,stream);
            CKC(cudaStreamSynchronize(stream));
            if (st!=CUBLAS_STATUS_SUCCESS || cudaGetLastError()!=cudaSuccess){ printf("%s,%d,%d,%d,%d,%d,%s,%s,%d,RUNFAIL\n",prec,M,N,K,i,cfgId,tile_name(tileId),stages_name(stagesId),splitk); continue; }
            double flops = 2.0*M*N*K;
            int iters = (int)std::max(2.0, std::min(12.0, 1.5e11/flops));
            cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0,stream);
            for(int k=0;k<iters;k++) cublasLtMatmul(lt,desc,&alpha,A,Ad,B,Bd,&beta,C,Cd,D,Dd,&heur[i].algo,workspace,ws,stream);
            cudaEventRecord(e1,stream); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms,e0,e1);
            double tf = flops*iters/(ms/1e3)/1e12;
            printf("%s,%d,%d,%d,%d,%d,%s,%s,%d,%.1f\n",prec,M,N,K,i,cfgId,tile_name(tileId),stages_name(stagesId),splitk,tf);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        fflush(stdout);
        cudaFree(A);cudaFree(B);cudaFree(C);cudaFree(D);
        cublasLtMatmulDescDestroy(desc); cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Ad);cublasLtMatrixLayoutDestroy(Bd);
        cublasLtMatrixLayoutDestroy(Cd);cublasLtMatrixLayoutDestroy(Dd);
    }
    return 0;
}
