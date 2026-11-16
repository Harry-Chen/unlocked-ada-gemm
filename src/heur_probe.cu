// heur_probe: query cublasLt heuristic picks WITHOUT running anything.
// Prints rank0..3 algo config per (dtype, shape). Host-only (needs CUDA ctx only).
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "lt_enum_names.h"

int main(){
    struct DT { const char* n; cudaDataType_t io, out; cublasComputeType_t c; };
    DT dts[] = {
        {"fp16", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F},
        {"bf16", CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F},
        {"tf32", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32},
        {"fp8",  CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F},
    };
    struct S { int M,N,K; } shapes[] = {
        {2048,2048,2048},{4096,4096,4096},{8192,8192,8192},{16384,16384,16384},
        {8192,14336,4096},{8192,4096,14336},{64,4096,4096},{1024,4096,4096},
    };
    cublasLtHandle_t lt;
    if (cublasLtCreate(&lt)) { printf("lt create fail\n"); return 1; }
    cudaFree(0); // establish ctx
    cudaDeviceProp prop; int dev; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    int clk=0, memclk=0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev);
    printf("device: %s cc=%d.%d SMs=%d clock=%dMHz memBW~%.0fGB/s\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, clk/1000,
           2.0*prop.memoryBusWidth/8*memclk*2/1e6);
    for (auto& d : dts) {
        for (auto& s : shapes) {
            cublasLtMatmulDesc_t desc;
            cublasLtMatmulDescCreate(&desc, d.c, CUDA_R_32F);
            cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT));
            cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN));
            cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
            cublasLtMatrixLayoutCreate(&Ad,d.io,s.K,s.M,s.K);
            cublasLtMatrixLayoutCreate(&Bd,d.io,s.K,s.N,s.K);
            cublasLtMatrixLayoutCreate(&Cd,d.out,s.M,s.N,s.M);
            cublasLtMatrixLayoutCreate(&Dd,d.out,s.M,s.N,s.M);
            cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
            size_t ws=256u<<20;
            cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws));
            std::vector<cublasLtMatmulHeuristicResult_t> heur(8);
            int n=0;
            cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt,desc,Ad,Bd,Cd,Dd,pref,8,heur.data(),&n);
            if (hs==CUBLAS_STATUS_SUCCESS && n>0){
                printf("%s %6dx%6dx%6d : ", d.n, s.M,s.N,s.K);
                for (int i=0;i<n && i<3;i++){
                    int cfg=-1,tile=-1,stg=-1,sk=-1; size_t w;
                    cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_ID,&cfg,sizeof(int),&w);
                    cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(int),&w);
                    cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stg,sizeof(int),&w);
                    cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&sk,sizeof(int),&w);
                    printf("[#%d cfg%d %s/%s/sk%d] ", i, cfg, tile_name(tile), stages_name(stg), sk);
                }
                printf("\n");
            } else printf("%s %6dx%6dx%6d : status=%d n=%d\n", d.n, s.M,s.N,s.K,(int)hs,n);
            cublasLtMatmulDescDestroy(desc); cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(Ad);cublasLtMatrixLayoutDestroy(Bd);
            cublasLtMatrixLayoutDestroy(Cd);cublasLtMatrixLayoutDestroy(Dd);
        }
    }
    return 0;
}
