// gemm_precision.cu — GEMM-level accumulation precision vs fp64 reference, K sweep.
// fp16 (HMMA path) and fp8 e4m3 (QMMA path), cuBLASLt default-pick vs autotuned-pick.
// TN layout: A col-major (K x M), B col-major (K x N), D col-major (M x N), opA=T, opB=N.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "cublaslt_autotune.hpp"

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
#define CKL(x) do { cublasStatus_t s=(x); if(s){printf("cuBLAS err %d @%d\n",(int)s,__LINE__); exit(1);} } while(0)

int main(){
    const int M=256, N=256;
    int Ks[] = {1024, 4096, 16384};
    cublasLtHandle_t lt; CKL(cublasLtCreate(&lt));
    cudaStream_t st; CKC(cudaStreamCreate(&st));
    void* ws; CKC(cudaMalloc(&ws, 64u<<20));
    ltat::Autotuner tuner;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);

    printf("dtype,path,K,mean_signed_relerr,p50_abs_relerr,max_abs_relerr\n");
    for (int dtype_i=0; dtype_i<2; dtype_i++){
        cudaDataType_t io = dtype_i==0?CUDA_R_16F:CUDA_R_8F_E4M3;
        const char* dn = dtype_i==0?"fp16":"fp8e4m3";
        for (int K : Ks) {
            size_t nA=(size_t)M*K, nB=(size_t)K*N;
            // host float copies holding the EXACT values the GPU will see
            std::vector<float> hAf(nA), hBf(nB);
            void *dA,*dB,*dC,*dD;
            CKC(cudaMalloc(&dA,nA*(io==CUDA_R_16F?2:1))); CKC(cudaMalloc(&dB,nB*(io==CUDA_R_16F?2:1)));
            CKC(cudaMalloc(&dC,(size_t)M*N*2)); CKC(cudaMalloc(&dD,(size_t)M*N*2));
            if (io==CUDA_R_16F){
                std::vector<__half> tA(nA), tB(nB);
                for (size_t i=0;i<nA;i++){ tA[i]=__float2half(ud(rng)); hAf[i]=__half2float(tA[i]); }
                for (size_t i=0;i<nB;i++){ tB[i]=__float2half(ud(rng)); hBf[i]=__half2float(tB[i]); }
                CKC(cudaMemcpy(dA,tA.data(),nA*2,cudaMemcpyHostToDevice));
                CKC(cudaMemcpy(dB,tB.data(),nB*2,cudaMemcpyHostToDevice));
            } else {
                std::vector<__nv_fp8_storage_t> tA(nA), tB(nB);
                for (size_t i=0;i<nA;i++){ tA[i]=__nv_cvt_float_to_fp8(ud(rng),__NV_SATFINITE,__NV_E4M3);
                    __half_raw hr=__nv_cvt_fp8_to_halfraw(tA[i],__NV_E4M3); hAf[i]=__half2float(*(__half*)&hr); }
                for (size_t i=0;i<nB;i++){ tB[i]=__nv_cvt_float_to_fp8(ud(rng),__NV_SATFINITE,__NV_E4M3);
                    __half_raw hr=__nv_cvt_fp8_to_halfraw(tB[i],__NV_E4M3); hBf[i]=__half2float(*(__half*)&hr); }
                CKC(cudaMemcpy(dA,tA.data(),nA,cudaMemcpyHostToDevice));
                CKC(cudaMemcpy(dB,tB.data(),nB,cudaMemcpyHostToDevice));
            }
            cublasLtMatmulDesc_t desc;
            CKL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT));
            cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN));
            cublasLtMatrixLayout_t Ad,Bd,Cd,Dd;
            CKL(cublasLtMatrixLayoutCreate(&Ad,io,K,M,K));
            CKL(cublasLtMatrixLayoutCreate(&Bd,io,K,N,K));
            CKL(cublasLtMatrixLayoutCreate(&Cd,CUDA_R_16F,M,N,M));
            CKL(cublasLtMatrixLayoutCreate(&Dd,CUDA_R_16F,M,N,M));
            float alpha=1,beta=0;
            for (int path=0; path<2; path++){
                if (path==0) CKL(cublasLtMatmul(lt,desc,&alpha,dA,Ad,dB,Bd,&beta,dC,Cd,dD,Dd,nullptr,ws,64u<<20,st));
                else CKL(tuner.run(lt,desc,&alpha,dA,Ad,dB,Bd,&beta,dC,Cd,dD,Dd,ws,64u<<20,st));
                CKC(cudaStreamSynchronize(st));
                std::vector<__half> out((size_t)M*N);
                CKC(cudaMemcpy(out.data(),dD,(size_t)M*N*2,cudaMemcpyDeviceToHost));
                // fp64 reference: D[m][n] = sum_k A_cm[k][m] * B_cm[k][n];  A_cm[k][m] at hAf[k + m*K], B_cm[k][n] at hBf[k + n*K]
                double sb=0, mx=0; long cnt=0; std::vector<double> rels;
                for (int m=0;m<M;m++) for(int n=0;n<N;n++){
                    double e=0;
                    for(int k=0;k<K;k++) e += (double)hAf[(size_t)k+(size_t)m*K] * (double)hBf[(size_t)k+(size_t)n*K];
                    double g=(double)__half2float(out[(size_t)m+(size_t)n*M]);
                    double denom=fmax(1e-2, fabs(e));
                    double r=(g-e)/denom;
                    sb+=r; double ar=fabs(r); mx=fmax(mx,ar); rels.push_back(ar); cnt++;
                }
                std::sort(rels.begin(),rels.end());
                printf("%s,%s,%d,%+.4e,%.4e,%.4e\n", dn, path?"tuned":"default", K, sb/cnt, rels[cnt/2], mx);
            }
            cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dD);
            cublasLtMatmulDescDestroy(desc);
            cublasLtMatrixLayoutDestroy(Ad);cublasLtMatrixLayoutDestroy(Bd);
            cublasLtMatrixLayoutDestroy(Cd);cublasLtMatrixLayoutDestroy(Dd);
        }
    }
    return 0;
}
