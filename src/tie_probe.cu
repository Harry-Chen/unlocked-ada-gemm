// RN-vs-RZ tie probe on bf16 (wide exponent range, no fp16 subnormal issues).
// D = 1 + 3*2^-25 exactly midway between 1+2^-24 and 1+2^-23.
//   RN-to-even -> 1+2^-23 (even mantissa)  => D-1 = +1.1921e-07
//   RZ         -> 1+2^-24                  => D-1 = +5.9605e-08
// Also: D = 1 + 2^-24 + 2^-24 (two tie terms sum up -> NOT a tie anymore, exact = 1+2^-23)
//   distinguishes rounding-vs-truncation of the multi-term adder tree.
#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
using namespace nvcuda;
__global__ void k(const __nv_bfloat16* A, const __nv_bfloat16* B, float* D){
    wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16,16,16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, A, 16);
    wmma::load_matrix_sync(fb, B, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(D, fc, 16, wmma::mem_row_major);
}
int main(){
    for (int variant=0; variant<2; variant++){
        __nv_bfloat16 hA[256]={}, hB[256]={}; float hD[256];
        hA[0]=__float2bfloat16(1.0f); hB[0]=__float2bfloat16(1.0f);       // p0=1
        if (variant==0){ hA[1]=__float2bfloat16(1.5f); hB[1]=__float2bfloat16(0x1.0p-24f); } // p1=3*2^-25? no: 1.5*2^-24 = 3*2^-25
        else { hA[1]=__float2bfloat16(1.0f); hB[1]=__float2bfloat16(0x1.0p-24f);             // p1=2^-24
               hA[2]=__float2bfloat16(1.0f); hB[2]=__float2bfloat16(0x1.0p-24f); }           // p2=2^-24
        __nv_bfloat16 *dA,*dB; float *dD;
        cudaMalloc(&dA,512); cudaMalloc(&dB,512); cudaMalloc(&dD,1024);
        cudaMemcpy(dA,hA,512,cudaMemcpyHostToDevice); cudaMemcpy(dB,hB,512,cudaMemcpyHostToDevice);
        k<<<1,32>>>(dA,dB,dD);
        cudaMemcpy(hD,dD,1024,cudaMemcpyDeviceToHost);
        if (variant==0) printf("tie 1+3*2^-25: D-1=%+.9e  -> %s (RN-even expects 1.1921e-07, RZ expects 5.9605e-08)\n",
                               hD[0]-1.0f, (hD[0]-1.0f>1e-7)?"RN":"RZ");
        else printf("two-ties 1+2^-24+2^-24: D-1=%+.9e (exact 1.1921e-07; partial-drop would give less)\n", hD[0]-1.0f);
        cudaFree(dA);cudaFree(dB);cudaFree(dD);
    }
    return 0;
}
