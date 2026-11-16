// precision_probe.cu — tensor-core accumulator precision probe.
// Directed: D = 1 + 2^-q (single mma k-step, or via C-input path). Threshold q where the
//           small term is lost => effective accumulator mantissa width.
// Random:   random power-of-two magnitudes; rel-error distribution + systematic bias (RZ fingerprint).
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using namespace nvcuda;

#define CKC(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

// one warp per case; A/B/C laid out 16x16 per case
template<typename Tin, typename Tacc, bool USE_C>
__global__ void probe(const Tin* A, const Tin* B, const float* Cf, Tacc* D, int ldD) {
    int w = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    wmma::fragment<wmma::matrix_a, 16,16,16, Tin, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16,16,16, Tin, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16,16,16, Tacc> fc;
    if (USE_C) wmma::load_matrix_sync(fc, Cf + w*256, 16, wmma::mem_row_major);
    else wmma::fill_fragment(fc, (Tacc)0);
    wmma::load_matrix_sync(fa, A + w*256, 16);
    wmma::load_matrix_sync(fb, B + w*256, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(D + w*256, fc, ldD, wmma::mem_row_major);
}

template<typename Tin>
void fill_case(Tin* A, Tin* B, int w, double aval, double bval, double a0, double b0){
    // A[0][0]=a0, B[0][0]=b0 (product p0), A[0][1]=aval, B[1][0]=bval (product p1); rest zero
    Tin z{}; z = (Tin)0; // zero-init handled by caller
}

// run directed sweep for one dtype; returns gpu value of D[0][0] per q
template<typename Tin>
void run_directed(const char* name){
    std::vector<int> qs; for (int q=18;q<=30;q++) qs.push_back(q);
    int nAB = qs.size(), nC = qs.size(), n = nAB+nC;
    std::vector<Tin> hA(n*256), hB(n*256); std::vector<float> hC(n*256); std::vector<float> hD(n*256);
    std::vector<double> exact(n);
    for (int i=0;i<n;i++){
        // zero
        for (int e=0;e<256;e++){ hA[i*256+e]=(Tin)0.0f; hB[i*256+e]=(Tin)0.0f; hC[i*256+e]=0.0f; }
    }
    for (size_t i=0;i<qs.size();i++){
        int q = qs[i];
        // AB path: p0=1 (A[0][0]=1,B[0][0]=1); p1=2^-q via 2^-14 * 2^-(q-14) (fp16-safe down to q=28)
        double a1 = std::ldexp(1.0, -14), b1 = std::ldexp(1.0, -(q-14));
        hA[i*256 + 0*16+0] = (Tin)1.0f; hB[i*256 + 0] = (Tin)1.0f;
        hA[i*256 + 0*16+1] = (Tin)a1;   hB[i*256 + 1] = (Tin)b1;
        exact[i] = 1.0 + std::ldexp(1.0, -q);
        // C path: C=1, product p1=2^-q
        int j = nAB+i;
        hC[j*256 + 0] = 1.0f;
        hA[j*256 + 0*16+1] = (Tin)a1;   hB[j*256 + 1] = (Tin)b1;
        exact[j] = 1.0 + std::ldexp(1.0, -q);
    }
    Tin *dA,*dB; float *dC,*dD;
    CKC(cudaMalloc(&dA, n*256*sizeof(Tin))); CKC(cudaMalloc(&dB, n*256*sizeof(Tin)));
    CKC(cudaMalloc(&dC, n*256*4)); CKC(cudaMalloc(&dD, n*256*4));
    CKC(cudaMemcpy(dA,hA.data(),n*256*sizeof(Tin),cudaMemcpyHostToDevice));
    CKC(cudaMemcpy(dB,hB.data(),n*256*sizeof(Tin),cudaMemcpyHostToDevice));
    CKC(cudaMemcpy(dC,hC.data(),n*256*4,cudaMemcpyHostToDevice));
    // two kernels: AB (C=0) for i<nAB; C-path for i>=nAB
    dim3 grid((nAB*32+63)/64), blk(64);
    probe<Tin,float,false><<<grid,blk>>>(dA,dB,dC,dD,16);
    probe<Tin,float,true><<<grid,blk>>>(dA + nAB*256, dB + nAB*256, dC + nAB*256, dD + nAB*256, 16);
    CKC(cudaMemcpy(hD.data(),dD,n*256*4,cudaMemcpyDeviceToHost));
    printf("%s directed: q: gpu_vs_exact  (AB path | C path)\n", name);
    for (size_t i=0;i<qs.size();i++){
        int q=qs[i];
        double gAB=hD[i*256+0], gC=hD[nAB*256+i*256+0];
        printf("  q=%2d  AB: D-1=%+.4e exact-1=%.4e %s | C: D-1=%+.4e %s\n", q,
               gAB-1.0, exact[i]-1.0, (gAB==exact[i])?"exact":((gAB==1.0)?"LOST":(gAB<exact[i]?"TRUNC-DN":"DIFF")),
               gC-1.0, (gC==exact[i])?"exact":((gC==1.0)?"LOST":(gC<exact[i]?"TRUNC-DN":"DIFF")));
    }
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dD);
}

// random-pattern statistics
template<typename Tin>
void run_random(const char* name, int ncases){
    std::mt19937_64 rng(1234);
    std::vector<Tin> hA(ncases*256), hB(ncases*256); std::vector<float> hC(ncases*256,0.0f), hD(ncases*256);
    std::vector<double> ex(ncases*256, 0.0);
    std::vector<float> f32(ncases*256, 0.0f);
    std::uniform_int_distribution<int> exd(-4,4), sd(0,1);
    for (int c=0;c<ncases;c++){
        for (int i=0;i<16;i++) for(int k=0;k<16;k++)
            hA[c*256+i*16+k]=(Tin)std::ldexp(sd(rng)?1.0:-1.0, exd(rng));
        for (int k=0;k<16;k++) for(int j=0;j<16;j++)
            hB[c*256+k+j*16]=(Tin)std::ldexp(sd(rng)?1.0:-1.0, exd(rng));
        for (int i=0;i<16;i++) for(int j=0;j<16;j++){
            double es=0, ef32=0;
            for (int k=0;k<16;k++){
                double a=(double)(float)hA[c*256+i*16+k], b=(double)(float)hB[c*256+k+j*16];
                es += a*b; ef32 += (float)((float)a*(float)b); // fp32 RN control (product exact, sum RN)
            }
            ex[c*256+i*16+j]=es; f32[c*256+i*16+j]=ef32;
        }
    }
    Tin *dA,*dB; float *dC,*dD;
    CKC(cudaMalloc(&dA, ncases*256*sizeof(Tin))); CKC(cudaMalloc(&dB, ncases*256*sizeof(Tin)));
    CKC(cudaMalloc(&dC, ncases*256*4)); CKC(cudaMalloc(&dD, ncases*256*4));
    CKC(cudaMemcpy(dA,hA.data(),ncases*256*sizeof(Tin),cudaMemcpyHostToDevice));
    CKC(cudaMemcpy(dB,hB.data(),ncases*256*sizeof(Tin),cudaMemcpyHostToDevice));
    CKC(cudaMemcpy(dC,hC.data(),ncases*256*4,cudaMemcpyHostToDevice));
    dim3 grid((ncases*32+63)/64), blk(64);
    probe<Tin,float,false><<<grid,blk>>>(dA,dB,dC,dD,16);
    CKC(cudaMemcpy(hD.data(),dD,ncases*256*4,cudaMemcpyDeviceToHost));
    // DEBUG case 0 element (0,0): recompute from the actual host arrays
    {
        double chk=0; for(int k=0;k<16;k++) chk += (double)(float)hA[0*16+k] * (double)(float)hB[k+0*16];
        printf("[dbg] case0 D00: gpu=%.9g  ex=%.9g  hostRecompute=%.9g\n", hD[0], ex[0], chk);
        double chk01=0; for(int k=0;k<16;k++) chk01 += (double)(float)hA[0*16+k] * (double)(float)hB[k+1*16];
        printf("[dbg] case0 D01: gpu=%.9g  ex=%.9g  hostRecompute=%.9g\n", hD[1], ex[1], chk01);
    }
    // stats over elements with |exact| not tiny
    double sumBias=0, maxrel=0, sumBiasF=0, maxrelF=0; long cnt=0;
    double med_e=0; std::vector<double> rels, relsF;
    for (size_t i=0;i<ex.size();i++){
        double d0 = std::fabs(ex[i]); if (d0 < 1e-3) continue;
        double r = (hD[i]-ex[i])/d0, rf = (f32[i]-ex[i])/d0;
        sumBias += r; sumBiasF += rf;
        maxrel = fmax(maxrel, fabs(r)); maxrelF = fmax(maxrelF, fabs(rf));
        rels.push_back(fabs(r)); relsF.push_back(fabs(rf)); cnt++;
    }
    std::sort(rels.begin(),rels.end()); std::sort(relsF.begin(),relsF.end());
    printf("%s random (%ld samples): GPU  bias=%+.3e  p50|rel|=%.3e  max|rel|=%.3e | fp32-RN bias=%+.3e p50=%.3e max=%.3e\n",
           name, cnt, sumBias/cnt, rels[cnt/2], maxrel, sumBiasF/cnt, relsF[cnt/2], maxrelF);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dD);
}

int main(){
    run_directed<__half>("fp16");
    run_directed<__nv_bfloat16>("bf16");
    run_random<__half>("fp16", 512);
    run_random<__nv_bfloat16>("bf16", 512);
    return 0;
}
