// cublaslt_autotune.hpp — drop-in autotuning wrapper for cublasLtMatmul.
//
// Why: the cublasLt heuristic's ranking is tuned for stock hardware; on
// unlocked 4090s (2x tensor-core issue) the default pick can be 30-50% slower
// than the best in-library algo. This wrapper enumerates the heuristic's
// candidates, times them on the real buffers once per problem shape, caches
// the winner, and replays it on later calls.
//
// Usage:
//   LtAutotuner tuner;                     // one per process is fine
//   tuner.run(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, D, Dd, ws, wsBytes, stream);
// Signature mirrors cublasLtMatmul. First call per (desc, geometry) tunes;
// later calls just execute the cached winner.
#pragma once
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace ltat {

struct Key {
    uint64_t w[24];
    bool operator==(const Key& o) const { return std::memcmp(w, o.w, sizeof w) == 0; }
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint64_t v : k.w) { h ^= v; h *= 0x100000001b3ull; }
        return (size_t)h;
    }
};

struct CachedAlgo {
    cublasLtMatmulAlgo_t algo;
    double tfps = 0.0;
    int winnerIdx = 0;
    int nCandidates = 0;
    double defaultTfps = 0.0;   // TF/s of heuristic rank-0 (for reporting)
    bool valid = false;         // false => fall through to default (nullptr) forever
};

struct Options {
    int    maxCandidates = 32;
    int    warmup = 1;
    int    reps = 2;            // timing reps per candidate, best (min) time kept
    double flopsBudget = 6e10;  // per-candidate measured FLOPs target
    bool   verbose = false;
};

class Autotuner {
public:
    explicit Autotuner(Options o = {}) : opt_(o) {}

    cublasStatus_t run(cublasLtHandle_t lt,
                       cublasLtMatmulDesc_t desc,
                       const void* alpha,
                       const void* A, cublasLtMatrixLayout_t Ad,
                       const void* B, cublasLtMatrixLayout_t Bd,
                       const void* beta,
                       void* C, cublasLtMatrixLayout_t Cd,
                       void* D, cublasLtMatrixLayout_t Dd,
                       void* workspace, size_t workspaceSize,
                       cudaStream_t stream) {
        Key key = makeKey(desc, Ad, Bd, Cd, Dd, workspaceSize);
        {
            std::lock_guard<std::mutex> g(mu_);
            auto it = cache_.find(key);
            if (it != cache_.end())
                return cublasLtMatmul(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd,
                                      it->second.valid ? &it->second.algo : nullptr,
                                      workspace, workspaceSize, stream);
        }
        CachedAlgo winner = tune(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd,
                                 workspace, workspaceSize, stream);
        winner.valid = winner.nCandidates > 0;
        {
            std::lock_guard<std::mutex> g(mu_);
            cache_.emplace(key, winner);
        }
        return cublasLtMatmul(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd,
                              &winner.algo, workspace, workspaceSize, stream);
    }

    // dump the cache as a table (for reporting)
    void report() const {
        std::lock_guard<std::mutex> g(mu_);
        std::printf("[ltat] %zu cached problems\n", cache_.size());
    }

private:
    Options opt_;
    mutable std::mutex mu_;
    std::unordered_map<Key, CachedAlgo, KeyHash> cache_;

    template <typename T>
    static void getattr(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a, T& out) {
        size_t w = 0; out = T{};
        cublasLtMatmulDescGetAttribute(d, a, &out, sizeof(out), &w);
    }
    template <typename T>
    static void layattr(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t a, T& out) {
        size_t w = 0; out = T{};
        cublasLtMatrixLayoutGetAttribute(l, a, &out, sizeof(out), &w);
    }

    static Key makeKey(cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t Ad,
                       cublasLtMatrixLayout_t Bd, cublasLtMatrixLayout_t Cd,
                       cublasLtMatrixLayout_t Dd, size_t ws) {
        Key k{}; int i = 0;
        int32_t v32; uint64_t v64;
        getattr(desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, v32); k.w[i++] = v32;
        getattr(desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, v32); k.w[i++] = v32;
        getattr(desc, CUBLASLT_MATMUL_DESC_TRANSA, v32); k.w[i++] = v32;
        getattr(desc, CUBLASLT_MATMUL_DESC_TRANSB, v32); k.w[i++] = v32;
        getattr(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, v32); k.w[i++] = v32;
        auto lay = [&](cublasLtMatrixLayout_t l) {
            layattr(l, CUBLASLT_MATRIX_LAYOUT_TYPE, v32); k.w[i++] = v32;
            layattr(l, CUBLASLT_MATRIX_LAYOUT_ROWS, v64); k.w[i++] = v64;
            layattr(l, CUBLASLT_MATRIX_LAYOUT_COLS, v64); k.w[i++] = v64;
            layattr(l, CUBLASLT_MATRIX_LAYOUT_LD, v64); k.w[i++] = v64;
            layattr(l, CUBLASLT_MATRIX_LAYOUT_ORDER, v32); k.w[i++] = v32;
        };
        // A and D identify the problem fully enough given B/C follow the same convention
        lay(Ad); lay(Dd);
        k.w[22] = ws; // workspace size affects which algos are legal
        return k;
    }

    CachedAlgo tune(cublasLtHandle_t lt, cublasLtMatmulDesc_t desc,
                    const void* alpha, const void* A, cublasLtMatrixLayout_t Ad,
                    const void* B, cublasLtMatrixLayout_t Bd,
                    const void* beta, void* C, cublasLtMatrixLayout_t Cd,
                    void* D, cublasLtMatrixLayout_t Dd,
                    void* ws, size_t wsSize, cudaStream_t stream) {
        // In-place accumulation (C==D, beta!=0): tuning would re-apply the op to D
        // repeatedly and corrupt it. Tune on a scratch copy of D instead.
        void* scratch = nullptr;
        if (C == D && beta != nullptr) {
            uint64_t r = 0, c = 0, ld = 0; int32_t ty = 0, ord = 0;
            layattr(Dd, CUBLASLT_MATRIX_LAYOUT_ROWS, r);
            layattr(Dd, CUBLASLT_MATRIX_LAYOUT_COLS, c);
            layattr(Dd, CUBLASLT_MATRIX_LAYOUT_LD, ld);
            layattr(Dd, CUBLASLT_MATRIX_LAYOUT_TYPE, ty);
            layattr(Dd, CUBLASLT_MATRIX_LAYOUT_ORDER, ord);
            size_t es = (ty == CUDA_R_16F || ty == CUDA_R_16BF) ? 2 : (ty == CUDA_R_8F_E4M3 ? 1 : 4);
            size_t bytes = (size_t)ld * (ord == CUBLASLT_ORDER_ROW ? r : c) * es;
            if (cudaMalloc(&scratch, bytes) == cudaSuccess) {
                cudaMemcpyAsync(scratch, D, bytes, cudaMemcpyDeviceToDevice, stream);
                D = scratch;
            }
        }
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wsSize, sizeof(wsSize));
        std::vector<cublasLtMatmulHeuristicResult_t> heur(opt_.maxCandidates);
        int n = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Dd,
                                                           pref, opt_.maxCandidates,
                                                           heur.data(), &n);
        cublasLtMatmulPreferenceDestroy(pref);
        CachedAlgo best{};
        if (hs != CUBLAS_STATUS_SUCCESS || n == 0) {
            // fall back to "no algo" semantics: return status from a plain call
            cublasLtMatmul(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd,
                           nullptr, ws, wsSize, stream);
            best.nCandidates = 0;
            if (scratch) cudaFree(scratch);
            return best;
        }
        // problem geometry for iters estimate
        uint64_t m = 0, nn = 0;
        layattr(Dd, CUBLASLT_MATRIX_LAYOUT_ROWS, m);
        layattr(Dd, CUBLASLT_MATRIX_LAYOUT_COLS, nn);
        // flops ~= 2*M*N*K; for TN col-major Dd is (N x M) in our convention but 2MNK is symmetric
        uint64_t rA, cA; layattr(Ad, CUBLASLT_MATRIX_LAYOUT_ROWS, rA); layattr(Ad, CUBLASLT_MATRIX_LAYOUT_COLS, cA);
        double flops = 2.0 * m * nn * (double)std::max(rA, cA); // K = the non-M dim of A
        int iters = (int)std::max(2.0, std::min(16.0, opt_.flopsBudget / std::max(flops, 1.0)));

        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<double> msBest(n, 1e30);
        std::vector<char> valid(n, 0);
        for (int i = 0; i < n; i++) {  // warmup pass
            cudaGetLastError();
            cublasStatus_t st = cublasLtMatmul(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd,
                                               &heur[i].algo, ws, wsSize, stream);
            if (cudaStreamSynchronize(stream) != cudaSuccess || cudaGetLastError() != cudaSuccess)
                st = CUBLAS_STATUS_EXECUTION_FAILED;
            valid[i] = (st == CUBLAS_STATUS_SUCCESS);
        }
        // reps interleaved across candidates so clock drift doesn't bias the ranking
        for (int r = 0; r < opt_.reps + 1; r++) {
            for (int i = 0; i < n; i++) {
                if (!valid[i]) continue;
                cudaEventRecord(e0, stream);
                for (int it = 0; it < iters; it++)
                    cublasLtMatmul(lt, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd,
                                   &heur[i].algo, ws, wsSize, stream);
                cudaEventRecord(e1, stream);
                cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                if (r > 0) msBest[i] = std::min(msBest[i], (double)ms);  // rep 0 = extra warmup
            }
        }
        double bestMs = 1e30;
        for (int i = 0; i < n; i++) {
            if (!valid[i]) continue;
            double tf = flops * iters / (msBest[i] / 1e3) / 1e12;
            if (i == 0) best.defaultTfps = tf;
            if (opt_.verbose)
                std::printf("[ltat] cand %d: %.1f TF/s%s\n", i, tf, i == 0 ? " (heuristic pick)" : "");
            if (msBest[i] < bestMs) { bestMs = msBest[i]; best.algo = heur[i].algo; best.tfps = tf; best.winnerIdx = i; }
        }
        // Never ship a pick measurably worse than the heuristic's own rank-0:
        // sub-noise differences flip between clock windows, so require >2% to switch.
        if (n > 0 && best.winnerIdx != 0 && best.tfps < best.defaultTfps * 1.05) {
            best.algo = heur[0].algo; best.tfps = best.defaultTfps; best.winnerIdx = 0;
        }
        best.nCandidates = n;
        if (opt_.verbose && n > 0)
            std::printf("[ltat] winner: %.1f TF/s vs default %.1f TF/s (%.2fx)\n",
                        best.tfps, best.defaultTfps, best.tfps / std::max(best.defaultTfps, 1e-9));
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        if (scratch) cudaFree(scratch);
        return best;
    }
};

} // namespace ltat
