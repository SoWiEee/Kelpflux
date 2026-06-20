// gpu_workload — parameterized real-CUDA workload for the live placement A/B.
//
// Replaces `sleep N` in live_ab_heavytail.py with an actual GPU job so the
// placement decision has real compute consequences: MPS co-residency
// interference, VRAM pressure, and heterogeneous-GPU speed all become visible
// in JCT (a `sleep` job hides all three — its runtime is placement-independent).
//
// Design — FIXED WORK, not fixed duration:
//   We run a fixed number of cuBLAS sgemm iterations. On an idle GPU this takes
//   ~target_s (calibrated offline → iters); under MPS contention the SAME iters
//   take LONGER, so interference extends JCT. (A self-timed "run until target_s"
//   loop would instead hide interference — it would just do less work.) The host
//   (live_ab) computes iters = round(true_runtime_s * ITERS_PER_SEC) from an
//   offline idle calibration and passes it in.
//
//   dim     sets per-iter compute intensity (FLOPs/iter = 2*dim^3) AND part of
//           the VRAM footprint (3*dim^2*4 bytes for A,B,C).
//   vram_mb adds an independent scratch allocation so VRAM pressure can be
//           dialed separately from compute (e.g. a "big-memory" job on the 10GB
//           3080 vs 12GB 4070).
//   seed    deterministic fill (CRN: pass the job_id so the same logical job is
//           byte-identical across arms).
//
// Build (inside a worker pod, nvidia/cuda:12.4.1-devel — has nvcc + cuBLAS):
//   nvcc -O3 -o /shared/bin/gpu_workload /shared/src/gpu_workload.cu -lcublas
//
// Usage:
//   gpu_workload <iters> <dim> <vram_mb> <seed>
//
// Exit codes: 0 ok; 2 bad args; 3 CUDA/cuBLAS error (incl. OOM → real VRAM
// constraint surfaces as a failed job, exactly as we want on the 10GB card).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
    return 3; } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS error %d at %s:%d\n",(int)s,__FILE__,__LINE__); \
    return 3; } } while(0)

// xorshift32 → deterministic per-seed fill, no host RNG dependency.
static inline float rnd(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (float)(s & 0xFFFFFF) / (float)0xFFFFFF;  // [0,1)
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <iters> <dim> <vram_mb> <seed>\n", argv[0]);
        return 2;
    }
    const long iters   = atol(argv[1]);
    const int  dim     = atoi(argv[2]);
    const long vram_mb = atol(argv[3]);
    uint32_t   seed    = (uint32_t)strtoul(argv[4], nullptr, 10);
    if (iters < 0 || dim <= 0 || vram_mb < 0) {
        fprintf(stderr, "bad args: iters>=0 dim>0 vram_mb>=0\n");
        return 2;
    }
    if (seed == 0) seed = 0x9E3779B9u;  // xorshift must not start at 0

    const size_t n = (size_t)dim * (size_t)dim;
    const size_t bytes = n * sizeof(float);

    // Host-fill A,B deterministically (CRN), C starts zero.
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    if (!hA || !hB) { fprintf(stderr, "host alloc failed\n"); return 3; }
    for (size_t i = 0; i < n; ++i) { hA[i] = rnd(seed); hB[i] = rnd(seed); }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // Independent VRAM pressure: touch it so it's truly resident.
    char *scratch = nullptr;
    if (vram_mb > 0) {
        const size_t sb = (size_t)vram_mb << 20;
        CUDA_CHECK(cudaMalloc(&scratch, sb));
        CUDA_CHECK(cudaMemset(scratch, 1, sb));
    }

    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    const float alpha = 1.0f, beta = 1.0f;  // C = A*B + C → C grows, no DCE

    for (long it = 0; it < iters; ++it) {
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                 dim, dim, dim,
                                 &alpha, dA, dim, dB, dim,
                                 &beta, dC, dim));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read one element back so the loop can't be optimized away.
    float probe = 0.0f;
    CUDA_CHECK(cudaMemcpy(&probe, dC, sizeof(float), cudaMemcpyDeviceToHost));
    printf("done iters=%ld dim=%d vram_mb=%ld seed=%u probe=%.3f\n",
           iters, dim, vram_mb, seed, probe);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (scratch) cudaFree(scratch);
    free(hA); free(hB);
    return 0;
}
