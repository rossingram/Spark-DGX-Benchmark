// gb10_gemm_bf16.cu
// Simple cublasLt BF16 GEMM microbenchmark for NVIDIA GB10 / Spark DGX

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>

#include <cuda_runtime.h>
#include <cublasLt.h>

#define CHECK_CUDA(expr)                                         \
    do {                                                         \
        cudaError_t _err = (expr);                               \
        if (_err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",          \
                    cudaGetErrorString(_err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                             \
        }                                                        \
    } while (0)

#define CHECK_CUBLAS(expr)                                       \
    do {                                                         \
        cublasStatus_t _st = (expr);                             \
        if (_st != CUBLAS_STATUS_SUCCESS) {                      \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",        \
                    (int)_st, __FILE__, __LINE__);               \
            std::exit(EXIT_FAILURE);                             \
        }                                                        \
    } while (0)

int main(int argc, char** argv) {
    // Default: 8192 x 8192 GEMM, 50 iterations
    int M = 8192;
    int N = 8192;
    int K = 8192;
    int iters = 50;

    if (argc >= 5) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
        iters = std::atoi(argv[4]);
    }

    printf("BF16 GEMM: M=%d, N=%d, K=%d, iters=%d\n", M, N, K, iters);

    // Init cuBLASLt
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Allocate BF16 matrices on device
    size_t bytesA = (size_t)M * K * sizeof(__nv_bfloat16);
    size_t bytesB = (size_t)K * N * sizeof(__nv_bfloat16);
    size_t bytesC = (size_t)M * N * sizeof(__nv_bfloat16);

    __nv_bfloat16* dA = nullptr;
    __nv_bfloat16* dB = nullptr;
    __nv_bfloat16* dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    // Initialize with some values (we don't care what)
    CHECK_CUDA(cudaMemset(dA, 0, bytesA));
    CHECK_CUDA(cudaMemset(dB, 0, bytesB));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    // GEMM: C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;

    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t layoutA = nullptr;
    cublasLtMatrixLayout_t layoutB = nullptr;
    cublasLtMatrixLayout_t layoutC = nullptr;

    // Compute type: FP32 accumulation, BF16 inputs/outputs
    CHECK_CUBLAS(cublasLtMatmulDescCreate(
        &matmulDesc,
        CUBLAS_COMPUTE_32F,  // accumulate in FP32
        CUDA_R_32F));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &opA,
        sizeof(opA)));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &opB,
        sizeof(opB)));

    // Matrix layouts (row-major via leading dimensions)
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &layoutA,
        CUDA_R_16BF,
        M, K, K));  // lda = K

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &layoutB,
        CUDA_R_16BF,
        K, N, N));  // ldb = N

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &layoutC,
        CUDA_R_16BF,
        M, N, N));  // ldc = N

    // Choose heuristics (we’ll let cuBLASLt pick a kernel)
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));

    size_t workspaceSize = 1 << 26; // 64MB
    void* dWorkspace = nullptr;
    CHECK_CUDA(cudaMalloc(&dWorkspace, workspaceSize));

    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof(workspaceSize)));

    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    cublasLtMatmulHeuristicResult_t heuristics[requestedAlgoCount];

    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmulDesc,
        layoutA,
        layoutB,
        layoutC,
        layoutC,
        preference,
        requestedAlgoCount,
        heuristics,
        &returnedAlgoCount));

    if (returnedAlgoCount == 0) {
        fprintf(stderr, "No suitable cuBLASLt heuristic found.\n");
        return EXIT_FAILURE;
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            matmulDesc,
            &alpha,
            dA, layoutA,
            dB, layoutB,
            &beta,
            dC, layoutC,
            dC, layoutC,
            &heuristics[0].algo,
            dWorkspace,
            workspaceSize,
            0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            matmulDesc,
            &alpha,
            dA, layoutA,
            dB, layoutB,
            &beta,
            dC, layoutC,
            dC, layoutC,
            &heuristics[0].algo,
            dWorkspace,
            workspaceSize,
            0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt = t1 - t0;
    double seconds = dt.count();

    // FLOPs = 2 * M * N * K * iters
    double flops = 2.0 * (double)M * (double)N * (double)K * (double)iters;
    double tflops = flops / seconds / 1.0e12;

    printf("Total time: %.3f s for %d GEMMs\n", seconds, iters);
    printf("Approx BF16 TFLOPs: %.2f\n", tflops);

    // Cleanup
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
    CHECK_CUBLAS(cublasLtDestroy(handle));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dWorkspace));

    return 0;
}
