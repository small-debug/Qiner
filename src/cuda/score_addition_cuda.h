#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Smoke test: runs a trivial CUDA kernel and returns 1 on success, 0 on failure.
 * Used to verify CUDA build and runtime.
 */
int score_addition_cuda_smoke_test(void);

/**
 * Batch score computation on GPU.
 * pool: from generateRandom2Pool(miningSeed)
 * publicKey: 32 bytes
 * nonces: numNonces * 32 bytes
 * scores: output, numNonces elements
 * Returns 1 on success, 0 on failure.
 */
int score_addition_cuda_batch(
    const unsigned char* pool,
    const unsigned char* publicKey,
    const unsigned char* nonces,
    unsigned int* scores,
    size_t numNonces);

#ifdef __cplusplus
}
#endif
