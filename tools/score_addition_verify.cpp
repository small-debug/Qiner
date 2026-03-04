#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <array>

#include "../src/score_addition.h"
#include "../src/score_common.h"
#include "../src/cuda/score_addition_cuda.h"

// 0 = full run (CPU + GPU + compare), 1 = GPU-only test (no env var)
#define RUN_GPU_ONLY_TEST 1

using AdditionMiner = score_addition::Miner<
    score_addition::NUMBER_OF_INPUT_NEURONS,
    score_addition::NUMBER_OF_OUTPUT_NEURONS,
    score_addition::NUMBER_OF_TICKS,
    score_addition::MAX_NEIGHBOR_NEURONS,
    score_addition::POPULATION_THRESHOLD,
    score_addition::NUMBER_OF_MUTATIONS,
    score_addition::SOLUTION_THRESHOLD>;

int main()
{
    std::cout << "[verify] Setting up mining seed, public key, pool...\n" << std::flush;

    unsigned char miningSeed[32] = {};
    unsigned char publicKey[32] = {};
    for (int i = 0; i < 32; ++i)
    {
        miningSeed[i] = static_cast<unsigned char>(i);
        publicKey[i] = static_cast<unsigned char>(i + 32);
    }

    std::vector<unsigned char> pool(POOL_VEC_PADDING_SIZE);
    generateRandom2Pool(miningSeed, pool.data());

#if RUN_GPU_ONLY_TEST
    constexpr int kNumNonces = 256;
    std::cout << "[verify] GPU-only mode: generating " << kNumNonces << " nonces (no CPU, no ref file)...\n" << std::flush;

    std::vector<std::array<unsigned char, 32>> nonces(kNumNonces);
    std::mt19937_64 rng(123456789ull);
    for (int i = 0; i < kNumNonces; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            const auto v = rng();
            std::memcpy(&nonces[i].data()[j * 8], &v, sizeof(v));
        }
    }

    std::vector<unsigned char> noncesFlat(kNumNonces * 32);
    for (int i = 0; i < kNumNonces; ++i)
        std::memcpy(&noncesFlat[i * 32], nonces[i].data(), 32);

    std::cout << "[verify] Running GPU batch (" << kNumNonces << " nonces in parallel)...\n" << std::flush;
    auto tGpu0 = std::chrono::steady_clock::now();
    std::vector<unsigned int> gpuScores(kNumNonces);
    int ok = score_addition_cuda_batch(
        pool.data(), publicKey, noncesFlat.data(), gpuScores.data(), kNumNonces);
    double gpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tGpu0).count();

    if (!ok)
    {
        std::cerr << "[verify] score_addition_cuda_batch failed\n";
        return 1;
    }
    // Measured GPU test time
    std::cout << "[verify] GPU test time: " << std::fixed << std::setprecision(2)
              << gpuSec << " s (" << (gpuSec / 60.0) << " min) for " << kNumNonces << " nonces\n";
    if (gpuSec > 0)
        std::cout << "[verify]   -> " << (gpuSec / kNumNonces) << " s/nonce, "
                  << (kNumNonces / gpuSec) << " nonces/s\n" << std::flush;
    std::cout << "[verify] GPU scores (visual verification):\n" << std::flush;
    for (int i = 0; i < kNumNonces; ++i)
        std::cout << "[verify]   nonce " << (i + 1) << "/" << kNumNonces << " score=" << gpuScores[i] << "\n" << std::flush;
    std::cout << "[verify] GPU-only run complete. Visual verification: check scores above.\n";
    // system("pause"); 
    return 0;
#else
    // Full run: generate nonces, run CPU, run GPU, compare.
    constexpr int kNumNonces = 32;
    std::cout << "[verify] Generating " << kNumNonces << " nonces...\n" << std::flush;

    std::vector<std::array<unsigned char, 32>> nonces(kNumNonces);
    std::mt19937_64 rng(123456789ull);
    for (int i = 0; i < kNumNonces; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            const auto v = rng();
            std::memcpy(&nonces[i].data()[j * 8], &v, sizeof(v));
        }
    }

    std::vector<unsigned int> cpuScores(kNumNonces);
    AdditionMiner miner;
    miner.initialize(miningSeed);

    std::cout << "[verify] Computing CPU scores (" << kNumNonces << " nonces, sequential; each ~1 min)...\n" << std::flush;
    auto tCpu0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kNumNonces; ++i)
    {
        std::cout << "[verify]   CPU nonce " << (i + 1) << "/" << kNumNonces << "..." << std::flush;
         cpuScores[i] = miner.computeScore(publicKey, nonces[i].data());
        std::cout << " score=" << cpuScores[i] << "\n" << std::flush;
    }
    double cpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tCpu0).count();
    std::cout << "[verify] CPU done: " << kNumNonces << " nonces in " << (cpuSec / 60.0) << " min\n" << std::flush;

    std::vector<unsigned char> noncesFlat(kNumNonces * 32);
    for (int i = 0; i < kNumNonces; ++i)
    {
        std::memcpy(&noncesFlat[i * 32], nonces[i].data(), 32);
    }

    std::cout << "[verify] Running GPU batch (" << kNumNonces << " nonces in parallel)...\n" << std::flush;
    if (kNumNonces == 1)
        std::cout << "[verify] With 1 nonce the GPU uses a single thread and can take 5-10 minutes (same work as CPU but slower per thread). Please wait.\n" << std::flush;
    auto tGpu0 = std::chrono::steady_clock::now();
    std::vector<unsigned int> gpuScores(kNumNonces);
    int ok = score_addition_cuda_batch(
        pool.data(), publicKey, noncesFlat.data(), gpuScores.data(), kNumNonces);
    double gpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tGpu0).count();

    if (!ok)
    {
        std::cerr << "[verify] score_addition_cuda_batch failed\n";
        return 1;
    }
    std::cout << "[verify] GPU done: " << kNumNonces << " nonces in " << (gpuSec / 60.0) << " min\n" << std::flush;
    std::cout << "[verify] Comparing results...\n" << std::flush;

    for (int i = 0; i < kNumNonces; ++i)
    {
        if (cpuScores[i] != gpuScores[i])
        {
            std::cerr << "[verify] Mismatch at nonce " << i << ": CPU=" << cpuScores[i]
                      << " GPU=" << gpuScores[i] << "\n";
            return 1;
        }
    }

    std::cout << "[verify] Verification PASSED: CPU and GPU scores match for " << kNumNonces << " nonces\n";
    if (gpuSec > 0 && cpuSec > 0)
    {
        double speedup = cpuSec / gpuSec;
        std::cout << "[verify] GPU was " << std::fixed << std::setprecision(1) << speedup
                  << "x faster for this batch (CPU " << (cpuSec / 60.0)
                  << " min vs GPU " << (gpuSec / 60.0) << " min)\n";
    }
    if (kNumNonces == 1)
        std::cout << "[verify] Note: For 1 nonce, CPU is usually faster. Increase kNumNonces (e.g. 32) to show GPU advantage.\n";
    return 0;
#endif
}
