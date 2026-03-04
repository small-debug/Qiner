#pragma once

#include "score_addition.h"
#include "score_common.h"
#include "K12AndKeyUtil.h"

#include <cstddef>
#include <cstring>

namespace score_addition
{

// InitValue layout matching Miner::InitValue for GPU transfer.
// Uses same constants: K=14, L=8, 2M=728, P=122, S=100
struct InitValueGPU
{
    static constexpr unsigned long long numberOfOutputNeurons = NUMBER_OF_OUTPUT_NEURONS;
    static constexpr unsigned long long maxNumberOfSynapses =
        POPULATION_THRESHOLD * MAX_NEIGHBOR_NEURONS;
    static constexpr unsigned long long paddingNumberOfSynapses =
        (maxNumberOfSynapses + 31) / 32 * 32;
    static constexpr unsigned long long numberOfMutations = NUMBER_OF_MUTATIONS;

    unsigned long long outputNeuronPositions[numberOfOutputNeurons];
    unsigned long long synapseWeight[paddingNumberOfSynapses / 32];
    unsigned long long synpaseMutation[numberOfMutations];
};

// Precompute InitValue for each nonce. Pool must be from generateRandom2Pool(miningSeed).
inline void precomputeInitValues(
    const unsigned char* pool,
    const unsigned char* publicKey,
    const unsigned char* nonces,
    InitValueGPU* outInitValues,
    size_t numNonces)
{
    unsigned char combined[64];
    unsigned char hash[32];

    std::memcpy(combined, publicKey, 32);

    for (size_t i = 0; i < numNonces; ++i)
    {
        std::memcpy(combined + 32, nonces + i * 32, 32);
        KangarooTwelve(combined, 64, hash, 32);
        random2(hash, pool, reinterpret_cast<unsigned char*>(&outInitValues[i]), sizeof(InitValueGPU));
    }
}

} // namespace score_addition
