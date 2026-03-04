#include "score_addition_cuda.h"
#include "../score_addition_initvalue.h"
#include "../addition_training_table.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>

// ---------------------------------------------------------------------------
// Constants (match score_addition.h)
// ---------------------------------------------------------------------------
namespace
{
constexpr unsigned long long K = 14;
constexpr unsigned long long L = 8;
constexpr unsigned long long N = 120;
constexpr unsigned long long M2 = 728;
constexpr unsigned long long P = 122;
constexpr unsigned long long S = 100;
constexpr unsigned long long numberOfNeurons = K + L;  // 22
constexpr unsigned long long maxNumberOfSynapses = P * M2;
constexpr unsigned long long paddingNumberOfSynapses = (maxNumberOfSynapses + 31) / 32 * 32;
constexpr unsigned long long trainingSetSize = 1ULL << K;  // 16384
}  // namespace

// ---------------------------------------------------------------------------
// Device structs (plain, no std::)
// ---------------------------------------------------------------------------
struct Neuron
{
    // Keep this struct compact: it is copied per-thread in inferANN_parallel_kernel.
    // Values are small (0..2), so 1 byte is sufficient.
    unsigned char type;   // 0=kInput, 1=kOutput, 2=kEvolution
    char value;
    bool markForRemoval;
};

struct Synapse
{
    char weight;
};

// Layout must match InitValueGPU
struct InitValueDevice
{
    unsigned long long outputNeuronPositions[L];
    unsigned long long synapseWeight[paddingNumberOfSynapses / 32];
    unsigned long long synpaseMutation[S];
};

// Packed training entry for __constant__ (no std::)
struct PackedEntry
{
    unsigned short inputBits;
    unsigned char outputBits;
};

__constant__ PackedEntry g_trainingTable[trainingSetSize];

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ char extractWeight(unsigned long long packedValue, int position)
{
    unsigned char v = (packedValue >> (position * 2)) & 0b11;
    if (v == 2) return -1;
    if (v == 3) return 1;
    return 0;
}

__device__ __forceinline__ char clampNeuron(long long v)
{
    if (v > 1) return 1;
    if (v < -1) return -1;
    return (char)v;
}

__device__ __forceinline__ unsigned long long clampNeuronIndex(
    unsigned long long population, unsigned long long neuronIdx, long long value)
{
    long long nn = (value >= 0) ? (long long)neuronIdx + value
                               : (long long)neuronIdx + (long long)population + value;
    nn = nn % (long long)population;
    if (nn < 0) nn += population;
    return (unsigned long long)nn;
}

__device__ __forceinline__ unsigned long long getActualNeighborCount(unsigned long long population)
{
    unsigned long long maxN = (population > 0) ? population - 1 : 0;
    return (maxN < M2) ? maxN : M2;
}

__device__ __forceinline__ unsigned long long getLeftNeighborCount(unsigned long long population)
{
    return (getActualNeighborCount(population) + 1) / 2;
}

__device__ __forceinline__ unsigned long long getRightNeighborCount(unsigned long long population)
{
    return getActualNeighborCount(population) - getLeftNeighborCount(population);
}

__device__ __forceinline__ unsigned long long getSynapseStartIndex(unsigned long long population)
{
    return M2 / 2 - getLeftNeighborCount(population);
}

__device__ __forceinline__ unsigned long long getSynapseEndIndex(unsigned long long population)
{
    return M2 / 2 + getRightNeighborCount(population);
}

__device__ __forceinline__ long long bufferIndexToOffset(unsigned long long bufferIdx)
{
    const long long half = (long long)(M2 / 2);
    if (bufferIdx < (unsigned long long)half)
        return (long long)bufferIdx - half;
    return (long long)bufferIdx - half + 1;
}

__device__ __forceinline__ long long offsetToBufferIndex(long long offset)
{
    const long long half = (long long)(M2 / 2);
    if (offset == 0) return -1;
    if (offset < 0) return half + offset;
    return half + offset - 1;
}

__device__ __forceinline__ long long getIndexInSynapsesBuffer(
    unsigned long long population, long long neighborOffset)
{
    long long leftC = (long long)getLeftNeighborCount(population);
    long long rightC = (long long)getRightNeighborCount(population);
    if (neighborOffset == 0 || neighborOffset < -leftC || neighborOffset > rightC)
        return -1;
    return offsetToBufferIndex(neighborOffset);
}

__device__ __forceinline__ unsigned long long getNeighborNeuronIndex(
    unsigned long long population, unsigned long long neuronIndex, unsigned long long neighborOffset)
{
    unsigned long long leftN = getLeftNeighborCount(population);
    if (neighborOffset < leftN)
        return clampNeuronIndex(population, neuronIndex + neighborOffset, -(long long)leftN);
    return clampNeuronIndex(population, neuronIndex + neighborOffset + 1, -(long long)leftN);
}

__device__ __forceinline__ Synapse* getSynapses(Synapse* synapses, unsigned long long neuronIndex)
{
    return &synapses[neuronIndex * M2];
}

// ---------------------------------------------------------------------------
// initializeANN_from_InitValue
// ---------------------------------------------------------------------------
__device__ unsigned int initializeANN_from_InitValue(
    const InitValueDevice* initValue,
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long* neuronIndices,
    unsigned long long* outputNeuronIndices,
    unsigned long long& population)
{
    population = numberOfNeurons;

    for (unsigned long long i = 0; i < population; ++i)
    {
        neuronIndices[i] = i;
        neurons[i].type = 0;  // kInput
    }

    unsigned long long neuronCount = population;
    for (unsigned long long i = 0; i < L; ++i)
    {
        unsigned long long outIdx = initValue->outputNeuronPositions[i] % neuronCount;
        neurons[neuronIndices[outIdx]].type = 1;  // kOutput
        outputNeuronIndices[i] = neuronIndices[outIdx];
        neuronCount--;
        neuronIndices[outIdx] = neuronIndices[neuronCount];
    }

    for (unsigned long long i = 0; i < paddingNumberOfSynapses / 32; ++i)
    {
        for (int j = 0; j < 32; ++j)
        {
            synapses[32 * i + j].weight = extractWeight(initValue->synapseWeight[i], j);
        }
    }

    return 0;  // score computed by inferANN
}

// ---------------------------------------------------------------------------
// loadTrainingData
// ---------------------------------------------------------------------------
__device__ void loadTrainingData(
    unsigned long long trainingIndex,
    Neuron* neurons,
    unsigned long long population,
    char* outputNeuronExpectedValue)
{
    PackedEntry e = g_trainingTable[trainingIndex];

    unsigned long long inputIdx = 0;
    for (unsigned long long n = 0; n < population; ++n)
    {
        neurons[n].value = 0;
        if (neurons[n].type == 0)  // kInput
        {
            char v = ((e.inputBits >> inputIdx) & 1) ? 1 : -1;
            neurons[n].value = v;
            inputIdx++;
        }
    }

    for (unsigned long long i = 0; i < L; ++i)
    {
        outputNeuronExpectedValue[i] = ((e.outputBits >> i) & 1) ? 1 : -1;
    }
}

// ---------------------------------------------------------------------------
// processTick (neuronValueBuffer is int: max sum per neuron <= population*M2*1, fits in int32)
// ---------------------------------------------------------------------------
// Returns flags:
// - bit0: anyChanged (1 if any non-input neuron value changed this tick)
// - bit1: allOutputNonZero (1 if all output neurons are non-zero after this tick)
__device__ unsigned int processTick(
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long population,
    int* neuronValueBuffer)
{
    for (unsigned long long i = 0; i < population; ++i)
        neuronValueBuffer[i] = 0;

    unsigned long long startIdx = getSynapseStartIndex(population);
    unsigned long long endIdx = getSynapseEndIndex(population);

    for (long long n = 0; n < (long long)population; ++n)
    {
        Synapse* kSyn = getSynapses(synapses, (unsigned long long)n);
        int nv = (int)neurons[n].value;
#pragma unroll 8
        for (unsigned long long m = startIdx; m < endIdx; ++m)
        {
            int sw = (int)kSyn[m].weight;
            long long off = bufferIndexToOffset(m);
            unsigned long long nnIdx = clampNeuronIndex(population, (unsigned long long)n, off);
            neuronValueBuffer[nnIdx] += sw * nv;
        }
    }

    bool anyChanged = false;
    bool allOutputNonZero = true;
    for (long long n = 0; n < (long long)population; ++n)
    {
        if (neurons[n].type != 0)
        {
            char oldV = neurons[n].value;
            char newV = clampNeuron((long long)neuronValueBuffer[n]);
            neurons[n].value = newV;
            if (newV != oldV)
                anyChanged = true;
            if (neurons[n].type == 1 && newV == 0)
                allOutputNonZero = false;
        }
    }
    return (anyChanged ? 1u : 0u) | (allOutputNonZero ? 2u : 0u);
}

// ---------------------------------------------------------------------------
// runTickSimulation
// ---------------------------------------------------------------------------
__device__ void runTickSimulation(
    unsigned long long trainingIndex,
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long population,
    char* outputNeuronExpectedValue,
    int* neuronValueBuffer)
{
    loadTrainingData(trainingIndex, neurons, population, outputNeuronExpectedValue);

    for (unsigned long long tick = 0; tick < N; ++tick)
    {
        unsigned int flags = processTick(neurons, synapses, population, neuronValueBuffer);
        bool anyChanged = (flags & 1u) != 0u;
        bool allOutputNonZero = (flags & 2u) != 0u;
        if (allOutputNonZero || !anyChanged)
            break;
    }
}

// (Warp-vote early exit would require one training sample per warp and shared neuron state; not applied here.)

// ---------------------------------------------------------------------------
// computeMatchingOutput
// ---------------------------------------------------------------------------
__device__ unsigned int computeMatchingOutput(
    Neuron* neurons,
    unsigned long long population,
    const char* outputNeuronExpectedValue)
{
    unsigned int R = 0;
    unsigned long long outIdx = 0;
    for (unsigned long long i = 0; i < population; ++i)
    {
        if (neurons[i].type == 1)
        {
            if (neurons[i].value == outputNeuronExpectedValue[outIdx])
                R++;
            outIdx++;
        }
    }
    return R;
}

// ---------------------------------------------------------------------------
// inferANN
// ---------------------------------------------------------------------------
__device__ unsigned int inferANN(
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long population,
    unsigned long long* outputNeuronIndices,
    char* outputNeuronExpectedValue,
    int* neuronValueBuffer)
{
    (void)outputNeuronIndices;
    unsigned int score = 0;
    for (unsigned long long i = 0; i < trainingSetSize; ++i)
    {
        runTickSimulation(i, neurons, synapses, population,
                          outputNeuronExpectedValue, neuronValueBuffer);
        score += computeMatchingOutput(neurons, population, outputNeuronExpectedValue);
    }
    return score;
}

// ---------------------------------------------------------------------------
// removeNeuron
// ---------------------------------------------------------------------------
__device__ void removeNeuron(
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long& population,
    unsigned long long neuronIdx)
{
    long long leftC = (long long)getLeftNeighborCount(population);
    long long rightC = (long long)getRightNeighborCount(population);
    unsigned long long startIdx = getSynapseStartIndex(population);
    unsigned long long endIdx = getSynapseEndIndex(population);
    const unsigned long long halfMax = M2 / 2;

    for (long long off = -leftC; off <= rightC; ++off)
    {
        if (off == 0) continue;
        unsigned long long nnIdx = clampNeuronIndex(population, neuronIdx, off);
        Synapse* pNN = getSynapses(synapses, nnIdx);
        long long synIdx = getIndexInSynapsesBuffer(population, -off);
        if (synIdx < 0) continue;

        if ((long long)synIdx >= (long long)halfMax)
        {
            for (long long k = synIdx; k < (long long)endIdx - 1; ++k)
                pNN[k] = pNN[k + 1];
            pNN[endIdx - 1].weight = 0;
        }
        else
        {
            for (long long k = synIdx; k > (long long)startIdx; --k)
                pNN[k] = pNN[k - 1];
            pNN[startIdx].weight = 0;
        }
    }

    for (unsigned long long i = neuronIdx; i < population - 1; ++i)
    {
        neurons[i] = neurons[i + 1];
        Synapse* dst = getSynapses(synapses, i);
        Synapse* src = getSynapses(synapses, i + 1);
        for (unsigned long long k = 0; k < M2; ++k)
            dst[k] = src[k];
    }
    population--;
}

// ---------------------------------------------------------------------------
// insertNeuron
// ---------------------------------------------------------------------------
__device__ void insertNeuron(
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long& population,
    unsigned long long neuronIndex,
    unsigned long long synapseIndex)
{
    unsigned long long synFullIdx = neuronIndex * M2 + synapseIndex;
    unsigned long long oldStart = getSynapseStartIndex(population);
    unsigned long long oldEnd = getSynapseEndIndex(population);
    long long oldLeft = (long long)getLeftNeighborCount(population);
    long long oldRight = (long long)getRightNeighborCount(population);
    const unsigned long long halfMax = M2 / 2;

    Neuron ins;
    ins = neurons[neuronIndex];
    ins.type = 2;  // kEvolution
    unsigned long long insertedIdx = neuronIndex + 1;
    char origWeight = synapses[synFullIdx].weight;

    for (unsigned long long i = population; i > neuronIndex; --i)
    {
        neurons[i] = neurons[i - 1];
        Synapse* dst = getSynapses(synapses, i);
        Synapse* src = getSynapses(synapses, i - 1);
        for (unsigned long long k = 0; k < M2; ++k)
            dst[k] = src[k];
    }
    neurons[insertedIdx] = ins;
    population++;

    unsigned long long newStart = getSynapseStartIndex(population);
    unsigned long long newEnd = getSynapseEndIndex(population);
    unsigned long long newActual = getActualNeighborCount(population);

    Synapse* pIns = getSynapses(synapses, insertedIdx);
    for (unsigned long long k = 0; k < M2; ++k)
        pIns[k].weight = 0;

    if (synapseIndex < halfMax)
    {
        if (synapseIndex > newStart)
            pIns[synapseIndex - 1].weight = origWeight;
    }
    else
    {
        pIns[synapseIndex].weight = origWeight;
    }

    for (long long delta = -oldLeft; delta <= oldRight; ++delta)
    {
        if (delta == 0) continue;
        unsigned long long updIdx = clampNeuronIndex(population, insertedIdx, delta);

        long long insInNb = -1;
        for (unsigned long long k = 0; k < newActual; ++k)
        {
            if (getNeighborNeuronIndex(population, updIdx, k) == insertedIdx)
            {
                insInNb = (long long)(newStart + k);
                break;
            }
        }

        Synapse* pUpd = getSynapses(synapses, updIdx);
        if (delta < 0)
        {
            for (long long k = (long long)newEnd - 1; k >= insInNb; --k)
                pUpd[k] = pUpd[k - 1];
            if (delta == -1)
                pUpd[insInNb].weight = 0;
        }
        else
        {
            for (long long k = (long long)newStart; k < insInNb; ++k)
                pUpd[k] = pUpd[k + 1];
        }
    }
}

// ---------------------------------------------------------------------------
// scanRedundantNeurons
// ---------------------------------------------------------------------------
__device__ unsigned long long scanRedundantNeurons(
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long population)
{
    unsigned long long startIdx = getSynapseStartIndex(population);
    unsigned long long endIdx = getSynapseEndIndex(population);
    long long leftC = (long long)getLeftNeighborCount(population);
    long long rightC = (long long)getRightNeighborCount(population);
    unsigned long long count = 0;

    for (unsigned long long i = 0; i < population; ++i)
    {
        neurons[i].markForRemoval = false;
        if (neurons[i].type == 2)
        {
            bool allOutZero = true;
            bool allInZero = true;

            for (unsigned long long m = startIdx; m < endIdx; ++m)
            {
                if (synapses[i * M2 + m].weight != 0)
                {
                    allOutZero = false;
                    break;
                }
            }

            for (long long off = -leftC; off <= rightC && (allOutZero || allInZero); ++off)
            {
                if (off == 0) continue;
                unsigned long long nnIdx = clampNeuronIndex(population, i, off);
                long long synIdx = getIndexInSynapsesBuffer(population, -off);
                if (synIdx < 0) continue;
                if (getSynapses(synapses, nnIdx)[synIdx].weight != 0)
                {
                    allInZero = false;
                    break;
                }
            }

            if (allOutZero || allInZero)
            {
                neurons[i].markForRemoval = true;
                count++;
            }
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// cleanANN
// ---------------------------------------------------------------------------
__device__ void cleanANN(
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long& population)
{
    unsigned long long idx = 0;
    while (idx < population)
    {
        if (neurons[idx].markForRemoval)
        {
            removeNeuron(neurons, synapses, population, idx);
        }
        else
        {
            idx++;
        }
    }
}

// ---------------------------------------------------------------------------
// mutate
// ---------------------------------------------------------------------------
__device__ void mutate(
    const InitValueDevice* initValue,
    int mutateStep,
    Neuron* neurons,
    Synapse* synapses,
    unsigned long long& population)
{
    unsigned long long actualN = getActualNeighborCount(population);
    unsigned long long totalSyn = population * actualN;
    unsigned long long mut = initValue->synpaseMutation[mutateStep];
    unsigned long long flatIdx = (mut >> 1) % totalSyn;
    unsigned long long neuronIdx = flatIdx / actualN;
    unsigned long long localSyn = flatIdx % actualN;
    unsigned long long synIdx = localSyn + getSynapseStartIndex(population);
    unsigned long long synFullIdx = neuronIdx * M2 + synIdx;

    char delta = (mut & 1) ? 1 : -1;
    char newW = synapses[synFullIdx].weight + delta;

    if (newW >= -1 && newW <= 1)
    {
        synapses[synFullIdx].weight = newW;
    }
    else
    {
        insertNeuron(neurons, synapses, population, neuronIdx, synIdx);
    }

    while (scanRedundantNeurons(neurons, synapses, population) > 0)
    {
        cleanANN(neurons, synapses, population);
    }
}

// ---------------------------------------------------------------------------
// computeScore (device, per-thread). Best state in global to reduce local memory.
// ---------------------------------------------------------------------------
__device__ unsigned int computeScoreDevice(
    size_t nonceIndex,
    const InitValueDevice* initValue,
    int verbose,
    Neuron* d_best_neurons,
    Synapse* d_best_synapses,
    unsigned long long* d_best_population)
{
    Neuron neurons[P];
    Synapse synapses[maxNumberOfSynapses];
    unsigned long long neuronIndices[numberOfNeurons];
    unsigned long long outputNeuronIndices[L];
    char outputNeuronExpectedValue[L];
    int neuronValueBuffer[P];

    unsigned long long population;
    initializeANN_from_InitValue(
        initValue, neurons, synapses, neuronIndices, outputNeuronIndices, population);

    if (verbose)
        printf("[cuda-kernel] nonce %llu: initial inferANN starting\n", (unsigned long long)nonceIndex);

    unsigned int bestR = inferANN(neurons, synapses, population,
                                  outputNeuronIndices, outputNeuronExpectedValue,
                                  neuronValueBuffer);

    Neuron* bestNeurons = d_best_neurons + nonceIndex * P;
    Synapse* bestSynapses = d_best_synapses + nonceIndex * paddingNumberOfSynapses;
    unsigned long long bestPop = population;
    for (unsigned long long i = 0; i < population; ++i)
        bestNeurons[i] = neurons[i];
    for (unsigned long long i = 0; i < population * M2; ++i)
        bestSynapses[i] = synapses[i];
    d_best_population[nonceIndex] = bestPop;

    for (int s = 0; s < (int)S; ++s)
    {
        mutate(initValue, s, neurons, synapses, population);

        if (population >= P)
            break;

        unsigned int R = inferANN(neurons, synapses, population,
                                  outputNeuronIndices, outputNeuronExpectedValue,
                                  neuronValueBuffer);

        if (R >= bestR)
        {
            bestR = R;
            bestPop = population;
            for (unsigned long long i = 0; i < population; ++i)
                bestNeurons[i] = neurons[i];
            for (unsigned long long i = 0; i < population * M2; ++i)
                bestSynapses[i] = synapses[i];
            d_best_population[nonceIndex] = bestPop;
        }
        else
        {
            population = bestPop;
            for (unsigned long long i = 0; i < population; ++i)
                neurons[i] = bestNeurons[i];
            for (unsigned long long i = 0; i < population * M2; ++i)
                synapses[i] = bestSynapses[i];
        }
        if (verbose && (s + 1) % 10 == 0)
            printf("[cuda-kernel] nonce %llu: mutation %d/%llu\n",
                   (unsigned long long)nonceIndex, s + 1, (unsigned long long)S);
    }
    return bestR;
}

// ---------------------------------------------------------------------------
// Batch kernel (best state in global memory to reduce per-thread local memory)
// ---------------------------------------------------------------------------
__global__ void score_batch_kernel(
    const InitValueDevice* initValues,
    unsigned int* scores,
    size_t numNonces,
    int verbose,
    Neuron* d_best_neurons,
    Synapse* d_best_synapses,
    unsigned long long* d_best_population)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNonces)
    {
        scores[tid] = computeScoreDevice(tid, &initValues[tid], verbose,
                                         d_best_neurons, d_best_synapses, d_best_population);
    }
}

// ---------------------------------------------------------------------------
// Parallel inferANN path: init state to global (one thread per nonce)
// ---------------------------------------------------------------------------
__global__ void init_to_global_kernel(
    const InitValueDevice* d_init,
    Neuron* d_neurons,
    Synapse* d_synapses,
    unsigned long long* d_population,
    size_t numNonces)
{
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numNonces)
        return;

    Neuron neurons[P];
    Synapse synapses[maxNumberOfSynapses];
    unsigned long long neuronIndices[numberOfNeurons];
    unsigned long long outputNeuronIndices[L];
    unsigned long long population;

    initializeANN_from_InitValue(&d_init[n], neurons, synapses, neuronIndices,
                                  outputNeuronIndices, population);

    for (unsigned long long i = 0; i < population; ++i)
        d_neurons[n * P + i] = neurons[i];
    for (unsigned long long i = 0; i < population * M2; ++i)
        d_synapses[n * paddingNumberOfSynapses + i] = synapses[i];
    d_population[n] = population;
}

// ---------------------------------------------------------------------------
// Parallel inferANN path: mutate (one thread per nonce, reads/writes global state)
// ---------------------------------------------------------------------------
__global__ void mutate_from_global_kernel(
    const InitValueDevice* d_init,
    Neuron* d_neurons,
    Synapse* d_synapses,
    unsigned long long* d_population,
    int mutateStep,
    size_t numNonces)
{
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numNonces)
        return;

    Neuron* neurons = d_neurons + n * P;
    Synapse* synapses = d_synapses + n * paddingNumberOfSynapses;
    unsigned long long population = d_population[n];

    mutate(&d_init[n], mutateStep, neurons, synapses, population);

    d_population[n] = population;
}

// ---------------------------------------------------------------------------
// Parallel inferANN path: one training sample per thread, warp reduce then atomicAdd
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void inferANN_parallel_kernel(
    const Neuron* d_neurons,
    const Synapse* d_synapses,
    const unsigned long long* d_population,
    unsigned int* d_scores,
    size_t numNonces)
{
    constexpr unsigned long long kBlocksPerNonce = trainingSetSize / 256;  // 64
    size_t blockId = (size_t)blockIdx.x;
    size_t nonceIdx = blockId / kBlocksPerNonce;
    if (nonceIdx >= numNonces)
        return;

    unsigned long long trainingIdx = (blockId % kBlocksPerNonce) * 256 + (unsigned long long)threadIdx.x;
    if (trainingIdx >= trainingSetSize)
        return;

    const Neuron* nonce_neurons = d_neurons + nonceIdx * P;
    const Synapse* my_synapses = d_synapses + nonceIdx * paddingNumberOfSynapses;
    unsigned long long population = d_population[nonceIdx];

    // Cooperative load: one coalesced read of neurons into shared, then each thread copies to private
    __shared__ Neuron sh_neurons[P];
    for (unsigned long long i = (unsigned long long)threadIdx.x; i < population; i += 256u)
        sh_neurons[i] = nonce_neurons[i];
    __syncthreads();

    Neuron neurons[P];
    for (unsigned long long i = 0; i < population; ++i)
        neurons[i] = sh_neurons[i];

    char outputNeuronExpectedValue[L];
    int neuronValueBuffer[P];

    runTickSimulation(trainingIdx, neurons, (Synapse*)my_synapses, population,
                     outputNeuronExpectedValue, neuronValueBuffer);
    unsigned int myScore = computeMatchingOutput(neurons, population, outputNeuronExpectedValue);

    // Warp-level reduce with shuffle (8 warps of 32 threads)
    unsigned int warpSum = myScore;
    for (int offset = 16; offset > 0; offset >>= 1)
        warpSum += __shfl_down_sync(0xffffffffu, warpSum, offset);
    __shared__ unsigned int warpSums[8];
    if (threadIdx.x % 32 == 0)
        warpSums[threadIdx.x / 32] = warpSum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int blockSum = warpSums[0] + warpSums[1] + warpSums[2] + warpSums[3]
                             + warpSums[4] + warpSums[5] + warpSums[6] + warpSums[7];
        atomicAdd(&d_scores[nonceIdx], blockSum);
    }
}

// ---------------------------------------------------------------------------
// Parallel inferANN path: update best or rollback (one thread per nonce)
// ---------------------------------------------------------------------------
__global__ void update_best_rollback_kernel(
    Neuron* d_neurons,
    Synapse* d_synapses,
    unsigned long long* d_population,
    unsigned int* d_scores,
    Neuron* d_best_neurons,
    Synapse* d_best_synapses,
    unsigned long long* d_best_population,
    unsigned int* d_best_scores,
    size_t numNonces)
{
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numNonces)
        return;

    unsigned long long pop = d_population[n];
    unsigned int curScore = d_scores[n];
    unsigned int bestScore = d_best_scores[n];

    if (curScore >= bestScore)
    {
        d_best_scores[n] = curScore;
        d_best_population[n] = pop;
        for (unsigned long long i = 0; i < pop; ++i)
            d_best_neurons[n * P + i] = d_neurons[n * P + i];
        for (unsigned long long i = 0; i < pop * M2; ++i)
            d_best_synapses[n * paddingNumberOfSynapses + i] = d_synapses[n * paddingNumberOfSynapses + i];
    }
    else
    {
        pop = d_best_population[n];
        d_population[n] = pop;
        for (unsigned long long i = 0; i < pop; ++i)
            d_neurons[n * P + i] = d_best_neurons[n * P + i];
        for (unsigned long long i = 0; i < pop * M2; ++i)
            d_synapses[n * paddingNumberOfSynapses + i] = d_best_synapses[n * paddingNumberOfSynapses + i];
    }
}

// ---------------------------------------------------------------------------
// Host: GPU memory reporting (optional, set SCORE_ADDITION_CUDA_MEMINFO=1)
// ---------------------------------------------------------------------------
static void logGpuMemInfo(const char* label)
{
    const char* env = std::getenv("SCORE_ADDITION_CUDA_MEMINFO");
    if (!env || env[0] == '0')
        return;
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
        return;
    size_t usedBytes = totalBytes - freeBytes;
    std::fprintf(stderr, "[cuda-mem] %s: used %.1f MB, free %.1f MB, total %.1f MB\n",
                 label,
                 usedBytes / (1024.0 * 1024.0),
                 freeBytes / (1024.0 * 1024.0),
                 totalBytes / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
// Host: upload training table (cached, upload once per process)
// ---------------------------------------------------------------------------
static bool s_trainingTableUploaded = false;

static bool ensureTrainingTableUploaded()
{
    if (s_trainingTableUploaded)
        return true;
    std::array<score_addition::PackedTrainingEntry, score_addition::PACKED_TRAINING_SET_SIZE> table;
    score_addition::buildPackedTrainingTable(table);

    PackedEntry* hBuf = (PackedEntry*)std::malloc(sizeof(PackedEntry) * trainingSetSize);
    if (!hBuf)
        return false;
    for (size_t i = 0; i < trainingSetSize; ++i)
    {
        hBuf[i].inputBits = table[i].inputBits;
        hBuf[i].outputBits = table[i].outputBits;
    }
    cudaError_t err = cudaMemcpyToSymbol(g_trainingTable, hBuf, sizeof(PackedEntry) * trainingSetSize);
    std::free(hBuf);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMemcpyToSymbol training table failed: %s\n", cudaGetErrorString(err));
        if (err == cudaErrorInsufficientDriver)
            std::fprintf(stderr, "[CUDA] Fix: upgrade the NVIDIA driver (nvidia-smi shows max CUDA version), or rebuild with an older CUDA toolkit to match the driver. CUDA 12.x needs driver >= 525; CUDA 11.x needs driver >= 450.\n");
        return false;
    }
    s_trainingTableUploaded = true;
    return true;
}

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------
__global__ void smoke_kernel(unsigned int* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = idx + 1;
}

int score_addition_cuda_smoke_test(void)
{
    const int N = 256;
    unsigned int* d_out = nullptr;
    unsigned int* h_out = nullptr;

    cudaError_t err = cudaMalloc(&d_out, N * sizeof(unsigned int));
    if (err != cudaSuccess)
    {
        const char* msg = cudaGetErrorString(err);
        std::fprintf(stderr, "cudaMalloc failed: %s (code %d). Check: nvidia-smi, driver supports CUDA 13.1, no other process holding GPU.\n",
                     msg ? msg : "unknown", (int)err);
        return 0;
    }

    h_out = (unsigned int*)std::malloc(N * sizeof(unsigned int));
    if (!h_out)
    {
        cudaFree(d_out);
        return 0;
    }

    smoke_kernel<<<(N + 255) / 256, 256>>>(d_out, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "kernel launch/sync failed: %s\n", cudaGetErrorString(err));
        std::free(h_out);
        cudaFree(d_out);
        return 0;
    }

    err = cudaMemcpy(h_out, d_out, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        std::free(h_out);
        cudaFree(d_out);
        return 0;
    }

    unsigned int expected = (unsigned int)N * (N + 1) / 2;
    unsigned int actual = 0;
    for (int i = 0; i < N; ++i)
        actual += h_out[i];

    std::free(h_out);
    cudaFree(d_out);

    if (actual != expected)
    {
        std::fprintf(stderr, "smoke test failed: expected %u, got %u\n", expected, actual);
        return 0;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Batch API
// ---------------------------------------------------------------------------
int score_addition_cuda_batch(
    const unsigned char* pool,
    const unsigned char* publicKey,
    const unsigned char* nonces,
    unsigned int* scores,
    size_t numNonces)
{
    if (!pool || !publicKey || !nonces || !scores || numNonces == 0)
        return 0;

    const char* verboseEnv = std::getenv("SCORE_ADDITION_CUDA_VERBOSE");
    int verbose = (verboseEnv && verboseEnv[0] == '1') ? 1 : (numNonces <= 8 ? 1 : 0);  // progress by default for small batches
    auto t0 = std::chrono::steady_clock::now();

    if (!ensureTrainingTableUploaded())
        return 0;
    if (verbose)
        std::fprintf(stderr, "[cuda-verbose] training table upload: %.2f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    logGpuMemInfo("after training table upload");

    auto t1 = std::chrono::steady_clock::now();
    std::vector<score_addition::InitValueGPU> initValues(numNonces);
    score_addition::precomputeInitValues(pool, publicKey, nonces, initValues.data(), numNonces);
    auto t_after_precompute = std::chrono::steady_clock::now();
    if (verbose)
        std::fprintf(stderr, "[cuda-verbose] precompute init values: %.2f s\n",
                     std::chrono::duration<double>(t_after_precompute - t1).count());

    const char* legacyEnv = std::getenv("SCORE_ADDITION_CUDA_LEGACY");
    int useLegacy = (legacyEnv && legacyEnv[0] == '1') ? 1 : 0;

    if (!useLegacy)
    {
        // Parallel inferANN path: init/mutate/inferANN kernels, host loop over steps.
        // Cache device buffers and reuse across batches (same numNonces).
        static size_t s_parallel_cached_numNonces = 0;
        static InitValueDevice* s_d_init_p = nullptr;
        static Neuron* s_d_neurons = nullptr;
        static Synapse* s_d_synapses = nullptr;
        static unsigned long long* s_d_population = nullptr;
        static unsigned int* s_d_scores_p = nullptr;
        static Neuron* s_d_best_neurons_p = nullptr;
        static Synapse* s_d_best_synapses_p = nullptr;
        static unsigned long long* s_d_best_population_p = nullptr;
        static unsigned int* s_d_best_scores = nullptr;

        auto free_parallel_cache = [&]() {
            if (s_d_best_scores) { cudaFree(s_d_best_scores); s_d_best_scores = nullptr; }
            if (s_d_best_population_p) { cudaFree(s_d_best_population_p); s_d_best_population_p = nullptr; }
            if (s_d_best_synapses_p) { cudaFree(s_d_best_synapses_p); s_d_best_synapses_p = nullptr; }
            if (s_d_best_neurons_p) { cudaFree(s_d_best_neurons_p); s_d_best_neurons_p = nullptr; }
            if (s_d_scores_p) { cudaFree(s_d_scores_p); s_d_scores_p = nullptr; }
            if (s_d_population) { cudaFree(s_d_population); s_d_population = nullptr; }
            if (s_d_synapses) { cudaFree(s_d_synapses); s_d_synapses = nullptr; }
            if (s_d_neurons) { cudaFree(s_d_neurons); s_d_neurons = nullptr; }
            if (s_d_init_p) { cudaFree(s_d_init_p); s_d_init_p = nullptr; }
            s_parallel_cached_numNonces = 0;
        };

        if (s_parallel_cached_numNonces != numNonces) {
            free_parallel_cache();
        }

        InitValueDevice* d_init_p = s_d_init_p;
        Neuron* d_neurons = s_d_neurons;
        Synapse* d_synapses = s_d_synapses;
        unsigned long long* d_population = s_d_population;
        unsigned int* d_scores_p = s_d_scores_p;
        Neuron* d_best_neurons_p = s_d_best_neurons_p;
        Synapse* d_best_synapses_p = s_d_best_synapses_p;
        unsigned long long* d_best_population_p = s_d_best_population_p;
        unsigned int* d_best_scores = s_d_best_scores;

        auto t2p = std::chrono::steady_clock::now();
        if (d_init_p == nullptr) {
            cudaError_t err = cudaMalloc(&s_d_init_p, numNonces * sizeof(InitValueDevice));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_init (parallel) failed: %s\n", cudaGetErrorString(err)); return 0; }
            d_init_p = s_d_init_p;
            err = cudaMalloc(&s_d_neurons, numNonces * P * sizeof(Neuron));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_neurons failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_neurons = s_d_neurons;
            err = cudaMalloc(&s_d_synapses, numNonces * paddingNumberOfSynapses * sizeof(Synapse));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_synapses failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_synapses = s_d_synapses;
            err = cudaMalloc(&s_d_population, numNonces * sizeof(unsigned long long));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_population failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_population = s_d_population;
            err = cudaMalloc(&s_d_scores_p, numNonces * sizeof(unsigned int));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_scores (parallel) failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_scores_p = s_d_scores_p;
            err = cudaMalloc(&s_d_best_neurons_p, numNonces * P * sizeof(Neuron));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_best_neurons (parallel) failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_best_neurons_p = s_d_best_neurons_p;
            err = cudaMalloc(&s_d_best_synapses_p, numNonces * paddingNumberOfSynapses * sizeof(Synapse));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_best_synapses (parallel) failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_best_synapses_p = s_d_best_synapses_p;
            err = cudaMalloc(&s_d_best_population_p, numNonces * sizeof(unsigned long long));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_best_population (parallel) failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_best_population_p = s_d_best_population_p;
            err = cudaMalloc(&s_d_best_scores, numNonces * sizeof(unsigned int));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMalloc d_best_scores failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
            d_best_scores = s_d_best_scores;
            s_parallel_cached_numNonces = numNonces;
        }

        cudaError_t err = cudaMemcpy(d_init_p, initValues.data(), numNonces * sizeof(InitValueDevice), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::fprintf(stderr, "cudaMemcpy initValues (parallel) failed: %s\n", cudaGetErrorString(err)); free_parallel_cache(); return 0; }
        auto t_after_h2d = std::chrono::steady_clock::now();

        if (verbose)
            std::fprintf(stderr, "[cuda-verbose] parallel path alloc + H2D: %.2f s\n",
                         std::chrono::duration<double>(t_after_h2d - t2p).count());
        {
            static bool s_logged_once = false;
            if (!s_logged_once) {
                std::fprintf(stderr, "[GPU] Parallel inferANN path (%zu nonces, 101 steps)\n", (size_t)numNonces);
                std::fflush(stderr);
                s_logged_once = true;
            }
        }

        // Progress: poll from host every 1s, print every 30s.
        auto t_parallel_start = std::chrono::steady_clock::now();
        cudaEvent_t e_batch_done = nullptr;
        if (numNonces > 8u) {
            err = cudaEventCreate(&e_batch_done);
            if (err != cudaSuccess) e_batch_done = nullptr;
            if (e_batch_done) {
                std::fprintf(stderr, "[GPU] Kernel start running.\n");
                std::fflush(stderr);
            }
        }

        const char* timingEnv = std::getenv("SCORE_ADDITION_CUDA_TIMING");
        int doTiming = (timingEnv && timingEnv[0] == '1') ? 1 : 0;
        cudaEvent_t e_start = nullptr, e_after_init = nullptr, e_after_infer0 = nullptr, e_after_update0 = nullptr, e_after_loop = nullptr, e_after_d2h = nullptr;
        if (doTiming) {
            cudaEventCreate(&e_start);
            cudaEventCreate(&e_after_init);
            cudaEventCreate(&e_after_infer0);
            cudaEventCreate(&e_after_update0);
            cudaEventCreate(&e_after_loop);
            cudaEventCreate(&e_after_d2h);
            cudaEventRecord(e_start, 0);
        }

        init_to_global_kernel<<<(int)numNonces, 1>>>(d_init_p, d_neurons, d_synapses, d_population, numNonces);
        err = cudaGetLastError();
        if (err != cudaSuccess) { std::fprintf(stderr, "init_to_global_kernel failed: %s\n", cudaGetErrorString(err)); goto parallel_free; }
        if (doTiming) cudaEventRecord(e_after_init, 0);

        err = cudaMemset(d_best_scores, 0, numNonces * sizeof(unsigned int));
        if (err != cudaSuccess) { std::fprintf(stderr, "cudaMemset d_best_scores failed: %s\n", cudaGetErrorString(err)); goto parallel_free; }

        err = cudaMemset(d_scores_p, 0, numNonces * sizeof(unsigned int));
        if (err != cudaSuccess) { std::fprintf(stderr, "cudaMemset d_scores failed: %s\n", cudaGetErrorString(err)); goto parallel_free; }
        inferANN_parallel_kernel<<<(int)((trainingSetSize / 256) * numNonces), 256>>>(d_neurons, d_synapses, d_population, d_scores_p, numNonces);
        err = cudaGetLastError();
        if (err != cudaSuccess) { std::fprintf(stderr, "inferANN_parallel_kernel (initial) failed: %s\n", cudaGetErrorString(err)); goto parallel_free; }
        if (doTiming) cudaEventRecord(e_after_infer0, 0);
        update_best_rollback_kernel<<<(int)numNonces, 1>>>(d_neurons, d_synapses, d_population, d_scores_p,
            d_best_neurons_p, d_best_synapses_p, d_best_population_p, d_best_scores, numNonces);
        err = cudaGetLastError();
        if (err != cudaSuccess) { std::fprintf(stderr, "update_best_rollback_kernel failed: %s\n", cudaGetErrorString(err)); goto parallel_free; }
        if (doTiming) cudaEventRecord(e_after_update0, 0);

        for (int step = 0; step < (int)S; ++step)
        {
            mutate_from_global_kernel<<<(int)numNonces, 1>>>(d_init_p, d_neurons, d_synapses, d_population, step, numNonces);
            err = cudaGetLastError();
            if (err != cudaSuccess) { std::fprintf(stderr, "mutate_from_global_kernel step %d failed: %s\n", step, cudaGetErrorString(err)); goto parallel_free; }
            err = cudaMemset(d_scores_p, 0, numNonces * sizeof(unsigned int));
            if (err != cudaSuccess) { std::fprintf(stderr, "cudaMemset d_scores failed: %s\n", cudaGetErrorString(err)); goto parallel_free; }
            inferANN_parallel_kernel<<<(int)((trainingSetSize / 256) * numNonces), 256>>>(d_neurons, d_synapses, d_population, d_scores_p, numNonces);
            err = cudaGetLastError();
            if (err != cudaSuccess) { std::fprintf(stderr, "inferANN_parallel_kernel step %d failed: %s\n", step, cudaGetErrorString(err)); goto parallel_free; }
            update_best_rollback_kernel<<<(int)numNonces, 1>>>(d_neurons, d_synapses, d_population, d_scores_p,
                d_best_neurons_p, d_best_synapses_p, d_best_population_p, d_best_scores, numNonces);
            err = cudaGetLastError();
            if (err != cudaSuccess) { std::fprintf(stderr, "update_best_rollback_kernel step %d failed: %s\n", step, cudaGetErrorString(err)); goto parallel_free; }
        }
        if (doTiming) cudaEventRecord(e_after_loop, 0);
        if (e_batch_done) cudaEventRecord(e_batch_done, 0);

        // Poll until step loop is done; print progress every 30s so the process does not appear frozen.
        {
            auto t_last_progress = t_parallel_start;
            while (e_batch_done && cudaEventQuery(e_batch_done) == cudaErrorNotReady) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - t_last_progress).count() >= 30.0) {
                    double elapsedMin = std::chrono::duration<double>(now - t_parallel_start).count() / 60.0;
                    std::fprintf(stderr, "[GPU] Still computing... (%.1f min elapsed)\n", elapsedMin);
                    std::fflush(stderr);
                    t_last_progress = now;
                }
            }
            if (e_batch_done) {
                cudaEventDestroy(e_batch_done);
                e_batch_done = nullptr;
            }
        }

        err = cudaMemcpy(scores, d_best_scores, numNonces * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { std::fprintf(stderr, "cudaMemcpy scores (parallel) failed: %s\n", cudaGetErrorString(err)); goto parallel_free; }
        if (doTiming) cudaEventRecord(e_after_d2h, 0);

        if (doTiming) {
            err = cudaEventSynchronize(e_after_d2h);
            if (err == cudaSuccess) {
                float ms_init = 0.f, ms_infer0 = 0.f, ms_update0 = 0.f, ms_loop = 0.f, ms_d2h = 0.f;
                cudaEventElapsedTime(&ms_init, e_start, e_after_init);
                cudaEventElapsedTime(&ms_infer0, e_after_init, e_after_infer0);
                cudaEventElapsedTime(&ms_update0, e_after_infer0, e_after_update0);
                cudaEventElapsedTime(&ms_loop, e_after_update0, e_after_loop);
                cudaEventElapsedTime(&ms_d2h, e_after_loop, e_after_d2h);
                double precompute_sec = std::chrono::duration<double>(t_after_precompute - t1).count();
                double alloc_h2d_sec = std::chrono::duration<double>(t_after_h2d - t2p).count();
                double gpu_total_sec = (ms_init + ms_infer0 + ms_update0 + ms_loop + ms_d2h) * 0.001;
                double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "[GPU timing] --- Most time-consuming / optimizable areas ---\n");
                std::fprintf(stderr, "[GPU timing] precomputeInitValues (CPU):     %7.2f s  (%5.1f%%)\n", precompute_sec, 100.0 * precompute_sec / total_sec);
                std::fprintf(stderr, "[GPU timing] alloc + H2D initValues:         %7.2f s  (%5.1f%%)\n", alloc_h2d_sec, 100.0 * alloc_h2d_sec / total_sec);
                std::fprintf(stderr, "[GPU timing] init_to_global_kernel:          %7.2f s  (%5.1f%%)\n", ms_init * 0.001, 100.0 * (ms_init * 0.001) / total_sec);
                std::fprintf(stderr, "[GPU timing] initial inferANN_parallel:      %7.2f s  (%5.1f%%)  <-- optimizable\n", ms_infer0 * 0.001, 100.0 * (ms_infer0 * 0.001) / total_sec);
                std::fprintf(stderr, "[GPU timing] initial update_best_rollback:   %7.2f s  (%5.1f%%)\n", ms_update0 * 0.001, 100.0 * (ms_update0 * 0.001) / total_sec);
                std::fprintf(stderr, "[GPU timing] step loop (100x mutate+infer+up): %6.2f s  (%5.1f%%)  <-- optimizable (mostly inferANN)\n", ms_loop * 0.001, 100.0 * (ms_loop * 0.001) / total_sec);
                std::fprintf(stderr, "[GPU timing] D2H scores:                     %7.2f s  (%5.1f%%)\n", ms_d2h * 0.001, 100.0 * (ms_d2h * 0.001) / total_sec);
                std::fprintf(stderr, "[GPU timing] GPU kernel total:                %7.2f s\n", gpu_total_sec);
                std::fprintf(stderr, "[GPU timing] wall-clock total:                %7.2f s\n", total_sec);
                std::fflush(stderr);
            }
            cudaEventDestroy(e_start);
            cudaEventDestroy(e_after_init);
            cudaEventDestroy(e_after_infer0);
            cudaEventDestroy(e_after_update0);
            cudaEventDestroy(e_after_loop);
            cudaEventDestroy(e_after_d2h);
        }

        if (numNonces > 8u) {
            double totalMin = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_parallel_start).count() / 60.0;
            std::fprintf(stderr, "[GPU] Kernel finished (%.1f min)\n", totalMin);
            std::fflush(stderr);
        }
        if (verbose)
            std::fprintf(stderr, "[cuda-verbose] parallel path total: %.2f s\n",
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return 1;
parallel_free:
        if (e_batch_done) { cudaEventDestroy(e_batch_done); e_batch_done = nullptr; }
        free_parallel_cache();
        return 0;
    }

    // Legacy path (one thread per nonce, best state in global, block size 64)
    InitValueDevice* d_init = nullptr;
    unsigned int* d_scores = nullptr;
    Neuron* d_best_neurons = nullptr;
    Synapse* d_best_synapses = nullptr;
    unsigned long long* d_best_population = nullptr;

    auto t2 = std::chrono::steady_clock::now();
    cudaError_t err = cudaMalloc(&d_init, numNonces * sizeof(InitValueDevice));
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc initValues failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaMalloc(&d_scores, numNonces * sizeof(unsigned int));
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc scores failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_init);
        return 0;
    }

    err = cudaMalloc(&d_best_neurons, numNonces * P * sizeof(Neuron));
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc best_neurons failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }
    err = cudaMalloc(&d_best_synapses, numNonces * paddingNumberOfSynapses * sizeof(Synapse));
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc best_synapses failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_best_neurons);
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }
    err = cudaMalloc(&d_best_population, numNonces * sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc best_population failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_best_synapses);
        cudaFree(d_best_neurons);
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }

    err = cudaMemcpy(d_init, initValues.data(), numNonces * sizeof(InitValueDevice), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMemcpy initValues failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_best_population);
        cudaFree(d_best_synapses);
        cudaFree(d_best_neurons);
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }
    if (verbose)
        std::fprintf(stderr, "[cuda-verbose] alloc + H2D copy: %.2f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count());
    logGpuMemInfo("after batch alloc and copy");

    // Best state is in global memory, so per-thread local memory is ~88 KB (one synapse array).
    // We can use 64 threads/block for better occupancy while staying under per-block limits.
    const int kMaxThreadsPerBlock = 64;
    int blockSize = (numNonces <= (size_t)kMaxThreadsPerBlock)
                        ? (int)numNonces
                        : kMaxThreadsPerBlock;
    int numBlocks = (int)((numNonces + blockSize - 1) / blockSize);
    if (verbose)
        std::fprintf(stderr, "[cuda-verbose] kernel launch (%zu nonces, %d blocks x %d threads)...\n",
                     (size_t)numNonces, numBlocks, blockSize);
    std::fprintf(stderr, "[GPU] Kernel running (may take several minutes for 1 nonce). Progress every 30 s.\n");
    std::fflush(stderr);

    cudaStream_t stream = nullptr;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_best_population);
        cudaFree(d_best_synapses);
        cudaFree(d_best_neurons);
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }
    cudaEvent_t kernelDone = nullptr;
    err = cudaEventCreate(&kernelDone);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(d_best_population);
        cudaFree(d_best_synapses);
        cudaFree(d_best_neurons);
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }

    auto t3 = std::chrono::steady_clock::now();
    score_batch_kernel<<<numBlocks, blockSize, 0, stream>>>(d_init, d_scores, numNonces, verbose,
                                                            d_best_neurons, d_best_synapses, d_best_population);
    cudaEventRecord(kernelDone, stream);

    // Periodic "still running" so the process does not appear frozen
    for (;;)
    {
        err = cudaEventQuery(kernelDone);
        if (err == cudaSuccess)
            break;
        if (err != cudaErrorNotReady)
        {
            std::fprintf(stderr, "cudaEventQuery failed: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(kernelDone);
            cudaStreamDestroy(stream);
            cudaFree(d_best_population);
            cudaFree(d_best_synapses);
            cudaFree(d_best_neurons);
            cudaFree(d_scores);
            cudaFree(d_init);
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::seconds(30));
        double elapsedMin = std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count() / 60.0;
        std::fprintf(stderr, "[GPU] Still computing... (%.1f min elapsed)\n", elapsedMin);
        std::fflush(stderr);
    }

    err = cudaEventSynchronize(kernelDone);
    cudaEventDestroy(kernelDone);
    cudaStreamDestroy(stream);
    double kernelSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count();
    std::fprintf(stderr, "[GPU] Kernel finished (%.1f min)\n", kernelSec / 60.0);
    std::fflush(stderr);
    if (verbose)
        std::fprintf(stderr, "[cuda-verbose] kernel done: %.2f s\n", kernelSec);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "batch kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_best_population);
        cudaFree(d_best_synapses);
        cudaFree(d_best_neurons);
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }

    auto t4 = std::chrono::steady_clock::now();
    err = cudaMemcpy(scores, d_scores, numNonces * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMemcpy scores failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_best_population);
        cudaFree(d_best_synapses);
        cudaFree(d_best_neurons);
        cudaFree(d_scores);
        cudaFree(d_init);
        return 0;
    }
    cudaFree(d_best_population);
    cudaFree(d_best_synapses);
    cudaFree(d_best_neurons);
    cudaFree(d_scores);
    cudaFree(d_init);
    if (verbose)
        std::fprintf(stderr, "[cuda-verbose] D2H copy + free: %.2f s, total: %.2f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t4).count(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 1;
}
