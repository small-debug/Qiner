#pragma once

#include <array>
#include <cstdint>

#include "score_addition.h"

namespace score_addition
{

// Each entry packs:
// - 14 ternary input values (A and B) into 14 bits (1 = +1, 0 = -1)
// - 8 ternary output values (C) into 8 bits (1 = +1, 0 = -1)
struct PackedTrainingEntry
{
    std::uint16_t inputBits;  // 14 significant bits used
    std::uint8_t outputBits;  // 8 significant bits used
};

constexpr std::size_t PACKED_TRAINING_SET_SIZE =
    static_cast<std::size_t>(1ULL << NUMBER_OF_INPUT_NEURONS);

// Build a packed training table whose semantics match generateTrainingSet() +
// toTenaryBits() in score_addition.h.
inline void buildPackedTrainingTable(
    std::array<PackedTrainingEntry, PACKED_TRAINING_SET_SIZE>& out)
{
    static constexpr long long boundValue =
        (1LL << (NUMBER_OF_INPUT_NEURONS / 2)) / 2;

    unsigned long long index = 0;
    for (long long A = -boundValue; A < boundValue; ++A)
    {
        for (long long B = -boundValue; B < boundValue; ++B)
        {
            long long C = A + B;

            char input[NUMBER_OF_INPUT_NEURONS];
            char output[NUMBER_OF_OUTPUT_NEURONS];

            // Reuse the exact helper used by the CPU miner.
            toTenaryBits<NUMBER_OF_INPUT_NEURONS / 2>(A, input);
            toTenaryBits<NUMBER_OF_INPUT_NEURONS / 2>(
                B, input + NUMBER_OF_INPUT_NEURONS / 2);
            toTenaryBits<NUMBER_OF_OUTPUT_NEURONS>(C, output);

            std::uint16_t packedInput = 0;
            for (unsigned int i = 0; i < NUMBER_OF_INPUT_NEURONS; ++i)
            {
                if (input[i] == 1)
                {
                    packedInput |= static_cast<std::uint16_t>(1u << i);
                }
            }

            std::uint8_t packedOutput = 0;
            for (unsigned int i = 0; i < NUMBER_OF_OUTPUT_NEURONS; ++i)
            {
                if (output[i] == 1)
                {
                    packedOutput |= static_cast<std::uint8_t>(1u << i);
                }
            }

            out[index].inputBits = packedInput;
            out[index].outputBits = packedOutput;
            ++index;
        }
    }
}

} // namespace score_addition

