#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <array>
#include <queue>
#include <atomic>
#include <assert.h>
#ifdef _MSC_VER
#include <intrin.h>
#include <winsock2.h>
#pragma comment (lib, "ws2_32.lib")

#else
#include <signal.h>
#ifndef PORTABLE
#include <immintrin.h>
#endif
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#endif

#include <cstdint>
#include <random>
#if defined(PORTABLE) && defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef PORTABLE
static void portable_rand_bytes(void* buf, size_t len)
{
#if defined(__linux__)
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0)
    {
        size_t n = 0;
        while (n < len)
        {
            ssize_t r = read(fd, (char*)buf + n, len - n);
            if (r <= 0) break;
            n += (size_t)r;
        }
        close(fd);
        if (n == len) return;
    }
#endif
    std::random_device rd;
    for (size_t i = 0; i < len; ++i)
        ((unsigned char*)buf)[i] = (unsigned char)(rd() & 0xFF);
}
static inline int portable_rdrand32_step(unsigned int* p)
{
    portable_rand_bytes(p, 4);
    return 1;
}
static inline int portable_rdrand64_step(unsigned long long* p)
{
    portable_rand_bytes(p, 8);
    return 1;
}
#endif

#include "score_hyperidentity.h"
#include "score_addition.h"
#include "score_common.h"
#include "keyUtils.h"

#ifdef BUILD_CUDA_ADDITION
#include "cuda/score_addition_cuda.h"
#endif

#include <chrono>
#include <iostream>

struct RequestResponseHeader
{
private:
    unsigned char _size[3];
    unsigned char _type;
    unsigned int _dejavu;

public:
    inline unsigned int size() const
    {
        return (unsigned int)_size[0] | ((unsigned int)_size[1] << 8) | ((unsigned int)_size[2] << 16);
    }

    inline void setSize(unsigned int size)
    {
        _size[0] = (unsigned char)size;
        _size[1] = (unsigned char)(size >> 8);
        _size[2] = (unsigned char)(size >> 16);
    }

    inline bool isDejavuZero() const
    {
        return !_dejavu;
    }

    inline void zeroDejavu()
    {
        _dejavu = 0;
    }


    inline unsigned int dejavu() const
    {
        return _dejavu;
    }

    inline void setDejavu(unsigned int dejavu)
    {
        _dejavu = dejavu;
    }

    inline void randomizeDejavu()
    {
#ifdef PORTABLE
        portable_rdrand32_step(&_dejavu);
#else
        _rdrand32_step(&_dejavu);
#endif
        if (!_dejavu)
        {
            _dejavu = 1;
        }
    }

    inline unsigned char type() const
    {
        return _type;
    }

    inline void setType(const unsigned char type)
    {
        _type = type;
    }
};

#define BROADCAST_MESSAGE 1

typedef struct
{
    unsigned char sourcePublicKey[32];
    unsigned char destinationPublicKey[32];
    unsigned char gammingNonce[32];
} Message;

char* nodeIp = NULL;
int nodePort = 0;

static std::atomic<char> state(0);

static unsigned char computorPublicKey[32];
static unsigned char randomSeed[32];
static std::atomic<long long> numberOfMiningIterations(0);
static std::atomic<unsigned int> numberOfFoundSolutions(0);
static std::queue<std::array<unsigned char, 32>> foundNonce;
std::mutex foundNonceLock;

#ifdef _MSC_VER
static BOOL WINAPI ctrlCHandlerRoutine(DWORD dwCtrlType)
{
    if (!state)
    {
        state = 1;
    }
    else // User force exit quickly
    {
        std::exit(1);
    }
    return TRUE;
}
#else
void ctrlCHandlerRoutine(int signum)
{
    if (!state)
    {
        state = 1;
    }
    else // User force exit quickly
    {
        std::exit(1);
    }
}
#endif

void consoleCtrlHandler()
{
#ifdef _MSC_VER
    SetConsoleCtrlHandler(ctrlCHandlerRoutine, TRUE);
#else
    signal(SIGINT, ctrlCHandlerRoutine);
#endif
}

struct Stat
{
    std::atomic<unsigned long long> totalAdditionNonce;
    std::atomic<unsigned long long> totalHyperIdentityNonce;
    std::atomic<unsigned long long> totalHyperIdentitySols;
    std::atomic<unsigned long long> totalAdditionSols;

    Stat()
    {
        totalAdditionNonce.store(0);
        totalHyperIdentityNonce.store(0);
        totalHyperIdentitySols.store(0);
        totalAdditionSols.store(0);
    }

} qinerStat;

using AdditionMiner = score_addition::Miner<
    score_addition::NUMBER_OF_INPUT_NEURONS,
    score_addition::NUMBER_OF_OUTPUT_NEURONS,
    score_addition::NUMBER_OF_TICKS,
    score_addition::MAX_NEIGHBOR_NEURONS,
    score_addition::POPULATION_THRESHOLD,
    score_addition::NUMBER_OF_MUTATIONS,
    score_addition::SOLUTION_THRESHOLD>;
using HyperIdentityMiner = score_hyberidentity::Miner<
    score_hyberidentity::NUMBER_OF_INPUT_NEURONS,
    score_hyberidentity::NUMBER_OF_OUTPUT_NEURONS,
    score_hyberidentity::NUMBER_OF_TICKS,
    score_hyberidentity::MAX_NEIGHBOR_NEURONS,
    score_hyberidentity::POPULATION_THRESHOLD,
    score_hyberidentity::NUMBER_OF_MUTATIONS,
    score_hyberidentity::SOLUTION_THRESHOLD>;

int miningThreadProc()
{
    std::unique_ptr<AdditionMiner> additionMiner(new AdditionMiner());
    additionMiner->initialize(randomSeed);

    std::unique_ptr<HyperIdentityMiner> hyperIdentityMiner(new HyperIdentityMiner());
    hyperIdentityMiner->initialize(randomSeed);

    std::array<unsigned char, 32> nonce;
    while (!state)
    {
#ifdef PORTABLE
        portable_rdrand64_step((unsigned long long*)&nonce.data()[0]);
        portable_rdrand64_step((unsigned long long*)&nonce.data()[8]);
        portable_rdrand64_step((unsigned long long*)&nonce.data()[16]);
        portable_rdrand64_step((unsigned long long*)&nonce.data()[24]);
#else
        _rdrand64_step((unsigned long long*)&nonce.data()[0]);
        _rdrand64_step((unsigned long long*)&nonce.data()[8]);
        _rdrand64_step((unsigned long long*)&nonce.data()[16]);
        _rdrand64_step((unsigned long long*)&nonce.data()[24]);
#endif

        bool solutionFound = false;

        // First byte of nonce is used for determine type of score
        if ((nonce[0] & 1) == 0)
        {
            solutionFound = hyperIdentityMiner->findSolution(computorPublicKey, nonce.data());
            // Stats
            qinerStat.totalHyperIdentityNonce.fetch_add(1);
            if (solutionFound)
            {
                qinerStat.totalHyperIdentitySols.fetch_add(1);
            }
        }
        else
        {
            solutionFound = additionMiner->findSolution(computorPublicKey, nonce.data());
            // Stats
            qinerStat.totalAdditionNonce.fetch_add(1);
            if (solutionFound)
            {
                qinerStat.totalAdditionSols.fetch_add(1);
            }
        }

        if (solutionFound)
        {
            {
                std::lock_guard<std::mutex> guard(foundNonceLock);
                foundNonce.push(nonce);
            }
            numberOfFoundSolutions++;
        }

        numberOfMiningIterations++;
    }
    return 0;
}

#ifdef BUILD_CUDA_ADDITION
static constexpr size_t GPU_ADDITION_BATCH_SIZE_DEFAULT = 256;
static constexpr size_t GPU_ADDITION_BATCH_SIZE_MIN = 64;
static constexpr size_t GPU_ADDITION_BATCH_SIZE_MAX = 4096;
static size_t g_gpuAdditionBatchSize = GPU_ADDITION_BATCH_SIZE_DEFAULT;

int gpuAdditionMiningThreadProc()
{
    size_t batchSize = g_gpuAdditionBatchSize;
    std::vector<unsigned char> pool(POOL_VEC_PADDING_SIZE);
    generateRandom2Pool(randomSeed, pool.data());

    std::vector<std::array<unsigned char, 32>> nonceBatch;
    nonceBatch.reserve(batchSize);
    std::vector<unsigned char> noncesFlat(batchSize * 32);
    std::vector<unsigned int> scores(batchSize);

    while (!state)
    {
        nonceBatch.clear();
        while (nonceBatch.size() < batchSize && !state)
        {
            std::array<unsigned char, 32> nonce;
#ifdef PORTABLE
            portable_rdrand64_step((unsigned long long*)&nonce[0]);
            portable_rdrand64_step((unsigned long long*)&nonce[8]);
            portable_rdrand64_step((unsigned long long*)&nonce[16]);
            portable_rdrand64_step((unsigned long long*)&nonce[24]);
#else
            _rdrand64_step((unsigned long long*)&nonce[0]);
            _rdrand64_step((unsigned long long*)&nonce[8]);
            _rdrand64_step((unsigned long long*)&nonce[16]);
            _rdrand64_step((unsigned long long*)&nonce[24]);
#endif
            if ((nonce[0] & 1) != 1)
                continue;
            nonceBatch.push_back(nonce);
        }

        if (nonceBatch.empty())
            continue;

        for (size_t i = 0; i < nonceBatch.size(); ++i)
            std::memcpy(&noncesFlat[i * 32], nonceBatch[i].data(), 32);

        int ok = score_addition_cuda_batch(
            pool.data(), computorPublicKey, noncesFlat.data(),
            scores.data(), nonceBatch.size());

        if (!ok)
            continue;

        qinerStat.totalAdditionNonce.fetch_add(nonceBatch.size());

        for (size_t i = 0; i < nonceBatch.size(); ++i)
        {
            numberOfMiningIterations++;
            if (scores[i] >= score_addition::SOLUTION_THRESHOLD)
            {
                qinerStat.totalAdditionSols.fetch_add(1);
                {
                    std::lock_guard<std::mutex> guard(foundNonceLock);
                    foundNonce.push(nonceBatch[i]);
                }
                numberOfFoundSolutions++;
            }
        }
    }
    return 0;
}
#endif

struct ServerSocket
{
#ifdef _MSC_VER
    ServerSocket()
    {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
    }

    ~ServerSocket()
    {
        WSACleanup();
    }

    void closeConnection()
    {
        closesocket(serverSocket);
    }

    bool establishConnection(char* address)
    {
        serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (serverSocket == INVALID_SOCKET)
        {
            printf("Fail to create a socket (%d)!\n", WSAGetLastError());
            return false;
        }

        sockaddr_in addr;
        ZeroMemory(&addr, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(nodePort);
        sscanf_s(address, "%hhu.%hhu.%hhu.%hhu", &addr.sin_addr.S_un.S_un_b.s_b1, &addr.sin_addr.S_un.S_un_b.s_b2, &addr.sin_addr.S_un.S_un_b.s_b3, &addr.sin_addr.S_un.S_un_b.s_b4);
        if (connect(serverSocket, (const sockaddr*)&addr, sizeof(addr)))
        {
            printf("Fail to connect to %d.%d.%d.%d (%d)!\n", addr.sin_addr.S_un.S_un_b.s_b1, addr.sin_addr.S_un.S_un_b.s_b2, addr.sin_addr.S_un.S_un_b.s_b3, addr.sin_addr.S_un.S_un_b.s_b4, WSAGetLastError());
            closeConnection();
            return false;
        }

        return true;
    }

    SOCKET serverSocket;
#else
    void closeConnection()
    {
        close(serverSocket);
    }
    bool establishConnection(char* address)
    {
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1)
        {
            printf("Fail to create a socket (%d)!\n", errno);
            return false;
        }

        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(nodePort);
        if (inet_pton(AF_INET, address, &addr.sin_addr) <= 0)
        {
            printf("Invalid address/ Address not supported (%s)\n", address);
            return false;
        }

        if (connect(serverSocket, (struct sockaddr*)&addr, sizeof(addr)) < 0)
        {
            printf("Fail to connect to %s (%d)\n", address, errno);
            closeConnection();
            return false;
        }

        return true;
    }

    int serverSocket;
#endif

    bool sendData(char* buffer, unsigned int size)
    {
        while (size)
        {
            int numberOfBytes;
            if ((numberOfBytes = send(serverSocket, buffer, size, 0)) <= 0)
            {
                return false;
            }
            buffer += numberOfBytes;
            size -= numberOfBytes;
        }

        return true;
    }
    bool receiveData(char* buffer, unsigned int size)
    {
        const auto beginningTime = std::chrono::steady_clock::now();
        unsigned long long deltaTime = 0;
        while (size && deltaTime <= 2000)
        {
            int numberOfBytes;
            if ((numberOfBytes = recv(serverSocket, buffer, size, 0)) <= 0)
            {
                return false;
            }
            buffer += numberOfBytes;
            size -= numberOfBytes;
            deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - beginningTime).count();
        }

        return true;
    }
};

static void hexToByte(const char* hex, uint8_t* byte, const int sizeInByte)
{
    for (int i = 0; i < sizeInByte; i++){
        sscanf(hex+i*2, "%2hhx", &byte[i]);
    }
}

void benchmark_addition_computeScore()
{
    // Fixed mining seed and public key for the benchmark
    unsigned char miningSeed[32] = { 0 };
    unsigned char publicKey[32] = { 0 };

    AdditionMiner miner;
    miner.initialize(miningSeed);

    unsigned char nonce[32];

    constexpr int kRuns = 10;
    using clock = std::chrono::high_resolution_clock;

    auto t0 = clock::now();
    for (int i = 0; i < kRuns; ++i)
    {
        // Simple varying nonce per run
        for (int j = 0; j < 32; ++j)
        {
            nonce[j] = static_cast<unsigned char>(i * 31 + j);
        }

        // Prevent the compiler from optimizing the call away
        volatile unsigned int score = miner.computeScore(publicKey, nonce);
        (void)score;
    }
    auto t1 = clock::now();

    auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double avg_us = static_cast<double>(total_us) / kRuns;

    std::cout << "Total time for " << kRuns
        << " computeScore() calls: " << total_us << " us\n";
    std::cout << "Average time per nonce: " << avg_us << " us\n";
}

int main(int argc, char* argv[])
{
    std::vector<std::thread> miningThreads;
    bool showHelp = (argc < 7 || argc > 8);
    if (!showHelp && argc >= 2)
    {
        if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)
            showHelp = true;
    }
    if (showHelp)
    {
        printf("Usage:   Qiner [Node IP] [Node Port] [MiningID] [Signing Seed] [Mining Seed] [Number of threads] [GPU batch size (optional)]\n");
        return 0;
    }
    else
    {
        nodeIp = argv[1];
        nodePort = std::atoi(argv[2]);
        char* miningID = argv[3];
        printf("Qiner is launched. Connecting to %s:%d\n", nodeIp, nodePort);

        consoleCtrlHandler();

        {
            getPublicKeyFromIdentity(miningID, computorPublicKey);

            // Data for signing the solution
            char* signingSeed = argv[4];
            unsigned char signingPrivateKey[32];
            unsigned char signingSubseed[32];
            unsigned char signingPublicKey[32];
            char privateKeyQubicFormat[128] = {0};
            char publicKeyQubicFormat[128] = {0};
            char publicIdentity[128] = {0};
            getSubseedFromSeed((unsigned char*)signingSeed, signingSubseed);
            getPrivateKeyFromSubSeed(signingSubseed, signingPrivateKey);
            getPublicKeyFromPrivateKey(signingPrivateKey, signingPublicKey);

            //getIdentityFromPublicKey(signingPublicKey, miningID, false);

            hexToByte(argv[5], randomSeed, 32);
            unsigned int numberOfThreads = atoi(argv[6]);
#ifdef BUILD_CUDA_ADDITION
            if (argc >= 8)
            {
                long batchArg = std::atoi(argv[7]);
                if (batchArg < (long)GPU_ADDITION_BATCH_SIZE_MIN)
                    g_gpuAdditionBatchSize = GPU_ADDITION_BATCH_SIZE_MIN;
                else if (batchArg > (long)GPU_ADDITION_BATCH_SIZE_MAX)
                    g_gpuAdditionBatchSize = GPU_ADDITION_BATCH_SIZE_MAX;
                else
                    g_gpuAdditionBatchSize = (size_t)batchArg;
            }
            else
            {
                g_gpuAdditionBatchSize = GPU_ADDITION_BATCH_SIZE_DEFAULT;
            }
            printf("%d threads are used. GPU batch size: %zu\n", numberOfThreads, g_gpuAdditionBatchSize);
#else
            printf("%d threads are used.\n", numberOfThreads);
#endif
            miningThreads.reserve(numberOfThreads);
#ifdef BUILD_CUDA_ADDITION
            miningThreads.emplace_back(gpuAdditionMiningThreadProc);
#endif
            for (unsigned int i = numberOfThreads; i-- > 0; )
            {
                miningThreads.emplace_back(miningThreadProc);
            }
            ServerSocket serverSocket;

            auto timestamp = std::chrono::steady_clock::now();
            long long prevNumberOfMiningIterations = 0;
            while (!state)
            {
                bool haveNonceToSend = false;
                size_t itemToSend = 0;
                std::array<unsigned char, 32> sendNonce;
                {
                    std::lock_guard<std::mutex> guard(foundNonceLock);
                    haveNonceToSend = foundNonce.size() > 0;
                    if (haveNonceToSend)
                    {
                        sendNonce = foundNonce.front();
                    }
                    itemToSend = foundNonce.size();
                }
                if (haveNonceToSend)
                {
                    if (serverSocket.establishConnection(nodeIp))
                    {
                        struct
                        {
                            RequestResponseHeader header;
                            Message message;
                            unsigned char solutionMiningSeed[32];
                            unsigned char solutionNonce[32];
                            unsigned char signature[64];
                        } packet;

                        packet.header.setSize(sizeof(packet));
                        packet.header.zeroDejavu();
                        packet.header.setType(BROADCAST_MESSAGE);

                        memcpy(packet.message.sourcePublicKey, signingPublicKey, sizeof(packet.message.sourcePublicKey));
                        memcpy(packet.message.destinationPublicKey, computorPublicKey, sizeof(packet.message.destinationPublicKey));

                        unsigned char sharedKeyAndGammingNonce[64];
                        // Default behavior when provided seed is just a signing address
                        // first 32 bytes of sharedKeyAndGammingNonce is set as zeros
                        memset(sharedKeyAndGammingNonce, 0, 32);
                        // If provided seed is the for computor public key, generate sharedKey into first 32 bytes to encrypt message
                        if (memcmp(computorPublicKey, signingPublicKey, 32) == 0)
                        {
                            getSharedKey(signingPrivateKey, computorPublicKey, sharedKeyAndGammingNonce);
                        }
                        // Last 32 bytes of sharedKeyAndGammingNonce is randomly created so that gammingKey[0] = 0 (MESSAGE_TYPE_SOLUTION)
                        unsigned char gammingKey[32];
                        do
                        {
#ifdef PORTABLE
                            portable_rdrand64_step((unsigned long long*)&packet.message.gammingNonce[0]);
                            portable_rdrand64_step((unsigned long long*)&packet.message.gammingNonce[8]);
                            portable_rdrand64_step((unsigned long long*)&packet.message.gammingNonce[16]);
                            portable_rdrand64_step((unsigned long long*)&packet.message.gammingNonce[24]);
#else
                            _rdrand64_step((unsigned long long*) & packet.message.gammingNonce[0]);
                            _rdrand64_step((unsigned long long*) & packet.message.gammingNonce[8]);
                            _rdrand64_step((unsigned long long*) & packet.message.gammingNonce[16]);
                            _rdrand64_step((unsigned long long*) & packet.message.gammingNonce[24]);
#endif
                            memcpy(&sharedKeyAndGammingNonce[32], packet.message.gammingNonce, 32);
                            KangarooTwelve(sharedKeyAndGammingNonce, 64, gammingKey, 32);
                        } while (gammingKey[0]);

                        // Encrypt the message payload
                        unsigned char gamma[32 + 32];
                        KangarooTwelve(gammingKey, sizeof(gammingKey), gamma, sizeof(gamma));
                        for (unsigned int i = 0; i < 32; i++)
                        {
                            packet.solutionMiningSeed[i] = randomSeed[i] ^ gamma[i];
                            packet.solutionNonce[i] = sendNonce[i] ^ gamma[i + 32];
                        }

                        // Sign the message
                        uint8_t digest[32] = {0};
                        uint8_t signature[64] = {0};
                        KangarooTwelve(
                            (unsigned char*)&packet + sizeof(RequestResponseHeader),
                            sizeof(packet) - sizeof(RequestResponseHeader) - 64,
                            digest,
                            32);
                        sign(signingSubseed, signingPublicKey, digest, signature);
                        memcpy(packet.signature, signature, 64);

                        // Send message
                        if (serverSocket.sendData((char*)&packet, packet.header.size()))
                        {
                            std::lock_guard<std::mutex> guard(foundNonceLock);
                            // Send data successfully. Remove it from the queue
                            foundNonce.pop();
                            itemToSend = foundNonce.size();
                        }
                        serverSocket.closeConnection();
                    }
                }

                std::this_thread::sleep_for(std::chrono::duration < double, std::milli>(1000));

                unsigned long long delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timestamp).count();
                if (delta >= 1000)
                {
                    // Get current time in UTC
                    std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    std::tm* utc_time = std::gmtime(&now_time);
                    printf("|   %04d-%02d-%02d %02d:%02d:%02d   |   %llu it/s   |   %d solutions   |   %.10s...   |\n",
                        utc_time->tm_year + 1900, utc_time->tm_mon, utc_time->tm_mday, utc_time->tm_hour, utc_time->tm_min, utc_time->tm_sec,
                        (numberOfMiningIterations - prevNumberOfMiningIterations) * 1000 / delta, numberOfFoundSolutions.load(), miningID);
                    fflush(stdout);
                    prevNumberOfMiningIterations = numberOfMiningIterations;
                    timestamp = std::chrono::steady_clock::now();
                }
            }
        }
        printf("Shutting down...Press Ctrl+C again to force stop.\n");

        // Wait for all threads to join
        for (auto& miningTh : miningThreads)
        {
            if (miningTh.joinable())
            {
                miningTh.join();
            }
        }

        // Print stats
        printf("Hyperidentity sols / nonces: %llu / %llu \n", qinerStat.totalHyperIdentitySols.load(), qinerStat.totalHyperIdentityNonce.load());
        printf("Addition sols / nonces: %llu / %llu \n", qinerStat.totalAdditionSols.load(), qinerStat.totalAdditionNonce.load());

        printf("Qiner is shut down.\n");
    }

    return 0;
}