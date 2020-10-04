#include <iostream>
#include <cuda_runtime_api.h>
#include "tuner_api.h" // KTT API
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver

using namespace std;

void md5_2words(unsigned int *words, unsigned int len, unsigned int *digest);
void IndexToKey(unsigned int index, int byteLength, int valsPerByte, unsigned char vals[8]);

int main(int argc, char* argv[]) {
    // Tune md5hash kernel
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/md5hash/md5hash_kernel.cu";
    string referenceKernelFile = "./reference_kernel.cu";
    string kernelName("FindKeyWithDigest_Kernel");
    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

    // Problem sizes used in the SHOC benchmark
    const int sizes_byteLength[4]  = { 7,  5,  6,  5};
    const int sizes_valsPerByte[4] = {10, 35, 25, 70};
    uint inputProblemSize = 1; // Default to the first problem size if no input

    string tuningTechnique = "";

    // If only one extra argument and the flag is set for size
    if (argc == 2 && (string(argv[1]) == "--size" || string(argv[1]) == "-s")) {
        cerr << "Error: You need to specify an integer for the problem size." << endl;
        exit(1);
    }

    // If only one extra argument and the flag is set for tuning technique
    if (argc == 2 && (string(argv[1]) == "--technique" || string(argv[1]) == "-t")) {
        cerr << "Error: You need to specify a tuning technique." << endl;
        exit(1);
    }

    // Check if the provided arguments does not match in size
    if ((argc - 1) % 2 != 0) {
        cerr << "Error: You need to specify correct number of input arguments." << endl;
        exit(1);
    }

    // Loop arguments and add if found
    for (int i = 1; i < argc; i++) {
        // Skip the argument value iterations
        if (i % 2 == 0) {
            continue;
        }

        // Check for problem size
        if (string(argv[i]) == "--size" || string(argv[i]) == "-s") {
            try {
                inputProblemSize = stoi(argv[i + 1]);

                // Ensure the input problem size is between 1 and 4
                if (inputProblemSize < 1 || inputProblemSize > 4) {
                    cerr << "Error: The problem size needs to be an integer in the range 1 to 4." << endl;
                    exit(1);
                }
            } catch (const invalid_argument &error) {
                cerr << "Error: You need to specify an integer for the problem size." << endl;
                exit(1);
            }
        // Check for tuning technique
        } else if (string(argv[i]) == "--technique" || string(argv[i]) == "-t") {
            tuningTechnique = argv[i + 1];
        } else {
            cerr << "Error: Unsupported argument " << "`" << argv[i] << "`" << endl;
            exit(1);
        }
    }

    // Calculate input arguments
    const int byteLength = sizes_byteLength[inputProblemSize-1];   
    const int valsPerByte = sizes_valsPerByte[inputProblemSize-1];

    int keyspace = 1;
    for (int i=0; i<byteLength; ++i) {
        if (keyspace >= 0x7fffffff / valsPerByte) {
            // error, we're about to overflow a signed int
            return -1;
        }
        keyspace *= valsPerByte;
    }
        
    srandom(time(NULL));
    int randomIndex = random() % keyspace;;
    unsigned char randomKey[8] = {0,0,0,0, 0,0,0,0};
    unsigned int randomDigest[4];
    IndexToKey(randomIndex, byteLength, valsPerByte, randomKey);
    md5_2words((unsigned int*)randomKey, byteLength, randomDigest);

    int digest[4] = {(int) randomDigest[0], (int) randomDigest[1], (int) randomDigest[2], (int) randomDigest[3]};
    vector<int> foundIndex = {-1};
    vector<char> foundKey = {0,0,0,0,0,0,0,0};
    vector<int> foundDigest = {0,0,0,0};

    size_t totalSize = ceil((double)keyspace / (double)valsPerByte);

    const ktt::DimensionVector totalThreads(totalSize);
    const ktt::DimensionVector blockSize;
    const ktt::DimensionVector blockSizeReference(384);

    // Add kernel and reference kernel
    ktt::KernelId kernelId = auto_tuner.addKernelFromFile(kernelFile, kernelName, totalThreads, blockSize);
    ktt::KernelId referenceKernelId = auto_tuner.addKernelFromFile(referenceKernelFile, kernelName, totalThreads, blockSizeReference);

    // Add arguments for kernel
    ktt::ArgumentId searchDigest0Id = auto_tuner.addArgumentScalar(digest[0]);
    ktt::ArgumentId searchDigest1Id = auto_tuner.addArgumentScalar(digest[1]);
    ktt::ArgumentId searchDigest2Id = auto_tuner.addArgumentScalar(digest[2]);
    ktt::ArgumentId searchDigest3Id = auto_tuner.addArgumentScalar(digest[3]);
    ktt::ArgumentId keyspaceId = auto_tuner.addArgumentScalar(keyspace);
    ktt::ArgumentId byteLengthId = auto_tuner.addArgumentScalar(byteLength);
    ktt::ArgumentId valsPerByteId = auto_tuner.addArgumentScalar(valsPerByte);
    ktt::ArgumentId foundIndexId = auto_tuner.addArgumentVector(foundIndex, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId foundKeyId = auto_tuner.addArgumentVector(foundKey, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId foundDigestId = auto_tuner.addArgumentVector(foundDigest, ktt::ArgumentAccessType::ReadWrite);

    // Add arguments to kernel and reference kernel
    auto_tuner.setKernelArguments(kernelId, vector<ktt::ArgumentId>{
        searchDigest0Id,
        searchDigest1Id,
        searchDigest2Id,
        searchDigest3Id,
        keyspaceId,
        byteLengthId,
        valsPerByteId,
        foundIndexId,
        foundKeyId,
        foundDigestId
    });
    auto_tuner.setKernelArguments(referenceKernelId, vector<ktt::ArgumentId>{
        searchDigest0Id,
        searchDigest1Id,
        searchDigest2Id,
        searchDigest3Id,
        keyspaceId,
        byteLengthId,
        valsPerByteId,
        foundIndexId,
        foundKeyId,
        foundDigestId
    });

    // Get the maximum threads per block 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
    // Define a vector of block sizes from 1 to maximum threads per block
    vector<long unsigned int> block_sizes = {};
    for(int i = 1; i < (maxThreads+1); i++) {
        block_sizes.push_back(i);
    }

    // Add parameters to tune
    auto_tuner.addParameter(kernelId, "BLOCK_SIZE", block_sizes);
    auto_tuner.addParameter(kernelId, "WORK_PER_THREAD_FACTOR", {1, 2, 3, 4, 5});
    auto_tuner.addParameter(kernelId, "ROUND_STYLE", {0, 1});
    auto_tuner.addParameter(kernelId, "UNROLL_LOOP_1", {0, 1});
    auto_tuner.addParameter(kernelId, "UNROLL_LOOP_2", {0, 1});
    auto_tuner.addParameter(kernelId, "UNROLL_LOOP_3", {0, 1});
    auto_tuner.addParameter(kernelId, "INLINE_1", {0, 1});
    auto_tuner.addParameter(kernelId, "INLINE_2", {0, 1});

    // The block size (local size) base (1) is multiplied by the BLOCK_SIZE parameter
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE", ktt::ModifierAction::Multiply);
    // The grid size (global size) base (totalThreads) is divided by BLOCK_SIZE and WORK_PER_THREAD_FACTOR
    // The grid size value is then multiplied by the BLOCK_SIZE parameter because it will be divided by BLOCK_SIZE at launch in KTT (without using ceil(), which is necessary)
    auto globalModifier = [](const size_t size, const std::vector<size_t>& vector) {
        return int(ceil(double(size) / double(vector.at(0)) / double(vector.at(1)))) * vector.at(0);
    };
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, 
        std::vector<std::string>{"BLOCK_SIZE", "WORK_PER_THREAD_FACTOR"}, globalModifier);

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.setReferenceKernel(kernelId, referenceKernelId, vector<ktt::ParameterPair>{}, vector<ktt::ArgumentId>{foundIndexId, foundKeyId, foundDigestId});

    // Select the tuning technique for this benchmark
    if (tuningTechnique == "annealing") {
        double maxTemperature = 4.0f;
        auto_tuner.setSearchMethod(ktt::SearchMethod::Annealing, {maxTemperature});
    } else if (tuningTechnique == "mcmc") {
        auto_tuner.setSearchMethod(ktt::SearchMethod::MCMC, {});
    } else if (tuningTechnique == "random") {
        auto_tuner.setSearchMethod(ktt::SearchMethod::RandomSearch, {});
    } else if (tuningTechnique == "brute_force") {
        auto_tuner.setSearchMethod(ktt::SearchMethod::FullSearch, {});
    } else {
        cerr << "Error: Unsupported tuning technique: `" << tuningTechnique << "`." << endl;
        exit(1);
    }

    // Set the tuner to print in nanoseconds
    auto_tuner.setPrintingTimeUnit(ktt::TimeUnit::Nanoseconds);

    // Tune the kernel
    auto_tuner.tuneKernel(kernelId);

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromKTTResults(auto_tuner.getBestComputationResult(kernelId), "best-" + kernelName + "-results.json", inputProblemSize, tuningTechnique);

    // Print the results to cout and save it as a CSV file
    auto_tuner.printResult(kernelId, cout, ktt::PrintFormat::Verbose);
    auto_tuner.printResult(kernelId, kernelName + "-results.csv", ktt::PrintFormat::CSV);

    return 0;
}

// Helper functions: 
// leftrotate function definition
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

#define F(x,y,z) ((x & y) | ((~x) & z))
#define G(x,y,z) ((x & z) | ((~z) & y))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | (~z)))

#define ROUND_INPLACE_VIA_SHIFT(w, r, k, v, x, y, z, func)       \
{                                                                \
    v += func(x,y,z) + w + k;                                    \
    v = x + LEFTROTATE(v, r);                                    \
}

// Here, we pick which style of ROUND we use.
#define ROUND ROUND_INPLACE_VIA_SHIFT

void md5_2words(unsigned int *words, unsigned int len, unsigned int *digest) {
    // For any block but the first one, these should be passed in, not
    // initialized, but we are assuming we only operate on a single block.
    unsigned int h0 = 0x67452301;
    unsigned int h1 = 0xefcdab89;
    unsigned int h2 = 0x98badcfe;
    unsigned int h3 = 0x10325476;

    unsigned int a = h0;
    unsigned int b = h1;
    unsigned int c = h2;
    unsigned int d = h3;

    unsigned int WL = len * 8;
    unsigned int W0 = words[0];
    unsigned int W1 = words[1];

    switch (len)
    {
      case 0: W0 |= 0x00000080; break;
      case 1: W0 |= 0x00008000; break;
      case 2: W0 |= 0x00800000; break;
      case 3: W0 |= 0x80000000; break;
      case 4: W1 |= 0x00000080; break;
      case 5: W1 |= 0x00008000; break;
      case 6: W1 |= 0x00800000; break;
      case 7: W1 |= 0x80000000; break;
    }

    // args: word data, per-round shift amt, constant, 4 vars, function macro
    ROUND(W0,   7, 0xd76aa478, a, b, c, d, F);
    ROUND(W1,  12, 0xe8c7b756, d, a, b, c, F);
    ROUND(0,   17, 0x242070db, c, d, a, b, F);
    ROUND(0,   22, 0xc1bdceee, b, c, d, a, F);
    ROUND(0,    7, 0xf57c0faf, a, b, c, d, F);
    ROUND(0,   12, 0x4787c62a, d, a, b, c, F);
    ROUND(0,   17, 0xa8304613, c, d, a, b, F);
    ROUND(0,   22, 0xfd469501, b, c, d, a, F);
    ROUND(0,    7, 0x698098d8, a, b, c, d, F);
    ROUND(0,   12, 0x8b44f7af, d, a, b, c, F);
    ROUND(0,   17, 0xffff5bb1, c, d, a, b, F);
    ROUND(0,   22, 0x895cd7be, b, c, d, a, F);
    ROUND(0,    7, 0x6b901122, a, b, c, d, F);
    ROUND(0,   12, 0xfd987193, d, a, b, c, F);
    ROUND(WL,  17, 0xa679438e, c, d, a, b, F);
    ROUND(0,   22, 0x49b40821, b, c, d, a, F);

    ROUND(W1,   5, 0xf61e2562, a, b, c, d, G);
    ROUND(0,    9, 0xc040b340, d, a, b, c, G);
    ROUND(0,   14, 0x265e5a51, c, d, a, b, G);
    ROUND(W0,  20, 0xe9b6c7aa, b, c, d, a, G);
    ROUND(0,    5, 0xd62f105d, a, b, c, d, G);
    ROUND(0,    9, 0x02441453, d, a, b, c, G);
    ROUND(0,   14, 0xd8a1e681, c, d, a, b, G);
    ROUND(0,   20, 0xe7d3fbc8, b, c, d, a, G);
    ROUND(0,    5, 0x21e1cde6, a, b, c, d, G);
    ROUND(WL,   9, 0xc33707d6, d, a, b, c, G);
    ROUND(0,   14, 0xf4d50d87, c, d, a, b, G);
    ROUND(0,   20, 0x455a14ed, b, c, d, a, G);
    ROUND(0,    5, 0xa9e3e905, a, b, c, d, G);
    ROUND(0,    9, 0xfcefa3f8, d, a, b, c, G);
    ROUND(0,   14, 0x676f02d9, c, d, a, b, G);
    ROUND(0,   20, 0x8d2a4c8a, b, c, d, a, G);

    ROUND(0,    4, 0xfffa3942, a, b, c, d, H);
    ROUND(0,   11, 0x8771f681, d, a, b, c, H);
    ROUND(0,   16, 0x6d9d6122, c, d, a, b, H);
    ROUND(WL,  23, 0xfde5380c, b, c, d, a, H);
    ROUND(W1,   4, 0xa4beea44, a, b, c, d, H);
    ROUND(0,   11, 0x4bdecfa9, d, a, b, c, H);
    ROUND(0,   16, 0xf6bb4b60, c, d, a, b, H);
    ROUND(0,   23, 0xbebfbc70, b, c, d, a, H);
    ROUND(0,    4, 0x289b7ec6, a, b, c, d, H);
    ROUND(W0,  11, 0xeaa127fa, d, a, b, c, H);
    ROUND(0,   16, 0xd4ef3085, c, d, a, b, H);
    ROUND(0,   23, 0x04881d05, b, c, d, a, H);
    ROUND(0,    4, 0xd9d4d039, a, b, c, d, H);
    ROUND(0,   11, 0xe6db99e5, d, a, b, c, H);
    ROUND(0,   16, 0x1fa27cf8, c, d, a, b, H);
    ROUND(0,   23, 0xc4ac5665, b, c, d, a, H);

    ROUND(W0,   6, 0xf4292244, a, b, c, d, I);
    ROUND(0,   10, 0x432aff97, d, a, b, c, I);
    ROUND(WL,  15, 0xab9423a7, c, d, a, b, I);
    ROUND(0,   21, 0xfc93a039, b, c, d, a, I);
    ROUND(0,    6, 0x655b59c3, a, b, c, d, I);
    ROUND(0,   10, 0x8f0ccc92, d, a, b, c, I);
    ROUND(0,   15, 0xffeff47d, c, d, a, b, I);
    ROUND(W1,  21, 0x85845dd1, b, c, d, a, I);
    ROUND(0,    6, 0x6fa87e4f, a, b, c, d, I);
    ROUND(0,   10, 0xfe2ce6e0, d, a, b, c, I);
    ROUND(0,   15, 0xa3014314, c, d, a, b, I);
    ROUND(0,   21, 0x4e0811a1, b, c, d, a, I);
    ROUND(0,    6, 0xf7537e82, a, b, c, d, I);
    ROUND(0,   10, 0xbd3af235, d, a, b, c, I);
    ROUND(0,   15, 0x2ad7d2bb, c, d, a, b, I);
    ROUND(0,   21, 0xeb86d391, b, c, d, a, I);

    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;

    // write the final result out
    digest[0] = h0;
    digest[1] = h1;
    digest[2] = h2;
    digest[3] = h3;
}

void IndexToKey(unsigned int index, int byteLength, int valsPerByte, unsigned char vals[8]) {
    for (int i = 0; i < 8; i++) {
        vals[i] = index % valsPerByte;
        index /= valsPerByte;
    }
}
