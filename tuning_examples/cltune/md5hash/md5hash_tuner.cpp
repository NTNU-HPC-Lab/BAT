#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <limits.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver

using namespace std;

void md5_2words(unsigned int *words, unsigned int len, unsigned int *digest);
void IndexToKey(unsigned int index, int byteLength, int valsPerByte, unsigned char vals[8]);

int main(int argc, char* argv[]) {
    // Problem sizes used in the SHOC benchmark
    const int sizes_byteLength[4]  = { 7,  5,  6,  5};
    const int sizes_valsPerByte[4] = {10, 35, 25, 70};

    uint inputProblemSize = 1; // Default to the first problem size

    string tuningTechnique = "";

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

    string kernelFile = "../../../src/kernels/md5hash/md5hash_kernel.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "FindKeyWithDigest_Kernel" kernel
    string kernelName("FindKeyWithDigest_Kernel");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

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
    
    // Get the maximum threads per block 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
    // Define a vector of block sizes from 1 to maximum threads per block
    vector<long unsigned int> block_sizes = {};
    for(int i = 1; i < (maxThreads+1); i++) {
        block_sizes.push_back(i);
    }
    unsigned long int totalSize = ceil((double)keyspace / (double)valsPerByte);
    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {totalSize}, {1});

    // Add parameters to tune
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", block_sizes);
    auto_tuner.AddParameter(kernel_id, "WORK_PER_THREAD_FACTOR", {1, 2, 3, 4, 5});
    auto_tuner.AddParameter(kernel_id, "ROUND_STYLE", {0, 1});
    auto_tuner.AddParameter(kernel_id, "UNROLL_LOOP_1", {0, 1});
    auto_tuner.AddParameter(kernel_id, "UNROLL_LOOP_2", {0, 1});
    auto_tuner.AddParameter(kernel_id, "UNROLL_LOOP_3", {0, 1});
    auto_tuner.AddParameter(kernel_id, "INLINE_1", {0, 1});
    auto_tuner.AddParameter(kernel_id, "INLINE_2", {0, 1});

  
    // Set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});
    // Divide the total number of threads by the work per thread factor
    auto_tuner.DivGlobalSize(kernel_id, {"WORK_PER_THREAD_FACTOR"});

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {totalSize}, {384});

    // Add arguments for kernel
    auto_tuner.AddArgumentScalar(digest[0]);
    auto_tuner.AddArgumentScalar(digest[1]);
    auto_tuner.AddArgumentScalar(digest[2]);
    auto_tuner.AddArgumentScalar(digest[3]);
    auto_tuner.AddArgumentScalar(keyspace);
    auto_tuner.AddArgumentScalar(byteLength);
    auto_tuner.AddArgumentScalar(valsPerByte);
    auto_tuner.AddArgumentOutput(foundIndex);
    auto_tuner.AddArgumentOutput(foundKey);
    auto_tuner.AddArgumentOutput(foundDigest);

    // Use 50% of the total search space
    double searchFraction = 0.5;

    // Select the tuning technique for this benchmark
    if (tuningTechnique == "annealing") {
        double maxTemperature = 4.0f;
        auto_tuner.UseAnnealing(searchFraction, {maxTemperature});
    } else if (tuningTechnique == "pso") {
        double swarmSize = 4.0f;
        auto_tuner.UsePSO(searchFraction, swarmSize, 0.4, 0.0, 0.4);
    } else if (tuningTechnique == "random") {
        auto_tuner.UseRandomSearch(searchFraction);
    } else if (tuningTechnique == "brute_force") {
        auto_tuner.UseFullSearch();
    } else {
        cerr << "Error: Unsupported tuning technique: `" << tuningTechnique << "`." << endl;
        exit(1);
    }

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json", inputProblemSize, tuningTechnique);

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});

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
