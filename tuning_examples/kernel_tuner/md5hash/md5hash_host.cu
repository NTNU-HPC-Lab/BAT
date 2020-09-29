#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cudacommon.h"

extern "C" {

#include "md5hash_kernel.cu"

using namespace std;

// ****************************************************************************
// Function:  FindKeyspaceSize
//
// Purpose:
///   Multiply out the byteLength by valsPerByte to find the 
///   total size of the key space, with error checking.
//
// Arguments:
//   byteLength    number of bytes in a key
//   valsPerByte   number of values each byte can take on
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
__host__ __device__ int FindKeyspaceSize(int byteLength, int valsPerByte)
{
    int keyspace = 1;
    for (int i=0; i<byteLength; ++i)
    {
        if (keyspace >= 0x7fffffff / valsPerByte)
        {
            // error, we're about to overflow a signed int
            return -1;
        }
        keyspace *= valsPerByte;
    }
    return keyspace;
}




// ****************************************************************************
// Function:  AsHex
//
// Purpose:
///   For a given key string, return the raw hex string for its bytes.
//
// Arguments:
//   vals       key string
//   len        length of key string
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
std::string AsHex(unsigned char *vals, int len)
{
    ostringstream out;
    char tmp[256];
    for (int i=0; i<len; ++i)
    {
        sprintf(tmp, "%2.2X", vals[i]);
        out << tmp;
    }
    return out.str();
}


// ****************************************************************************
// Function:  FindKeyWithDigest_GPU
//
// Purpose:
///   On the GPU, search the key space to find a key with the given digest.
//
// Arguments:
//   searchDigest    the digest to search for
//   byteLength      number of bytes in a key
//   valsPerByte     number of values each byte can take on
//   foundIndex      output - the index of the found key (if found)
//   foundKey        output - the string of the found key (if found)
//   foundDigest     output - the digest of the found key (if found)
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
float FindKeyWithDigest_GPU(const unsigned int searchDigest[4],
                             const int byteLength,
                             const int valsPerByte,
                             int *foundIndex,
                             unsigned char foundKey[8],
                             unsigned int foundDigest[4])
{
    int keyspace = FindKeyspaceSize(byteLength, valsPerByte);

    //
    // allocate output buffers
    //
    int *d_foundIndex;
    cudaMalloc((void**)&d_foundIndex, sizeof(int) * 1);
    CHECK_CUDA_ERROR();
    unsigned char *d_foundKey;
    cudaMalloc((void**)&d_foundKey, 8);
    CHECK_CUDA_ERROR();
    unsigned int *d_foundDigest;
    cudaMalloc((void**)&d_foundDigest, sizeof(unsigned int) * 4);
    CHECK_CUDA_ERROR();

    //
    // initialize output buffers to show no found result
    //
    cudaMemcpy(d_foundIndex, foundIndex, sizeof(int) * 1, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    cudaMemcpy(d_foundKey, foundKey, 8, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    cudaMemcpy(d_foundDigest, foundDigest, sizeof(unsigned int) * 4, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();

    //
    // calculate work thread shape
    //
    int nthreads = BLOCK_SIZE;
    size_t nblocks  = ceil((double(keyspace) / double(valsPerByte)) / double(nthreads) / double(WORK_PER_THREAD_FACTOR));

    // Initialize timers
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    float totalElapsedTime = 0.0;

    // Start the timing
    cudaEventRecord(start, 0);
    //
    // run the kernel
    //
    FindKeyWithDigest_Kernel<<<nblocks, nthreads>>>(searchDigest[0],
                                                    searchDigest[1],
                                                    searchDigest[2],
                                                    searchDigest[3],
                                                    keyspace,
                                                    byteLength, valsPerByte,
                                                    d_foundIndex,
                                                    d_foundKey,
                                                    d_foundDigest);

    CHECK_CUDA_ERROR();

    // Stop the events and save elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    totalElapsedTime += elapsedTime;

    //
    // read the (presumably) found key
    //
    cudaMemcpy(foundIndex, d_foundIndex, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR();
    cudaMemcpy(foundKey, d_foundKey, 8, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR();
    cudaMemcpy(foundDigest, d_foundDigest, sizeof(unsigned int) * 4, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR();

    //
    // free device memory
    //
    cudaFree(d_foundIndex);
    CHECK_CUDA_ERROR();
    cudaFree(d_foundKey);
    CHECK_CUDA_ERROR();
    cudaFree(d_foundDigest);
    CHECK_CUDA_ERROR();

    //
    // return the runtime in seconds
    //
    return totalElapsedTime;
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the MD5 Hash benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: July 23, 2014
//
// Modifications:
//
// ****************************************************************************
float RunBenchmark() {
    bool verbose = false;

    int size = PROBLEM_SIZE;
    if (size < 1 || size > 4)
    {
        cerr << "ERROR: Invalid size parameter\n";
        throw "ERROR: Invalid size parameter";
        return 100000.0;
    }

    //
    // Determine the shape/size of key space
    //
    const int sizes_byteLength[]  = { 7,  5,  6,  5};
    const int sizes_valsPerByte[] = {10, 35, 25, 70};

    const int byteLength = sizes_byteLength[size-1];   
    const int valsPerByte = sizes_valsPerByte[size-1];

    char atts[1024];
    sprintf(atts, "%dx%d", byteLength, valsPerByte);

    if (verbose)
        cout << "Searching keys of length " << byteLength << " bytes "
             << "and " << valsPerByte << " values per byte" << endl;

    const int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
    if (keyspace < 0)
    {
        cerr << "Error: more than 2^31 bits of entropy is unsupported.\n";
        throw "Error: more than 2^31 bits of entropy is unsupported.";
        return 100000.0;
    }

    if (byteLength > 7)
    {
        cerr << "Error: more than 7 byte key length is unsupported.\n";
        throw "Error: more than 7 byte key length is unsupported.";
        return 100000.0;
    }

    if (verbose)
        cout << "|keyspace| = " << keyspace << " ("<<int(keyspace/1e6)<<"M)" << endl;

    //
    // Choose a random key from the keyspace, and calculate its hash.
    //
    //srandom(12345);
    srandom(time(NULL));

    int passes = 10;
    float totalTime = 0.0;
    for (int pass = 0 ; pass < passes ; ++pass)
    {
        int randomIndex = random() % keyspace;;
        unsigned char randomKey[8] = {0,0,0,0, 0,0,0,0};
        unsigned int randomDigest[4];
        IndexToKey(randomIndex, byteLength, valsPerByte, randomKey);
        md5_2words((unsigned int*)randomKey, byteLength, randomDigest);

        if (verbose)
        {
            cout << endl;
            cout << "--- pass " << pass << " ---" << endl;
            cout << "Looking for random key:" << endl;
            cout << " randomIndex = " << randomIndex << endl;
            cout << " randomKey   = 0x" << AsHex(randomKey, 8/*byteLength*/) << endl;
            cout << " randomDigest= " << AsHex((unsigned char*)randomDigest, 16) << endl;
        }

        //
        // Use the GPU to brute force search the keyspace for this key.
        //
        unsigned int foundDigest[4] = {0,0,0,0};
        int foundIndex = -1;
        unsigned char foundKey[8] = {0,0,0,0, 0,0,0,0};

        // in seconds
        totalTime += FindKeyWithDigest_GPU(randomDigest, byteLength, valsPerByte,
            &foundIndex, foundKey, foundDigest);


        //
        // Double check everything matches (index, key, hash).
        //
        if (foundIndex < 0)
        {
            cerr << "\nERROR: could not find a match.\n";
            throw "ERROR: could not find a match.";
            return 100000.0;
        }
        else if (foundIndex != randomIndex)
        {
            cerr << "\nERROR: mismatch in key index found.\n";
            throw "ERROR: mismatch in key index found.";
            return 100000.0;
        }
        else if (foundKey[0] != randomKey[0] ||
            foundKey[1] != randomKey[1] ||
            foundKey[2] != randomKey[2] ||
            foundKey[3] != randomKey[3] ||
            foundKey[4] != randomKey[4] ||
            foundKey[5] != randomKey[5] ||
            foundKey[6] != randomKey[6] ||
            foundKey[7] != randomKey[7])
        {
            cerr << "\nERROR: mismatch in key value found.\n";
            throw "ERROR: mismatch in key value found.";
            return 100000.0;
        }        
        else if (foundDigest[0] != randomDigest[0] ||
            foundDigest[1] != randomDigest[1] ||
            foundDigest[2] != randomDigest[2] ||
            foundDigest[3] != randomDigest[3])
        {
            cerr << "\nERROR: mismatch in digest of key.\n";
            throw "ERROR: mismatch in digest of key.";
            return 100000.0;
        }
        else
        {
            if (verbose)
                cout << endl << "Successfully found match (index, key, hash):" << endl;
        }

        //
        // Add the calculated performancethe results
        //

        if (verbose)
        {
            cout << " foundIndex  = " << foundIndex << endl;
            cout << " foundKey    = 0x" << AsHex(foundKey, 8/*byteLength*/) << endl;
            cout << " foundDigest = " << AsHex((unsigned char*)foundDigest, 16) << endl;
            cout << endl;
        }
    }
    return totalTime;
}
}
