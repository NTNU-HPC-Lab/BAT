#include <cassert>
#include <cfloat>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <list>
#include <math.h>
#include <stdlib.h>

extern "C" {

#include "md_kernel.cu"

using namespace std;

// Problem Constants
static const float  cutsq        = 16.0f; // Square of cutoff distance
static const int    maxNeighbors = 128;  // Max number of nearest neighbors
static const double domainEdge   = 20.0; // Edge length of the cubic domain
static const float  lj1          = 1.5;  // LJ constants
static const float  lj2          = 2.0;
static const float  EPSILON      = 0.1f; // Relative Error between CPU/GPU

// Select which precision that are used in the calculations
// And define the replacements for the template inputs
#if PRECISION == 32
    #define T float
    #define forceVecType float3
    #define posVecType float4
    #define texReader texReader_sp
#elif PRECISION == 64
    #define T double
    #define forceVecType double3
    #define posVecType double4
    #define texReader texReader_dp
#endif

#define useTexture TEXTURE_MEMORY

// Forward Declarations
float runTest(const string& testName);

inline T distance(const posVecType* position, const int i, const int j);

inline void insertInOrder(std::list<T>& currDist, std::list<int>& currList,
        const int j, const T distIJ, const int maxNeighbors);

inline int buildNeighborList(const int nAtom, const posVecType* position, int* neighborList);

inline int populateNeighborList(std::list<T>& currDist,
        std::list<int>& currList, const int j, const int nAtom,
        int* neighborList);

// ****************************************************************************
// Function: checkResults
//
// Purpose:
//   Check device results against cpu results -- this is the CPU equivalent of
//
// Arguments:
//      d_force:   forces calculated on the device
//      position:  positions of atoms
//      neighList: atom neighbor list
//      nAtom:     number of atoms
// Returns:  true if results match, false otherwise
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
bool checkResults(forceVecType* d_force, posVecType *position, int *neighList, int nAtom)
{
    for (int i = 0; i < nAtom; i++)
    {
        posVecType ipos = position[i];
        forceVecType f = {0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < maxNeighbors)
        {
            int jidx = neighList[j*nAtom + i];
            posVecType jpos = position[jidx];
            // Calculate distance
            T delx = ipos.x - jpos.x;
            T dely = ipos.y - jpos.y;
            T delz = ipos.z - jpos.z;
            T r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {

                r2inv = 1.0f/r2inv;
                T r6inv = r2inv * r2inv * r2inv;
                T force = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x += delx * force;
                f.y += dely * force;
                f.z += delz * force;
            }
            j++;
        }
        // Check the results
        T diffx = (d_force[i].x - f.x) / d_force[i].x;
        T diffy = (d_force[i].y - f.y) / d_force[i].y;
        T diffz = (d_force[i].z - f.z) / d_force[i].z;
        T err = sqrt(diffx*diffx) + sqrt(diffy*diffy) + sqrt(diffz*diffz);
        if (err > (3.0 * EPSILON))
        {
            cout << "Test Failed, idx: " << i << " diff: " << err << "\n";
            cout << "f.x: " << f.x << " df.x: " << d_force[i].x << "\n";
            cout << "f.y: " << f.y << " df.y: " << d_force[i].y << "\n";
            cout << "f.z: " << f.z << " df.z: " << d_force[i].z << "\n";
            cout << "Test FAILED\n";
            cerr << "Error: incorrect computed result." << endl;
            return false;
        }
    }
    cout << "Test Passed\n";
    return true;
}

// ********************************************************
// Originated from the SHOC benchmark
// Function: RunBenchmark
//
// Purpose:
//   Executes the md benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ********************************************************

// Return the time used in float to help choosing the best configuration
float md_host() {
    #if PRECISION == 32
        cout << "Running single precision test" << endl;
        return runTest("MD-LJ");
    #elif PRECISION == 64
        cout << "Running double precision test" << endl;
        return runTest("MD-LJ-DP");
    #endif
}

float runTest(const string& testName)
{
    // Problem Parameters
    const int probSizes[4] = { 12288, 24576, 36864, 73728 };
    int sizeClass = 1; // TODO: change this?
    assert(sizeClass >= 0 && sizeClass < 5);
    int nAtom = probSizes[sizeClass - 1];

    // Allocate problem data on host
    posVecType*   position;
    forceVecType* force;
    int* neighborList;

    cudaMallocHost((void**)&position, nAtom*sizeof(posVecType));
    cudaMallocHost((void**)&force,    nAtom*sizeof(forceVecType));
    cudaMallocHost((void**)&neighborList, nAtom*maxNeighbors*sizeof(int));

    // Allocate device memory for position and force
    forceVecType* d_force;
    posVecType*   d_position;
    cudaMalloc((void**)&d_force,    nAtom*sizeof(forceVecType));
    cudaMalloc((void**)&d_position, nAtom*sizeof(posVecType));

    // Allocate device memory for neighbor list
    int* d_neighborList;
    cudaMalloc((void**)&d_neighborList, nAtom*maxNeighbors*sizeof(int));

    cout << "Initializing test problem (this can take several minutes for large problems)" << endl;

    // Seed random number generator
    srand48(8650341L);

    // Initialize positions -- random distribution in cubic domain
    // domainEdge constant specifies edge length
    for (int i = 0; i < nAtom; i++)
    {
        position[i].x = (T)(drand48() * domainEdge);
        position[i].y = (T)(drand48() * domainEdge);
        position[i].z = (T)(drand48() * domainEdge);
    }

    if (useTexture)
    {
        // Set up 1D texture to cache position info
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Bind a 1D texture to the position array
        cudaBindTexture(0, posTexture, d_position, channelDesc, nAtom*sizeof(float4));

        cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<int4>();

        // Bind a 1D texture to the position array
        cudaBindTexture(0, posTexture_dp, d_position, channelDesc2, nAtom*sizeof(double4));
    }

    // Keep track of how many atoms are within the cutoff distance to
    // accurately calculate FLOPS later
    int totalPairs = buildNeighborList(nAtom, position, neighborList);

    cout << "Finished.\n";
    cout << totalPairs << " of " << nAtom*maxNeighbors << " pairs within cutoff distance = " <<
            100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %" << endl;

    // Time the transfer of input data to the GPU
    cudaEvent_t inputTransfer_start, inputTransfer_stop;
    cudaEventCreate(&inputTransfer_start);
    cudaEventCreate(&inputTransfer_stop);

    cudaEventRecord(inputTransfer_start, 0);
    // Copy neighbor list data to GPU
    cudaMemcpy(d_neighborList, neighborList, maxNeighbors*nAtom*sizeof(int), cudaMemcpyHostToDevice);
    // Copy position to GPU
    cudaMemcpy(d_position, position, nAtom*sizeof(posVecType), cudaMemcpyHostToDevice);
    cudaEventRecord(inputTransfer_stop, 0);
    cudaEventSynchronize(inputTransfer_stop);

    // Get elapsed time
    float inputTransfer_time = 0.0f;
    cudaEventElapsedTime(&inputTransfer_time, inputTransfer_start, inputTransfer_stop);
    inputTransfer_time *= 1.e-3;

    int blockSize = BLOCK_SIZE;
    int gridSize  = ceil((double)nAtom / (double)blockSize / (double) WORK_PER_THREAD);

    // Warm up the kernel and check correctness
    compute_lj_force<<<gridSize, blockSize>>>
                    (d_force, d_position, maxNeighbors, d_neighborList, cutsq, lj1, lj2, nAtom);
    cudaDeviceSynchronize();

    // Copy back forces
    cudaMemcpy(force, d_force, nAtom*sizeof(forceVecType), cudaMemcpyDeviceToHost);

    // If results are incorrect, skip the performance tests
    cout << "Performing Correctness Check (can take several minutes)\n";
    if (!checkResults(force, position, neighborList, nAtom))
    {
        throw "Correctness verification failed";
    }

    // For measuring the time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalElapsedTime = 0.0;

    int passes = 10;
    int iter   = 1;
    
    for (int i = 0; i < passes; i++)
    {
        // Start the timing
        cudaEventRecord(start, 0);

        // Other kernels will be involved in true parallel versions
        for (int j = 0; j < iter; j++)
        {
            compute_lj_force<<<gridSize, blockSize>>>
                (d_force, d_position, maxNeighbors, d_neighborList, cutsq, lj1, lj2, nAtom);
        }
        
        // Stop the events and save elapsed time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        totalElapsedTime += elapsedTime;
    }

    // Clean up
    // Host
    cudaFreeHost(position);
    cudaFreeHost(force);
    cudaFreeHost(neighborList);
    // Device
    cudaUnbindTexture(posTexture);
    cudaFree(d_position);
    cudaFree(d_force);
    cudaFree(d_neighborList);

    return totalElapsedTime;
}

// ********************************************************
// Function: distance
//
// Purpose:
//   Calculates distance squared between two atoms
//
// Arguments:
//   position: atom position information
//   i, j: indexes of the two atoms
//
// Returns:  the computed distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
inline T distance(const posVecType* position, const int i, const int j)
{
    posVecType ipos = position[i];
    posVecType jpos = position[j];
    T delx = ipos.x - jpos.x;
    T dely = ipos.y - jpos.y;
    T delz = ipos.z - jpos.z;
    T r2inv = delx * delx + dely * dely + delz * delz;
    return r2inv;
}

// ********************************************************
// Function: insertInOrder
//
// Purpose:
//   Adds atom j to current neighbor list and distance list
//   if it's distance is low enough.
//
// Arguments:
//   currDist: distance between current atom and each of its neighbors in the
//             current list, sorted in ascending order
//   currList: neighbor list for current atom, sorted by distance in asc. order
//   j:        atom to insert into neighbor list
//   distIJ:   distance between current atom and atom J
//   maxNeighbors: max length of neighbor list
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
inline void insertInOrder(list<T>& currDist, list<int>& currList, const int j, const T distIJ, const int maxNeighbors)
{

    typename list<T>::iterator   it;
    typename list<int>::iterator it2;

    it2 = currList.begin();

    T currMax = currDist.back();

    if (distIJ > currMax) return;

    for (it=currDist.begin(); it!=currDist.end(); it++)
    {
        if (distIJ < (*it))
        {
            // Insert into appropriate place in list
            currDist.insert(it,distIJ);
            currList.insert(it2, j);

            // Trim end of list
            currList.resize(maxNeighbors);
            currDist.resize(maxNeighbors);
            return;
        }
        it2++;
    }
}

// ********************************************************
// Function: buildNeighborList
//
// Purpose:
//   Builds the neighbor list structure for all atoms for GPU coalesced reads
//   and counts the number of pairs within the cutoff distance, so
//   the benchmark gets an accurate FLOPS count
//
// Arguments:
//   nAtom:    total number of atoms
//   position: pointer to the atom's position information
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//   Jeremy Meredith, Tue Oct  9 17:35:16 EDT 2012
//   On some slow systems and without optimization, this
//   could take a while.  Give users a rough completion
//   percentage so they don't give up.
//
// ********************************************************
inline int buildNeighborList(const int nAtom, const posVecType* position, int* neighborList)
{
    int totalPairs = 0;
    // Build Neighbor List
    // Find the nearest N atoms to each other atom, where N = maxNeighbors
    for (int i = 0; i < nAtom; i++)
    {
        // Print progress every 10% completion.
        if (int((i+1)/(nAtom/10)) > int(i/(nAtom/10)))
            cout << "  " << 10*int((i+1)/(nAtom/10)) << "% done\n";

        // Current neighbor list for atom i, initialized to -1
        list<int>   currList(maxNeighbors, -1);
        // Distance to those neighbors.  We're populating this with the
        // closest neighbors, so initialize to FLT_MAX
        list<T> currDist(maxNeighbors, FLT_MAX);

        for (int j = 0; j < nAtom; j++)
        {
            if (i == j) continue; // An atom cannot be its own neighbor

            // Calculate distance and insert in order into the current lists
            T distIJ = distance(position, i, j);
            insertInOrder(currDist, currList, j, distIJ, maxNeighbors);
        }
        // We should now have the closest maxNeighbors neighbors and their
        // distances to atom i. Populate the neighbor list data structure
        // for GPU coalesced reads.
        // The populate method returns how many of the maxNeighbors closest
        // neighbors are within the cutoff distance.  This will be used to
        // calculate GFLOPS later.
        totalPairs += populateNeighborList(currDist, currList, i, nAtom, neighborList);
    }
    return totalPairs;
}

// ********************************************************
// Function: populateNeighborList
//
// Purpose:
//   Populates the neighbor list structure for a *single* atom for
//   GPU coalesced reads and counts the number of pairs within the cutoff
//   distance, (for current atom) so the benchmark gets an accurate FLOPS count
//
// Arguments:
//   currDist: distance between current atom and each of its maxNeighbors
//             neighbors
//   currList: current list of neighbors
//   i:        current atom
//   nAtom:    total number of atoms
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
inline int populateNeighborList(list<T>& currDist,
        list<int>& currList, const int i, const int nAtom,
        int* neighborList)
{
    int idx = 0;
    int validPairs = 0; // Pairs of atoms closer together than the cutoff

    // Iterate across distance and neighbor list
    typename list<T>::iterator distanceIter = currDist.begin();
    for (list<int>::iterator neighborIter = currList.begin(); neighborIter != currList.end(); neighborIter++)
    {
        // Populate packed neighbor list
        neighborList[(idx * nAtom) + i] = *neighborIter;

        // If the distance is less than cutoff, increment valid counter
        if (*distanceIter < cutsq)
            validPairs++;

        // Increment idx and distance iterator
        idx++;
        distanceIter++;
    }
    return validPairs;
}
}
