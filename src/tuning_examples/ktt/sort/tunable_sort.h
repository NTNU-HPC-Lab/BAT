#pragma once

#include "tuner_api.h"

using namespace std;

// Constant from SHOC's Sort.h
const uint SORT_BITS = 32;

class TunableSort : public ktt::TuningManipulator {
public:
    TunableSort(const vector<ktt::KernelId> kernelIds,
                const uint size,
                ktt::ArgumentId startBitId,
                ktt::ArgumentId keysInId,
                ktt::ArgumentId valuesInId,
                ktt::ArgumentId keysOutId,
                ktt::ArgumentId valuesOutId,
                ktt::ArgumentId countersId,
                ktt::ArgumentId counterSumsId,
                ktt::ArgumentId blockOffsetsId,
                ktt::ArgumentId reorderFindBlocksId,
                ktt::ArgumentId numberOfElementsId,
                ktt::ArgumentId blockSumsId,
                ktt::ArgumentId scanOutputId,
                ktt::ArgumentId scanInputId,
                ktt::ArgumentId fullBlockId,
                ktt::ArgumentId storeSumId) :
        kernelIds(kernelIds),
        size(size),
        startBitId(startBitId),
        keysInId(keysInId),
        valuesInId(valuesInId),
        keysOutId(keysOutId),
        valuesOutId(valuesOutId),
        countersId(countersId),
        counterSumsId(counterSumsId),
        blockOffsetsId(blockOffsetsId),
        reorderFindBlocksId(reorderFindBlocksId),
        numberOfElementsId(numberOfElementsId),
        blockSumsId(blockSumsId),
        scanOutputId(scanOutputId),
        scanInputId(scanInputId),
        fullBlockId(fullBlockId),
        storeSumId(storeSumId)
    {}

    // Run the kernels from SHOC's main Sort.cu
    void launchComputation(const ktt::KernelId kernelId) override {
        vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

        const uint SCAN_BLOCK_SIZE = getParameterValue("SCAN_BLOCK_SIZE", parameterValues);
        const uint SORT_BLOCK_SIZE = getParameterValue("SORT_BLOCK_SIZE", parameterValues);
        const uint SCAN_DATA_SIZE = getParameterValue("SCAN_DATA_SIZE", parameterValues);
        const uint SORT_DATA_SIZE = getParameterValue("SORT_DATA_SIZE", parameterValues);

        // Allocate space for block sums in the scan kernel.
        uint numScanElts = size;
        uint level = 0;
        vector<vector<uint>> scanBlockSums;

        do {
            uint numBlocks = max(1, (int) ceil((float) numScanElts / (SORT_DATA_SIZE * SCAN_BLOCK_SIZE)));

            if (numBlocks > 1) {
                // Add space for block sums
                scanBlockSums.push_back(vector<uint>(numBlocks));
                level++;
            }

            numScanElts = numBlocks;
        } while (numScanElts > 1);

        // Add one extra item due to malloc((level + 1) * sizeof(uint*)) from original code
        scanBlockSums.push_back(vector<uint>(1));

        // Each thread in the sort kernel handles "SORT_DATA_SIZE" elements
        size_t numSortGroups = size / (SORT_DATA_SIZE * SORT_BLOCK_SIZE);
        uint itemSize = 32 * numSortGroups;

        vector<uint> counters(itemSize);
        vector<uint> counterSums(itemSize);
        vector<uint> blockOffsets(itemSize);
        updateArgumentVector(countersId, counters.data(), itemSize);
        updateArgumentVector(counterSumsId, counterSums.data(), itemSize);
        updateArgumentVector(blockOffsetsId, blockOffsets.data(), itemSize);

        // Update blocks for "findRadixOffsets" kernel and "reorderData" kernel
        uint reorderFindBlocks = (size / SCAN_DATA_SIZE) / SCAN_BLOCK_SIZE;
        updateArgumentScalar(reorderFindBlocksId, &reorderFindBlocks);
        
        // Swap the input and output keys and values before the loop
        swapKernelArguments(kernelIds[2], keysOutId, keysInId);
        swapKernelArguments(kernelIds[2], valuesOutId, valuesInId);

        // Threads handle either 4 or two elements each
        const size_t radixGlobalWorkSize   = size / SORT_DATA_SIZE;
        const size_t findGlobalWorkSize    = size / SCAN_DATA_SIZE;
        const size_t reorderFindGlobalWorkSize = size / SCAN_DATA_SIZE;

        const size_t reorderBlocks = reorderFindGlobalWorkSize / SCAN_BLOCK_SIZE;

        // Perform Radix Sort (4 bits at a time)
        for (int i = 0; i < SORT_BITS; i += 4) {
            // Update kernel argument
            updateArgumentScalar(startBitId, &i);
            
            // Run "radixSortBlocks" kernel
            runKernel(kernelIds[0], ktt::DimensionVector(radixGlobalWorkSize, 1, 1), ktt::DimensionVector(SORT_BLOCK_SIZE, 1, 1));

            // Run "findRadixOffsets" kernel
            runKernel(kernelIds[1], ktt::DimensionVector(reorderFindGlobalWorkSize, 1, 1), ktt::DimensionVector(SCAN_BLOCK_SIZE, 1, 1));

            // Get the updated value from "findRadixOffsets" kernel
            getArgumentVector(countersId, counters.data());

            scanArrayRecursive(counterSums, counters, 16 * reorderBlocks, 0, scanBlockSums);

            // Update arguments used by "scanArrayRecursive"
            updateArgumentVector(counterSumsId, counterSums.data(), counterSums.size());

            // Run "reorderData" kernel
            runKernel(kernelIds[2], ktt::DimensionVector(reorderFindGlobalWorkSize, 1, 1), ktt::DimensionVector(SCAN_BLOCK_SIZE, 1, 1));
        }
    }

    /**
     * scanArrayRecursive function from SHOC's Sort.cu
    */
    void scanArrayRecursive(vector<uint> &outArray, vector<uint> &inArray, int numElements, int level, vector<vector<uint>> &blockSums) {        
        vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

        const uint SCAN_BLOCK_SIZE = getParameterValue("SCAN_BLOCK_SIZE", parameterValues);
        const uint SORT_DATA_SIZE = getParameterValue("SORT_DATA_SIZE", parameterValues);

        // Kernels handle 8 elems per thread
        unsigned int numBlocks = max(1, (int) ceil((float) numElements / (SORT_DATA_SIZE * SCAN_BLOCK_SIZE)));

        bool fullBlock = (numElements == numBlocks * SORT_DATA_SIZE * SCAN_BLOCK_SIZE);

        const ktt::DimensionVector grid(numBlocks * SCAN_BLOCK_SIZE, 1, 1);
        const ktt::DimensionVector threads(SCAN_BLOCK_SIZE, 1, 1);

        bool storeSum = numBlocks > 1;

        // Update the kernel arguments
        updateArgumentVector(scanOutputId, outArray.data(), outArray.size());
        updateArgumentVector(scanInputId, inArray.data(), inArray.size());
        updateArgumentVector(blockSumsId, blockSums[level].data(), blockSums[level].size());
        updateArgumentScalar(numberOfElementsId, &numElements);
        updateArgumentScalar(fullBlockId, &fullBlock);
        updateArgumentScalar(storeSumId, &storeSum);

        // Run "scan" kernel
        runKernel(kernelIds[4], grid, threads);

        // Get the updated output argument
        getArgumentVector(scanOutputId, outArray.data(), outArray.size());

        if (storeSum) {
            // Get the updated output that is needed for the "vectorAddUniform4" kernel
            getArgumentVector(blockSumsId, blockSums[level].data(), blockSums[level].size());

            // Call this function recursive
            scanArrayRecursive(blockSums[level], blockSums[level], numBlocks, level + 1, blockSums);

            // Update arguments used for the "vectorAddUniform4" kernel
            updateArgumentVector(scanOutputId, outArray.data(), outArray.size());
            updateArgumentVector(blockSumsId, blockSums[level].data(), blockSums[level].size());
            updateArgumentScalar(numberOfElementsId, &numElements);
            
            // Run "vectorAddUniform4" kernel
            runKernel(kernelIds[3], grid, threads);
            getArgumentVector(scanOutputId, outArray.data(), outArray.size());
        }
    }

private:
    vector<ktt::KernelId> kernelIds;
    uint size;
    ktt::ArgumentId startBitId;
    ktt::ArgumentId keysInId;
    ktt::ArgumentId valuesInId;
    ktt::ArgumentId keysOutId;
    ktt::ArgumentId valuesOutId;
    ktt::ArgumentId countersId;
    ktt::ArgumentId counterSumsId;
    ktt::ArgumentId blockOffsetsId;
    ktt::ArgumentId reorderFindBlocksId;
    ktt::ArgumentId numberOfElementsId;
    ktt::ArgumentId blockSumsId;
    ktt::ArgumentId scanOutputId;
    ktt::ArgumentId scanInputId;
    ktt::ArgumentId fullBlockId;
    ktt::ArgumentId storeSumId;
};