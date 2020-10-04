#pragma once

#include "tuner_api.h"

using namespace std;

class TunableScan : public ktt::TuningManipulator {
public:
    TunableScan(const vector<ktt::KernelId> kernelIds,
                const int sizef,
                const int sized,
                ktt::ArgumentId inDataIdf,
                ktt::ArgumentId inDataIdd,
                ktt::ArgumentId sizeIdf,
                ktt::ArgumentId sizeIdd,
                ktt::ArgumentId blockSumsIdf,
                ktt::ArgumentId blockSumsIdd,
                ktt::ArgumentId outDataIdf,
                ktt::ArgumentId outDataIdd) :
        kernelIds(kernelIds),
        sizef(sizef),
        sized(sized),
        inDataIdf(inDataIdf),
        inDataIdd(inDataIdd),
        sizeIdf(sizeIdf),
        sizeIdd(sizeIdd),
        blockSumsIdf(blockSumsIdf),
        blockSumsIdd(blockSumsIdd),
        outDataIdf(outDataIdf),
        outDataIdd(outDataIdd)
    {}

    // Run the kernels from SHOC's Scan.cu
    void launchComputation(const ktt::KernelId kernelId) override {
        vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
        size_t blockSize = getParameterValue("BLOCK_SIZE", parameterValues);
        size_t gridSize = getParameterValue("GRID_SIZE", parameterValues);
        size_t totalSize = blockSize * gridSize;

        resizeArgumentVector(blockSumsIdf, blockSize, true);
        resizeArgumentVector(blockSumsIdd, blockSize, true);

        // Run "reduce" kernel
        runKernel(kernelIds[0], ktt::DimensionVector(totalSize, 1, 1), ktt::DimensionVector(blockSize, 1, 1));
        
        // Run "scan_single_block_helper" kernel
        // Grid size for this kernel is 1. Block size is chosen as the global size because it will be divided by block size in runKernel()
        runKernel(kernelIds[1], ktt::DimensionVector(blockSize, 1, 1), ktt::DimensionVector(blockSize, 1, 1));
        
        // Run "bottom_scan_helper" kernel
        runKernel(kernelIds[2], ktt::DimensionVector(totalSize, 1, 1), ktt::DimensionVector(blockSize, 1, 1));
    }

private:
    vector<ktt::KernelId> kernelIds;
    const int sizef;
    const int sized;
    ktt::ArgumentId inDataIdf;
    ktt::ArgumentId inDataIdd;
    ktt::ArgumentId sizeIdf;
    ktt::ArgumentId sizeIdd;
    ktt::ArgumentId blockSumsIdf;
    ktt::ArgumentId blockSumsIdd;
    ktt::ArgumentId outDataIdf;
    ktt::ArgumentId outDataIdd;
};