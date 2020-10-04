#include <iostream> // For terminal output
#include <fstream> // For file saving
#include "tuner_api.h" // KTT API
#include "tunable_scan.h" // To help with the launch of kernels
#include "reference_scan.h" // To check the correctness of the computed kernels
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver

using namespace std;

int main(int argc, char* argv[]) {
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/scan/scan_kernel_no_template.cu";

    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA?
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

    const ktt::DimensionVector gridSize;
    const ktt::DimensionVector blockSize;

    vector<ktt::KernelId> kernelIds(3);
    kernelIds[0] = auto_tuner.addKernelFromFile(kernelFile, "reduce_helper", gridSize, blockSize);
    kernelIds[1] = auto_tuner.addKernelFromFile(kernelFile, "scan_single_block_helper", gridSize, blockSize);
    kernelIds[2] = auto_tuner.addKernelFromFile(kernelFile, "bottom_scan_helper", gridSize, blockSize);

    // Problem sizes from SHOC
    int probSizes[4] = { 1, 8, 32, 64 };
    uint inputProblemSize = 1; // Default to first problem size if no input

    // If only one extra argument and the flag is set for size
    if (argc == 2 && (string(argv[1]) == "--size" || string(argv[1]) == "-s")) {
        cerr << "Error: You need to specify an integer for the problem size." << endl;
        exit(1);
    }

    // If more than two extra arguments and flag is set for size
    if (argc > 2 && (string(argv[1]) == "--size" || string(argv[1]) == "-s")) {
        try {
            inputProblemSize = stoi(argv[2]);

            // Ensure the input problem size is between 1 and 4
            if (inputProblemSize < 1 || inputProblemSize > 4) {
                cerr << "Error: The problem size needs to be an integer in the range 1 to 4." << endl;
                exit(1);
            }
        } catch (const invalid_argument &error) {
            cerr << "Error: You need to specify an integer for the problem size." << endl;
            exit(1);
        }
    }

    // Convert to size to MiB
    int sizef = (probSizes[inputProblemSize - 1] * 1024 * 1024) / sizeof(float);
    int sized = (probSizes[inputProblemSize - 1] * 1024 * 1024) / sizeof(double);

    vector<float> inDataf(sizef);
    vector<float> blockSumsf(1024); // New block size is set in the tuning manipulator
    vector<float> outDataf(sizef);
    vector<double> inDatad(sized);
    vector<double> blockSumsd(1024); // New block size is set in the tuning manipulator
    vector<double> outDatad(sized);

    // Initialize arg values
    for (int i = 0; i < sizef; i++) {
        inDataf[i] = i % 2; // Fill with some pattern
        outDataf[i] = -1;
    }
    for (int i = 0; i < sized; i++) {
        inDatad[i] = i % 2; // Fill with some pattern
        outDatad[i] = -1;
    }

    // Add arguments
    ktt::ArgumentId inDataIdf = auto_tuner.addArgumentVector(inDataf, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId inDataIdd = auto_tuner.addArgumentVector(inDatad, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId sizeIdf = auto_tuner.addArgumentScalar(sizef);
    ktt::ArgumentId sizeIdd = auto_tuner.addArgumentScalar(sized);
    ktt::ArgumentId blockSumsIdf = auto_tuner.addArgumentVector(blockSumsf, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId blockSumsIdd = auto_tuner.addArgumentVector(blockSumsd, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId outDataIdf = auto_tuner.addArgumentVector(outDataf, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId outDataIdd = auto_tuner.addArgumentVector(outDatad, ktt::ArgumentAccessType::ReadWrite);

    // Create a composition of the kernels
    ktt::KernelId compositionId = auto_tuner.addComposition("scan", kernelIds, make_unique<TunableScan>(kernelIds,
                                                                                                        sizef,
                                                                                                        sized,
                                                                                                        inDataIdf,
                                                                                                        inDataIdd,
                                                                                                        sizeIdf,
                                                                                                        sizeIdd,
                                                                                                        blockSumsIdf,
                                                                                                        blockSumsIdd,
                                                                                                        outDataIdf,
                                                                                                        outDataIdd));

    // Add arguments for each composition kernel
    // Add arguments for "reduce" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[0], vector<ktt::ArgumentId>{inDataIdf, inDataIdd, blockSumsIdf, blockSumsIdd, sizeIdf, sizeIdd});
    // Add arguments for "scan_single_block_helper" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[1], vector<ktt::ArgumentId>{blockSumsIdf, blockSumsIdd});
    // Add arguments for "bottom_scan" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[2], vector<ktt::ArgumentId>{inDataIdf, inDataIdd, outDataIdf, outDataIdd, blockSumsIdf, blockSumsIdd, sizeIdf, sizeIdd});
   
    // Add parameters to tune
    auto_tuner.addParameter(compositionId, "BLOCK_SIZE", {16, 64, 128, 256, 512});
    auto_tuner.addParameter(compositionId, "GRID_SIZE", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512});
    auto_tuner.addParameter(compositionId, "PRECISION", {32});
    auto_tuner.addParameter(compositionId, "UNROLL_LOOP_1", {0, 1});
    auto_tuner.addParameter(compositionId, "UNROLL_LOOP_2", {0, 1});
    auto_tuner.addParameter(compositionId, "SET_SHARED_MEMORY_SIZE", {1});

    // Add constraint for valid grid and block sizes
    auto launchConstraint = [] (std::vector<size_t> v) {
        return (v[1] <= v[0]);
    };
    auto_tuner.addConstraint(compositionId, {"BLOCK_SIZE", "GRID_SIZE"}, launchConstraint);

    
    // Set reference class for correctness verification and compare to the computed result
    // Change input between inDatad (double precision) and inDataf (single precision) for reference testing
    // auto_tuner.setReferenceClass(compositionId, make_unique<ReferenceScan>(inDataf), vector<ktt::ArgumentId>{outDataIdf});

    // Set the tuner to print in nanoseconds
    auto_tuner.setPrintingTimeUnit(ktt::TimeUnit::Nanoseconds);

    // Tune the composition of kernels
    auto_tuner.tuneKernel(compositionId);

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromKTTResults(auto_tuner.getBestComputationResult(compositionId), "best-scan-results.json", inputProblemSize);

    // Print the results to cout and save it as a CSV file
    auto_tuner.printResult(compositionId, cout, ktt::PrintFormat::Verbose);
    auto_tuner.printResult(compositionId, "scan-results.csv", ktt::PrintFormat::CSV);

    return 0;
}