#include <iostream> // For terminal output
#include <fstream> // For file saving
#include "tuner_api.h" // KTT API
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver

using namespace std;

int main(int argc, char* argv[]) {
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/reduction/reduction_kernel_helper.cu";
    string referenceKernelFile = "./reference_kernel.cu";
    string kernelName("reduce_helper");
    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA?
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

    // Problem sizes from SHOC
    uint problemSizes[4] = { 1, 8, 32, 64 };
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

    int size = problemSizes[inputProblemSize - 1];
    int sizeFloat = (size * 1024 * 1024) / sizeof(float);
    int sizeDouble = (size * 1024 * 1024) / sizeof(double);

    const ktt::DimensionVector gridSize;
    const ktt::DimensionVector blockSize;

    const ktt::DimensionVector gridSizeReference(64*256);
    const ktt::DimensionVector blockSizeReference(256);

    // Add kernel and reference kernel
    ktt::KernelId kernelId = auto_tuner.addKernelFromFile(kernelFile, kernelName, gridSize, blockSize);
    ktt::KernelId referenceKernelId = auto_tuner.addKernelFromFile(referenceKernelFile, kernelName, gridSizeReference, blockSizeReference);

    vector<float> idataf(sizeFloat);
    vector<float> odataf(sizeFloat);
    vector<double> idatad(sizeDouble);
    vector<double> odatad(sizeDouble);

    // Initialize start values for input
    for(int i = 0; i < sizeFloat; i++)
    {
        // Fill with same pattern as in the SHOC benchmark
        idataf[i] = i % 3;
    }
    for(int i = 0; i < sizeDouble; i++)
    {
        // Fill with same pattern as in the SHOC benchmark
        idatad[i] = i % 3;
    }

    // Add arguments for kernel
    ktt::ArgumentId idatafId = auto_tuner.addArgumentVector(idataf, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId idataTextureObjectfId = auto_tuner.addArgumentScalar(0); // KTT does not support texture memory, so this is always disabled
    ktt::ArgumentId odatafId = auto_tuner.addArgumentVector(odataf, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId idatadId = auto_tuner.addArgumentVector(idatad, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId idataTextureObjectdId = auto_tuner.addArgumentScalar(0); // KTT does not support texture memory, so this is always disabled
    ktt::ArgumentId odatadId = auto_tuner.addArgumentVector(odatad, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId sizefId = auto_tuner.addArgumentScalar(sizeFloat);
    ktt::ArgumentId sizedId = auto_tuner.addArgumentScalar(sizeDouble);

    // Add arguments to kernel and reference kernel
    auto_tuner.setKernelArguments(kernelId, vector<ktt::ArgumentId>{idatafId, idataTextureObjectfId, odatafId, idatadId, idataTextureObjectdId, odatadId, sizefId, sizedId});
    auto_tuner.setKernelArguments(referenceKernelId, vector<ktt::ArgumentId>{idatafId, idataTextureObjectfId, odatafId, idatadId, idataTextureObjectdId, odatadId, sizefId, sizedId});

    // Add parameters to tune
    auto_tuner.addParameter(kernelId, "BLOCK_SIZE", {1, 2, 4, 8, 16, 64, 128, 256, 512, 1024});
    auto_tuner.addParameter(kernelId, "GRID_SIZE", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
    auto_tuner.addParameter(kernelId, "PRECISION", {32, 64});
    auto_tuner.addParameter(kernelId, "LOOP_UNROLL_REDUCE_1", {0, 1});
    auto_tuner.addParameter(kernelId, "LOOP_UNROLL_REDUCE_2", {0, 1});
    auto_tuner.addParameter(kernelId, "TEXTURE_MEMORY", {0}); // KTT does not support texture memory, so this is always disabled

    // Fallback "pseudo-parameter" to set shared memory size in the kernel rather than in the host code
    // This is not an actual parameter, but rather a way of specifying that the kernel should set the size of the shared memory
    // It sets the size equal to the block size
    // This is needed because there is no way to set the shared memory from KTT
    auto_tuner.addParameter(kernelId, "KERNEL_SHARED_MEMORY_SIZE", {1});

    auto globalModifier = [](const size_t size, const std::vector<size_t>& vector) {return size * vector.at(0) * vector.at(1);};

    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE", ktt::ModifierAction::Multiply);
    // To set the different grid sizes (global size) and block sizes (local size) multiplied by the grid size and block size parameters
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, vector<string>{"GRID_SIZE", "BLOCK_SIZE"}, globalModifier);

    // Set reference kernel for correctness verification and compare to the computed result
    // NOTE: Due to not being able to specify the precision to match the tuned kernel, this has only been used when developing and testing this benchmark
    // Remember to set the correct precision in the reference kernel when using the reference kernel
    // auto_tuner.setReferenceKernel(kernelId, referenceKernelId, vector<ktt::ParameterPair>{}, vector<ktt::ArgumentId>{odatafId, odatadId});

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