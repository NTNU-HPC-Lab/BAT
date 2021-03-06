#include <iostream>
#include <cuda_runtime_api.h>
#include "tuner_api.h" // KTT API
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver

using namespace std;

int main(int argc, char* argv[]) {
    // Tune "triad" kernel
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/triad/triad_kernel_helper.cu";
    string referenceKernelFile = "./reference_kernel.cu";
    string kernelName("triad_helper");
    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA?
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

    // Problem sizes from SHOC
    uint problemSizes[9] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    uint inputProblemSize = 9; // Default to the last problem size if no input

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

                // Ensure the input problem size is between 1 and 9
                if (inputProblemSize < 1 || inputProblemSize > 9) {
                    cerr << "Error: The problem size needs to be an integer in the range 1 to 9." << endl;
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

    size_t globalWorkSize = problemSizes[inputProblemSize - 1] * 1024;

    const ktt::DimensionVector gridSize(globalWorkSize);
    const ktt::DimensionVector blockSize;

    // Add kernel and reference kernel
    ktt::KernelId kernelId = auto_tuner.addKernelFromFile(kernelFile, kernelName, gridSize, blockSize);
    ktt::KernelId referenceKernelId = auto_tuner.addKernelFromFile(referenceKernelFile, kernelName, gridSize, blockSize);

    const uint size = problemSizes[inputProblemSize - 1];
    
    // Use the same seed for random number as in the SHOC benchmark
    srand48(8650341L);

    const size_t numMaxFloats = 1024 * size / 4;
    const size_t halfNumFloats = numMaxFloats / 2;

    vector<float> Af(numMaxFloats);
    vector<float> Bf(numMaxFloats);
    vector<float> Cf(numMaxFloats);
    vector<double> Ad(numMaxFloats);
    vector<double> Bd(numMaxFloats);
    vector<double> Cd(numMaxFloats);

    // Initialize start values for input
    for (size_t i = 0; i < halfNumFloats; i++) {
        double currentRandomNumber = drand48() * 10.0;
        Af[i] = Bf[i] = Af[halfNumFloats + i] = Bf[halfNumFloats + i] = (float) currentRandomNumber;
        Ad[i] = Bd[i] = Ad[halfNumFloats + i] = Bd[halfNumFloats + i] = currentRandomNumber;
    }

    // Add arguments for kernel
    // Arrays (A, B, C) have random numbers similar to the program. The numbers are in the range [0, 10)
    // <x>f are floats and <x>d are double values
    ktt::ArgumentId afId = auto_tuner.addArgumentVector(Af, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId bfId = auto_tuner.addArgumentVector(Bf, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId cfId = auto_tuner.addArgumentVector(Cf, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId sfId = auto_tuner.addArgumentScalar(1.75); // scalar
    ktt::ArgumentId adId = auto_tuner.addArgumentVector(Ad, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId bdId = auto_tuner.addArgumentVector(Bd, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId cdId = auto_tuner.addArgumentVector(Cd, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId sdId = auto_tuner.addArgumentScalar(1.75); // scalar
    ktt::ArgumentId globalWorkSizeId = auto_tuner.addArgumentScalar(globalWorkSize); // The number of elements

    // Add arguments to kernel and reference kernel
    auto_tuner.setKernelArguments(kernelId, vector<ktt::ArgumentId>{afId, bfId, cfId, sfId, adId, bdId, cdId, sdId, globalWorkSizeId});
    auto_tuner.setKernelArguments(referenceKernelId, vector<ktt::ArgumentId>{afId, bfId, cfId, sfId, adId, bdId, cdId, sdId, globalWorkSizeId});

    // Get CUDA properties from device 0 
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);

    vector<size_t> block_sizes;

    // Initialize the block sizes array
    for (size_t i = 0; i < properties.maxThreadsPerBlock; i++) {
        block_sizes.push_back(i + 1);
    }

    // Add parameters to tune
    auto_tuner.addParameter(kernelId, "BLOCK_SIZE", block_sizes);
    auto_tuner.addParameter(kernelId, "WORK_PER_THREAD", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto_tuner.addParameter(kernelId, "LOOP_UNROLL_TRIAD", {0, 1});
    auto_tuner.addParameter(kernelId, "PRECISION", {32, 64});

    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE", ktt::ModifierAction::Multiply);
    // To set the different grid sizes (global size) divided by the amount of work per thread
    // Divide on block size and multiply by block size after ceiling to ensure enough threads used
    // Also use precision to get the grid size
    // Using ceil because KTT does not ceil the divided grid size
    auto globalModifier = [](const size_t size, const vector<size_t>& vector) {
        return int(ceil(double(size) / double(vector.at(0)) / double(vector.at(1)))) * vector.at(0) * (vector.at(2)/32*4);
    };
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"BLOCK_SIZE", "WORK_PER_THREAD", "PRECISION"}, globalModifier);

    // Set reference kernel for correctness verification and compare to the computed result
    // NOTE: Due to not being able to specify the precision to match the tuned kernel, this has only been used when developing and testing this benchmark
    // Remember to set the correct precision in the reference kernel when using the reference kernel
    // auto_tuner.setReferenceKernel(kernelId, referenceKernelId, vector<ktt::ParameterPair>{}, vector<ktt::ArgumentId>{cfId, cdId});

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