#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <cuda_runtime_api.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver

using namespace std;

int main(int argc, char* argv[]) {
    // Problem sizes used in the SHOC benchmark
    uint problemSizes[9] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    uint inputProblemSize = 9; // Default to the last problem size if no input

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

    string kernelFile = "../../../src/kernels/triad/triad_kernel_helper.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "triad" kernel
    string kernelName("triad_helper");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    size_t globalWorkSize = problemSizes[inputProblemSize - 1] * 1024 / sizeof(float);

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {globalWorkSize}, {1});

    // Get CUDA properties from device 0 
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
    vector<size_t> block_sizes;

    // Initialize the block sizes array
    for (size_t i = 0; i < properties.maxThreadsPerBlock; i++) {
        block_sizes.push_back(i + 1);
    }

    // Add parameters to tune
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", block_sizes);
    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});

    auto_tuner.AddParameter(kernel_id, "WORK_PER_THREAD", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    // To set the different grid sizes (global size) divided by the amount of work per thread
    auto_tuner.DivGlobalSize(kernel_id, {"WORK_PER_THREAD"});

    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_TRIAD", {0, 1});
    auto_tuner.AddParameter(kernel_id, "PRECISION", {32, 64});

    // Set reference kernel for correctness verification and compare to the computed result
    // NOTE: Due to not being able to specify the precision to match the tuned kernel, this has only been used when developing and testing this benchmark
    // Remember to set the correct precision for the reference kernel below
    // auto_tuner.SetReference({referenceKernelFile}, kernelName, {globalWorkSize}, {128});
    // auto_tuner.AddParameterReference("PRECISION", 64);

    // Use the same seed for random number as in the SHOC benchmark
    srand48(8650341L);

    const size_t numMaxFloats = 1024 * problemSizes[inputProblemSize - 1] / 4;
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
    auto_tuner.AddArgumentInput(Af);
    auto_tuner.AddArgumentInput(Bf);
    auto_tuner.AddArgumentOutput(Cf);
    auto_tuner.AddArgumentScalar(1.75); // scalar
    auto_tuner.AddArgumentInput(Ad);
    auto_tuner.AddArgumentInput(Bd);
    auto_tuner.AddArgumentOutput(Cd);
    auto_tuner.AddArgumentScalar(1.75); // scalar
    auto_tuner.AddArgumentScalar(globalWorkSize); // The number of elements

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json", inputProblemSize);

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
    
    return 0;
}
