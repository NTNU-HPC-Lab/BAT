#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver

using namespace std;

int main(int argc, char* argv[]) {
    // Problem sizes used in the SHOC benchmark
    uint problemSizes[4] = { 1, 8, 32, 64 };
    uint inputProblemSize = 1; // Default to the first problem size if no input

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

    string kernelFile = "../../../src/kernels/reduction/reduction_kernel_helper.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "reduction" kernel
    string kernelName("reduce_helper");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    size_t globalWorkSize = problemSizes[inputProblemSize - 1];

    int size = problemSizes[inputProblemSize - 1];
    size = (size * 1024 * 1024) / sizeof(float);

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {globalWorkSize}, {1});

    // Add parameter to tune
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", {1, 2, 4, 8, 16, 64, 128, 256, 512, 1024});
    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});

    auto_tuner.AddParameter(kernel_id, "GRID_SIZE", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
    // To set the different grid sizes (global size) multiplied by the base
    auto_tuner.MulGlobalSize(kernel_id, {"GRID_SIZE"});

    auto_tuner.AddParameter(kernel_id, "PRECISION", {32, 64});
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_REDUCE_1", {0, 1});
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_REDUCE_2", {0, 1});
    auto_tuner.AddParameter(kernel_id, "TEXTURE_MEMORY", {0}); // CLTune does not support texture memory, so this is always disabled

    // Fallback "pseudo-parameter" to set shared memory size in the kernel rather than in the host code
    // This is not an actual parameter, but rather a way of specifying that the kernel should set the size of the shared memory
    // It sets the size equal to the block size
    // This is needed because there is no way to set the shared memory from CLTune
    auto_tuner.AddParameter(kernel_id, "KERNEL_SHARED_MEMORY_SIZE", {1});

    // Set reference kernel for correctness verification and compare to the computed result
    // NOTE: Due to not being able to specify the precision to match the tuned kernel, this has only been used when developing and testing this benchmark
    // Remember to set the correct precision in the reference kernel when using the reference kernel
    // auto_tuner.SetReference({referenceKernelFile}, kernelName, {globalWorkSize}, {1});

    vector<float> idataf(size);
    vector<float> odataf(size);
    vector<double> idatad(size);
    vector<double> odatad(size);

    // Initialize start values for input
    for(int i = 0; i < size; i++)
    {
        // Fill with same pattern as in the SHOC benchmark
        idatad[i] = idataf[i] = i % 3;
    }

    // Add arguments for kernel
    auto_tuner.AddArgumentInput(idataf);
    auto_tuner.AddArgumentScalar(0); // CLTune does not support texture memory, so this is always disabled
    auto_tuner.AddArgumentOutput(odataf);
    auto_tuner.AddArgumentInput(idatad);
    auto_tuner.AddArgumentScalar(0); // CLTune does not support texture memory, so this is always disabled
    auto_tuner.AddArgumentOutput(odatad);
    auto_tuner.AddArgumentScalar(size);

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json", inputProblemSize);

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
    
    return 0;
}
