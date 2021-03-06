#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver
#include "search_constants.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Problem sizes used in the SHOC benchmark
    uint problemSizes[4] = { 1, 8, 32, 64 };
    uint inputProblemSize = 1; // Default to the first problem size if no input

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

    string kernelFile = "../../../src/kernels/reduction/reduction_kernel_helper.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "reduction" kernel
    string kernelName("reduce_helper");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    //size_t globalWorkSize = problemSizes[inputProblemSize - 1];

    int size = problemSizes[inputProblemSize - 1];
    int sizeFloat = (size * 1024 * 1024) / sizeof(float);
    int sizeDouble = (size * 1024 * 1024) / sizeof(double);

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {1}, {1});

    // Add parameter to tune
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", {1, 2, 4, 8, 16, 64, 128, 256, 512, 1024});
    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});

    auto_tuner.AddParameter(kernel_id, "GRID_SIZE", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
    // To set the different grid sizes (global size) multiplied by the base (1)
    auto_tuner.MulGlobalSize(kernel_id, {"GRID_SIZE", "BLOCK_SIZE"});

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
    // auto_tuner.SetReference({referenceKernelFile}, kernelName, {64}, {256});
    // auto_tuner.AddParameterReference("PRECISION", 64);

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
    auto_tuner.AddArgumentInput(idataf);
    auto_tuner.AddArgumentScalar(0); // CLTune does not support texture memory, so this is always disabled
    auto_tuner.AddArgumentOutput(odataf);
    auto_tuner.AddArgumentInput(idatad);
    auto_tuner.AddArgumentScalar(0); // CLTune does not support texture memory, so this is always disabled
    auto_tuner.AddArgumentOutput(odatad);
    auto_tuner.AddArgumentScalar(sizeFloat);
    auto_tuner.AddArgumentScalar(sizeDouble);

    // Use a fraction of the total search space
    double searchFraction = SEARCH_FRACTION*2;

    // Select the tuning technique for this benchmark
    if (tuningTechnique == "annealing") {
        double maxTemperature = MAX_TEMPERATURE;
        auto_tuner.UseAnnealing(searchFraction, {maxTemperature});
    } else if (tuningTechnique == "pso") {
        double swarmSize = SWARM_SIZE;
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
