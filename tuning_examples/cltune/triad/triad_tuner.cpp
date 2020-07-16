#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver

using namespace std;

int main(int argc, char* argv[]) {
    // Problem sizes used in the SHOC benchmark
    uint problemSizes[9] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    uint inputProblemSize = 8; // Default to the last problem size if no input

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

    string kernelFile = "../../../src/kernels/triad/triad_kernel.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "triad" kernel
    string kernelName("triad");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    size_t globalWorkSize = problemSizes[inputProblemSize - 1] * 1024 / sizeof(float);

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {globalWorkSize}, {1});

    // Add parameter to tune
    // TODO: fix program code to be usable for all sizes
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {globalWorkSize}, {1});

    // Use the same seed for random as in the SHOC benchmark
    srand48(8650341L);

    const size_t numMaxFloats = 1024 * problemSizes[inputProblemSize] / 4;
    const size_t halfNumFloats = numMaxFloats / 2;

    vector<float> A(numMaxFloats);
    vector<float> B(numMaxFloats);
    vector<float> C(numMaxFloats);

    // Initialize start values for input
    for (size_t i = 0; i < halfNumFloats; i++) {
        float currentRandomNumber = (float) (drand48() * 10.0);
        A[i] = B[i] = C[i] = A[halfNumFloats + i] = B[halfNumFloats + i] = C[halfNumFloats + i] = currentRandomNumber;
    }

    // Add arguments for kernel
    // Arrays (A, B, C) have random numbers similar to the program. The numbers are in the range [0, 10)
    auto_tuner.AddArgumentInput(A);
    auto_tuner.AddArgumentInput(B);
    auto_tuner.AddArgumentOutput(C);
    auto_tuner.AddArgumentScalar(1.75); // scalar

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json");

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
    
    return 0;
}
