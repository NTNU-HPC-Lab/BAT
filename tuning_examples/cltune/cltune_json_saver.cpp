#include <fstream>
#include <unordered_map>
#include "cltune_json_saver.hpp"

void saveJSONFileFromCLTuneResults(const unordered_map<string, size_t> &computationResult, const string &fileName, const int &problemSize, const string &tuningTechnique) {
    string jsonOutput = "{\n\t\"PROBLEM_SIZE\": " + to_string(problemSize) + ",\n\t\"TUNING_TECHNIQUE\": \"" + tuningTechnique + "\"";

    // Counters to check if the last item in the map
    int counter = 0;
    int maxCount = computationResult.size();

    // Add comma if parameters
    if (maxCount > 0) {
        jsonOutput += ",";
    }

    // Loop all parameters and add one by one
    for (auto const& parameter: computationResult) {
        jsonOutput += "\n\t\"" + parameter.first + "\": " + to_string(parameter.second);

        counter++;

        // If not the last item in the vector add a comma (",")
        if (counter != maxCount) {
            jsonOutput += ",";
        }
    }

    jsonOutput += "\n}";

    // Save the JSON to file
    fstream fs;
    fs.open(fileName, fstream::out);
    fs << jsonOutput;
    fs.close();
}