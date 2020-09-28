#include <fstream>
#include "ktt_json_saver.hpp"

void saveJSONFileFromKTTResults(const ktt::ComputationResult &computationResult, const string &fileName, const int &problemSize) {
    string jsonOutput = "{\n\t\"PROBLEM_SIZE\": " + to_string(problemSize);

    auto bestParameters = computationResult.getConfiguration();

    // Add comma if parameters
    if (computationResult.getConfiguration().size() > 0) {
        jsonOutput += ",";
    }

    // Loop all parameters and add one by one
    for (auto const& parameter: bestParameters) {
        jsonOutput += "\n\t\"" + parameter.getName() + "\": " + to_string(parameter.getValue());
        
        // If not the last item in the vector add a comma (",")
        if (&parameter != &bestParameters.back()) {
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