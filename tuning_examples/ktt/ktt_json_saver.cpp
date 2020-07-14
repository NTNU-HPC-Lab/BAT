#include <fstream>
#include "ktt_json_saver.hpp"

void saveJSONFileFromKTTResults(const ktt::ComputationResult &computationResult, const string &fileName) {
    string jsonOutput = "{";

    auto bestParameters = computationResult.getConfiguration();

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