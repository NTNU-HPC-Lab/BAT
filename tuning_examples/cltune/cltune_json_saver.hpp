#ifndef _CLTUNE_JSON_SAVER_
#define _CLTUNE_JSON_SAVER_

using namespace std;

void saveJSONFileFromCLTuneResults(const unordered_map<string, size_t> &computationResult, const string &fileName, const int &problemSize);

#endif