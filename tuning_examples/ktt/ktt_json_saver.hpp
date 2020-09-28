#ifndef _KTT_JSON_SAVER_
#define _KTT_JSON_SAVER_
#include "tuner_api.h" // KTT API

using namespace std;

void saveJSONFileFromKTTResults(const ktt::ComputationResult &computationResult, const string &fileName, const int &problemSize);

#endif