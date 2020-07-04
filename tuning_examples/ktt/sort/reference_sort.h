#pragma once

#include "tuner_api.h"

using namespace std;

class ReferenceSort : public ktt::ReferenceClass {
public:
    ReferenceSort(const vector<uint>& input) :
        input(input)
    {}

    void computeResult() override {
        // Sort the input data
        sort(input.begin(), input.end());

        // TEST:
        // for (int i = 0; i < input.size(); i++) {
        //     cout << input[i] << ' ' << endl;
        // }
    }

    void* getData(const ktt::ArgumentId argumentId) override {
        return input.data();
    }

    size_t getNumberOfElements(const ktt::ArgumentId id) const override {
        return input.size();
    }

private:
    vector<uint> input;
};