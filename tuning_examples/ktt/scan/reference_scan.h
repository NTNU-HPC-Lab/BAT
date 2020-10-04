#pragma once

#include "tuner_api.h"

using namespace std;

class ReferenceScan : public ktt::ReferenceClass {
public:
    ReferenceScan(const vector<float> &input) :
        inputf(input) {
        single = true;
    }

    ReferenceScan(const vector<double> &input) :
        inputd(input)
    {}

    void computeResult() override {
        // Scan the input data
        if (single) {
            inputf = scanCPU(inputf.data(), inputf.size());
        } else {
            inputd = scanCPU(inputd.data(), inputd.size());
        }
    }

    void* getData(const ktt::ArgumentId argumentId) override {
        if (single) {
            return inputf.data();
        } else {
            return inputd.data();
        }
    }

    size_t getNumberOfElements(const ktt::ArgumentId id) const override {
        if (single) {
            return inputf.size();
        } else {
            return inputd.size();
        }
    }

private:
    vector<float> inputf;
    vector<double> inputd;
    bool single = false;

    template<class type>
    vector<type> scanCPU(type* data, const size_t size) {
        type last = 0.0f;
        vector<type> reference(size);

        for (unsigned int i = 0; i < size; ++i)
        {
            reference[i] = data[i] + last;
            last = reference[i];
        }

        return reference;
    }
};
