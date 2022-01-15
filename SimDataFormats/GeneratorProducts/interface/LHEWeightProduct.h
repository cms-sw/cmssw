#ifndef SimDataFormats_GeneratorProducts_LHEWeightProduct_h
#define SimDataFormats_GeneratorProducts_LHEWeightProduct_h

#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightsInfo.h"

typedef std::vector<std::vector<double>> WeightsContainer;

class LHEWeightProduct {
    public:
        LHEWeightProduct() { weightsVector_ = {}; }
        LHEWeightProduct& operator=(LHEWeightProduct&& other) {
        weightsVector_ = std::move(other.weightsVector_);
        return *this;
        }
        ~LHEWeightProduct() {}

        void setNumWeightSets(int num) { weightsVector_.resize(num); }
        void addWeightSet() { weightsVector_.push_back({}); }
        void addWeight(double weight, int setEntry, int weightNum) { 	  
            if (weightsVector_.size() == 0 && setEntry == 0)
                addWeightSet();
            if (static_cast<int>(weightsVector_.size()) <= setEntry)
                throw std::domain_error("Out of range weight");
            auto& weights = weightsVector_.at(setEntry);
            if (static_cast<int>(weights.size()) <= weightNum) {
                weights.resize(weightNum+1);
            }
            weights[weightNum] = weight;
        }
        const WeightsContainer& weights() const { return weightsVector_; }

    private:
        WeightsContainer weightsVector_;
};

#endif // GeneratorEvent_LHEInterface_LHEWeightProduct_h

