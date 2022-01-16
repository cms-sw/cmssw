#ifndef SimDataFormats_GeneratorProducts_GenWeightProduct_h
#define SimDataFormats_GeneratorProducts_GenWeightProduct_h

#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightsInfo.h"
#include "FWCore/Utilities/interface/Exception.h"

typedef std::vector<std::vector<double>> WeightsContainer;

class GenWeightProduct {
public:
  GenWeightProduct() {
    weightsVector_ = {};
    centralWeight_ = 1.;
  }
  GenWeightProduct(double w0) {
    weightsVector_ = {};
    centralWeight_ = w0;
  }
  GenWeightProduct& operator=(GenWeightProduct&& other) {
    weightsVector_ = std::move(other.weightsVector_);
    centralWeight_ = other.centralWeight_;
    return *this;
  }
  ~GenWeightProduct() {}

  void setNumWeightSets(int num) { weightsVector_.resize(num); }
  void addWeightSet() { weightsVector_.push_back({}); }
  void addWeight(double weight, int setEntry, int weightNum) {
    if (weightsVector_.empty() && setEntry == 0)
      addWeightSet();
    int maxSets = static_cast<int>(weightsVector_.size());
    if (maxSets <= setEntry)
      throw cms::Exception("GenWeightProduct")
          << "WeightGroup index " << setEntry << " is exceeds the number of WeightGroups expected (max " << maxSets
          << " )";
    auto& weights = weightsVector_[setEntry];
    if (static_cast<int>(weights.size()) <= weightNum) {
      weights.resize(weightNum + 1);
    }
    weights[weightNum] = weight / centralWeight_;
  }
  const WeightsContainer& weights() const { return weightsVector_; }
  double centralWeight() const { return centralWeight_; }

private:
  WeightsContainer weightsVector_;
  double centralWeight_;
};

#endif  // GeneratorEvent_LHEInterface_GenWeightProduct_h
