#include "SimDataFormats/GeneratorProducts/interface/MEParamWeightGroupInfo.h"
#include <iostream>

namespace gen {
  void MEParamWeightGroupInfo::copy(const MEParamWeightGroupInfo& other) { WeightGroupInfo::copy(other); }

  MEParamWeightGroupInfo* MEParamWeightGroupInfo::clone() const { return new MEParamWeightGroupInfo(*this); }

  void MEParamWeightGroupInfo::updateWeight(int globalIndex, std::string id, double weight) {
    size_t localIndex = weightMetaInfoByGlobalIndex(id, globalIndex).localIndex;
    auto lower =
        std::lower_bound(massValue.begin(), massValue.end(), std::make_pair(weight, std::numeric_limits<int>::min()));
    massValue.insert(lower, std::make_pair(weight, localIndex));
    isWellFormed_ = massValue.size() % 2 == 1;
    if (isWellFormed_) {
      numSigma = massValue.size() / 2;
      central = massValue.at(centralIdx).first;
      centralIdx = massValue.at(centralIdx).second;
    }
  }

  void MEParamWeightGroupInfo::updateWeight(int globalIndex, std::string id, std::vector<std::string>& content) {
    size_t localIndex = weightMetaInfoByGlobalIndex(id, globalIndex).localIndex;
    splitContent[localIndex] = content;
  }

}  // namespace gen
