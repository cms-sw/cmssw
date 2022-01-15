#include "SimDataFormats/GeneratorProducts/interface/PartonShowerWeightGroupInfo.h"

namespace gen {
  void PartonShowerWeightGroupInfo::copy(const PartonShowerWeightGroupInfo& other) { WeightGroupInfo::copy(other); }

  PartonShowerWeightGroupInfo* PartonShowerWeightGroupInfo::clone() const {
    return new PartonShowerWeightGroupInfo(*this);
  }

  void PartonShowerWeightGroupInfo::updateWeight(int globalIndex, std::string id, std::string subName, bool isUp) {
    size_t localIndex = weightMetaInfoByGlobalIndex(id, globalIndex).localIndex;
    if (weightNameToUpDown.find(subName) == weightNameToUpDown.end()) {
      weightNames.push_back(subName);
      weightNameToUpDown[subName] = std::pair<size_t, size_t>();
    }
    if (isUp)
      weightNameToUpDown[subName].first = localIndex;
    else
      weightNameToUpDown[subName].second = localIndex;
  }

}  // namespace gen
