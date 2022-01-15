#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include <stdexcept>
#include <iostream>

namespace gen {
  void ScaleWeightGroupInfo::copy(const ScaleWeightGroupInfo& other) {
    muIndices_ = other.muIndices_;
    dynVec_ = other.dynVec_;
    dynNames_ = other.dynNames_;
    WeightGroupInfo::copy(other);
  }

  ScaleWeightGroupInfo* ScaleWeightGroupInfo::clone() const { return new ScaleWeightGroupInfo(*this); }

  void ScaleWeightGroupInfo::addContainedId(int globalIndex, std::string id, std::string label, float muR, float muF) {
    int idxDiff = firstId_ - globalIndex;
    if (idxDiff > 0) {
      for (auto& entry : muIndices_) {
        entry += idxDiff;
      }
      for (auto& subVec : dynVec_) {
        for (auto& entry : subVec) {
          entry += idxDiff;
        }
      }
    }
    WeightGroupInfo::addContainedId(globalIndex, id, label);
    setMuRMuFIndex(globalIndex, id, muR, muF);
  }

  void ScaleWeightGroupInfo::setMuRMuFIndex(
      int globalIndex, std::string id, float muR, float muF, size_t dynNum, std::string dynName) {
    auto metaInfo = weightMetaInfoByGlobalIndex(id, globalIndex);
    if ((int)dynNum == -1)
      setMuRMuFIndex(metaInfo, muR, muF);
    else
      setMuRMuFIndex(metaInfo, muR, muF, dynNum, dynName);
  }

  void ScaleWeightGroupInfo::setMuRMuFIndex(WeightMetaInfo& info, float muR, float muF) {
    int index = getIndex(muR, muF);
    if (index < 0 || index > 8 || !(isValidValue(muR) && isValidValue(muF))) {
      isWellFormed_ = false;
      return;
    }
    if (index == 4)
      containsCentral_ = true;
    muIndices_[index] = info.localIndex;
  }

  void ScaleWeightGroupInfo::setMuRMuFIndex(
      WeightMetaInfo& info, float muR, float muF, size_t dynNum, std::string dynName) {
    int index = getIndex(muR, muF);
    if (index < 0 || index > 8 || !(isValidValue(muR) && isValidValue(muF))) {
      isWellFormed_ = false;
      return;
    }
    // resize if too small
    if (dynVec_.at(index).size() < dynNum + 1) {
      for (auto& dynIt : dynVec_)
        dynIt.resize(dynNum + 1);
      dynNames_.resize(dynNum + 1);
    }

    if (dynNames_.at(dynNum).empty())
      dynNames_[dynNum] = dynName;
    dynVec_[index][dynNum] = info.localIndex;
  }

  size_t ScaleWeightGroupInfo::getScaleIndex(float muR, float muF, std::string& dynName) const {
    auto it = std::find(dynNames_.begin(), dynNames_.end(), dynName);
    if (it == dynNames_.end())
      return -1;
    else
      return getScaleIndex(muR, muF, it - dynNames_.begin());
  }
  size_t ScaleWeightGroupInfo::getScaleIndex(int index, std::string& dynName) const {
    auto it = std::find(dynNames_.begin(), dynNames_.end(), dynName);
    if (it == dynNames_.end())
      return -1;
    else
      return getScaleIndex(index, it - dynNames_.begin());
  }

  size_t ScaleWeightGroupInfo::getScaleIndex(float muR, float muF, size_t dynNum) const {
    int index = getIndex(muR, muF);
    if (index < 0 || index > 8 || !(isValidValue(muR) && isValidValue(muF)) || dynNum + 1 > dynNames_.size()) {
      // Bad access!
      return -1;
    }
    return getScaleIndex(index, dynNum);
  }
  size_t ScaleWeightGroupInfo::getScaleIndex(float muR, float muF) const {
    int index = getIndex(muR, muF);
    if (index < 0 || index > 8 || !(isValidValue(muR) && isValidValue(muF))) {
      // Bad access!
      return -1;
    }
    return muIndices_.at(index);
  }

  std::vector<std::string> ScaleWeightGroupInfo::getDynNames() const {
    std::vector<std::string> returnVec;
    for (auto item : dynNames_) {
      if (!item.empty())
        returnVec.push_back(item);
    }
    return returnVec;
  }

}  // namespace gen
