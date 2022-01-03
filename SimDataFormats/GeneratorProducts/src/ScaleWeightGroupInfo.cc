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

  void ScaleWeightGroupInfo::setMuRMuFIndex(int globalIndex, std::string id, float muR, float muF) {
    auto info = weightMetaInfoByGlobalIndex(id, globalIndex);
    int index = indexFromMus(muR, muF);
    if (!(isValidValue(muR) && isValidValue(muF))) {
      setWeightIsCorrupt();
      return;
    }
    if (index == Central_idx)
      containsCentral_ = true;
    muIndices_[index] = info.localIndex;

    for (int muidx : muIndices_) {
      if (muidx == -1)
        return;
    }
    isWellFormed_ = !weightIsCorrupt_;
  }

  void ScaleWeightGroupInfo::setDyn(
      int globalIndex, std::string id, float muR, float muF, size_t dynNum, std::string_view dynName) {
    auto info = weightMetaInfoByGlobalIndex(id, globalIndex);
    int index = indexFromMus(muR, muF);
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

  size_t ScaleWeightGroupInfo::scaleIndex(float muR, float muF, std::string_view dynName) const {
    auto it = std::find(dynNames_.begin(), dynNames_.end(), dynName);
    if (it == dynNames_.end())
      return -1;
    else
      return scaleIndex(muR, muF, it - dynNames_.begin());
  }

  size_t ScaleWeightGroupInfo::scaleIndex(int index, std::string_view dynName) const {
    auto it = std::find(dynNames_.begin(), dynNames_.end(), dynName);
    if (it == dynNames_.end())
      return -1;
    else
      return scaleIndex(index, it - dynNames_.begin());
  }

  size_t ScaleWeightGroupInfo::scaleIndex(float muR, float muF, size_t dynNum) const {
    ;
    if (!(isValidValue(muR) && isValidValue(muF)) || dynNum + 1 > dynNames_.size())
      return -1;
    else
      return scaleIndex(indexFromMus(muR, muF), dynNum);
  }
  size_t ScaleWeightGroupInfo::scaleIndex(float muR, float muF) const {
    if (!(isValidValue(muR) && isValidValue(muF)))
      return -1;
    else
      return muIndices_.at(indexFromMus(muR, muF));
  }

  std::vector<std::string> ScaleWeightGroupInfo::dynNames() const {
    std::vector<std::string> returnVec;
    for (const auto& item : dynNames_) {
      if (!item.empty())
        returnVec.push_back(item);
    }
    return returnVec;
  }

}  // namespace gen
