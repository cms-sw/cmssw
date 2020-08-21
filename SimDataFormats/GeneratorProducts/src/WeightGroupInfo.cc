#include <string>
#include <algorithm>
#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

namespace gen {
  void WeightGroupInfo::copy(const WeightGroupInfo& other) {
    isWellFormed_ = other.isWellFormed_;
    headerEntry_ = other.headerEntry_;
    name_ = other.name_;
    description_ = other.description_;
    weightType_ = other.weightType_;
    idsContained_ = other.idsContained_;
    firstId_ = other.firstId_;
    lastId_ = other.lastId_;
  }

  WeightGroupInfo* WeightGroupInfo::clone() const {
    throw cms::Exception("LogicError", "WeightGroupInfo is abstract, so it's clone() method can't be implemented.\n");
  }

  WeightMetaInfo WeightGroupInfo::weightMetaInfo(int weightEntry) const { return idsContained_.at(weightEntry); }

  WeightMetaInfo WeightGroupInfo::weightMetaInfoByGlobalIndex(std::string wgtId, int weightEntry) const {
   if (wgtId.empty())
       wgtId = std::to_string(weightEntry);
    int entry = weightVectorEntry(wgtId, weightEntry);
    if (entry < 0 || entry >= static_cast<int>(idsContained_.size()))
      throw std::range_error("Weight entry " + std::to_string(weightEntry) + " is not a member of group " + name_ +
                             ". \n    firstID = " + std::to_string(firstId_) + " lastId = " + std::to_string(lastId_));
    return idsContained_.at(entry);
  }

  int WeightGroupInfo::weightVectorEntry(std::string& wgtId) const { return weightVectorEntry(wgtId, 0); }

  bool WeightGroupInfo::containsWeight(std::string& wgtId, int weightEntry) const {
    return weightVectorEntry(wgtId, weightEntry) != -1;
  }

  int WeightGroupInfo::weightVectorEntry(std::string& wgtId, int weightEntry) const {
    if (wgtId.empty())
      wgtId = std::to_string(weightEntry);
    // First try ordered search
    size_t orderedEntry = weightEntry - firstId_;
    if (indexInRange(weightEntry) && orderedEntry < idsContained_.size()) {
      if (wgtId.empty() || idsContained_.at(orderedEntry).id == wgtId) {
        return orderedEntry;
      }
    }
    // Fall back to search on ID
    else if (!wgtId.empty()) {
      auto it = std::find_if(
          idsContained_.begin(), idsContained_.end(), [wgtId](const WeightMetaInfo& w) { return w.id == wgtId; });
      if (it != idsContained_.end())
        return std::distance(idsContained_.begin(), it);
    }
    return -1;
  }

  void WeightGroupInfo::addContainedId(int weightEntry, std::string id, std::string label = "") {
   if (id.empty())
       id = std::to_string(weightEntry);

    if (firstId_ == -1 || weightEntry < firstId_) {
      firstId_ = weightEntry;
      for (auto& entry : idsContained_)  // Reset if indices need to be shifted
        entry.localIndex++;
    }
    if (weightEntry > lastId_)
      lastId_ = weightEntry;

    size_t localIndex = std::min(weightEntry - firstId_, static_cast<int>(idsContained_.size()));
    WeightMetaInfo info = {static_cast<size_t>(weightEntry), localIndex, id, label};
    // logic to insert for all cases e.g. inserting in the middle of the vector
    if (localIndex == idsContained_.size())
        idsContained_.emplace_back(info);
    else
        idsContained_.insert(idsContained_.begin() + localIndex, info);
  }

  std::vector<WeightMetaInfo> WeightGroupInfo::containedIds() const { return idsContained_; }

  bool WeightGroupInfo::indexInRange(int index) const { return (index <= lastId_ && index >= firstId_); }

  void WeightGroupInfo::cacheWeightIndicesByLabel() {
      for (const auto& weight : idsContained_)
        weightLabelsToIndices_[weight.label] = weight.localIndex;
  }

  std::vector<std::string> WeightGroupInfo::weightLabels() const {
    std::vector<std::string> labels;
    labels.reserve(idsContained_.size());
    for (const auto& weight : idsContained_)
      labels.push_back(weight.label);
    return labels;
  }

  int WeightGroupInfo::weightIndexFromLabel(std::string weightLabel) const {
      if (!weightLabelsToIndices_.empty()) {
        if (weightLabelsToIndices_.find(weightLabel) != weightLabelsToIndices_.end())
            return static_cast<int>(weightLabelsToIndices_.at(weightLabel));
        return -1;
      }
    
      auto it = std::find_if(idsContained_.begin(), idsContained_.end(), 
              [weightLabel](const auto& w) { return weightLabel == w.label; });
      if (it == idsContained_.end())
          return -1;
      return std::distance(idsContained_.begin(), it);
  }

}  // namespace gen
