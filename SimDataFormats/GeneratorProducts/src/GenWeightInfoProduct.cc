#include <vector>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "FWCore/Utilities/interface/Exception.h"

GenWeightInfoProduct::GenWeightInfoProduct(std::vector<std::unique_ptr<gen::WeightGroupInfo>>& weightGroups) {
  // Could just do a std::move on the vector, but copying is safer if the user expects the vector to still be usable
  for (auto& ptr : weightGroups) {
    std::unique_ptr<gen::WeightGroupInfo> cloneptr(ptr->clone());
    weightGroupsInfo_.emplace_back(std::move(cloneptr));
  }
  auto it = std::find_if(std::begin(weightGroupsInfo_), std::end(weightGroupsInfo_), [](auto& entry) {
    return entry->name() == "unassociated";
  });
  if (it != std::end(weightGroupsInfo_)) {
    unassociatedIdx_ = std::distance(std::begin(weightGroupsInfo_), it);
  } else
    unassociatedIdx_ = -1;
}

const std::vector<std::unique_ptr<gen::WeightGroupInfo>>& GenWeightInfoProduct::allWeightGroupsInfo() const {
  return weightGroupsInfo_;
}

const std::vector<gen::WeightGroupData> GenWeightInfoProduct::allWeightGroupsInfoWithIndices() const {
  std::vector<gen::WeightGroupData> groupInfo;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++)
    groupInfo.push_back({i, weightGroupsInfo_[i].get()});
  return groupInfo;
}

gen::WeightGroupData GenWeightInfoProduct::containingWeightGroupInfo(int index, size_t startSearch) const {
  // When filling the weights, most likely to find the weight matches the previous group or the one after
  if (startSearch < weightGroupsInfo_.size() && weightGroupsInfo_[startSearch]->containsWeight(index))
    return {startSearch, weightGroupsInfo_[startSearch].get()};
  else if (startSearch + 1 < weightGroupsInfo_.size() && weightGroupsInfo_[startSearch + 1]->containsWeight(index))
    return {startSearch + 1, weightGroupsInfo_[startSearch + 1].get()};

  auto it = std::find_if(std::begin(weightGroupsInfo_), std::end(weightGroupsInfo_), [index](auto& entry) {
    return entry->containsWeight(index);
  });
  if (it != std::end(weightGroupsInfo_))
    return {static_cast<size_t>(std::distance(std::begin(weightGroupsInfo_), it)), it->get()};

  throw cms::Exception("GenWeightInfoProduct") << "No weight group found containing weight index " << index
                                               << " in the " << weightGroupsInfo_.size() << " groups.";
}

const gen::WeightGroupInfo* GenWeightInfoProduct::orderedWeightGroupInfo(int weightGroupIndex) const {
  if (weightGroupIndex >= static_cast<int>(weightGroupsInfo_.size()))
    throw cms::Exception("GenWeightInfoProduct")
        << "Weight index requested is outside the range of weights in the product";
  return weightGroupsInfo_[weightGroupIndex].get();
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::weightGroupsAndIndicesByType(gen::WeightType type,
                                                                                     int maxStore) const {
  std::vector<gen::WeightGroupData> matchingGroups;
  size_t toStore = maxStore <= 0 ? weightGroupsInfo_.size() : std::min<size_t>(maxStore, weightGroupsInfo_.size());
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    const gen::WeightGroupInfo* group = weightGroupsInfo_[i].get();
    if (group->weightType() == type) {
      matchingGroups.push_back({i, group});
      if (matchingGroups.size() == toStore)
        break;
    }
  }
  return matchingGroups;
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::weightGroupsByType(gen::WeightType type) const {
  std::vector<gen::WeightGroupData> matchingGroups;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    const gen::WeightGroupInfo* group = weightGroupsInfo_[i].get();
    if (group->weightType() == type)
      matchingGroups.push_back({i, group});
  }
  return matchingGroups;
}

std::optional<gen::WeightGroupData> GenWeightInfoProduct::pdfGroupWithIndexByLHAID(int lhaid) const {
  std::vector<gen::WeightGroupData> pdfGroups = weightGroupsAndIndicesByType(gen::WeightType::kPdfWeights);

  auto matchingPdfSet = std::find_if(pdfGroups.begin(), pdfGroups.end(), [lhaid](auto& data) {
    const auto* pdfGroup = static_cast<const gen::PdfWeightGroupInfo*>(data.group);
    return pdfGroup->containsLhapdfId(lhaid);
  });

  return matchingPdfSet == pdfGroups.end()
             ? std::nullopt
             : std::optional<gen::WeightGroupData>({matchingPdfSet->index, matchingPdfSet->group});
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::pdfGroupsWithIndicesByLHAIDs(
    const std::vector<int>& lhaids) const {
  auto pdfGroups = weightGroupsAndIndicesByType(gen::WeightType::kPdfWeights);
  std::vector<gen::WeightGroupData> groups;

  for (auto lhaid : lhaids) {
    auto matchingPdfSet = std::find_if(pdfGroups.begin(), pdfGroups.end(), [lhaid](gen::WeightGroupData& data) {
      const auto* pdfGroup = static_cast<const gen::PdfWeightGroupInfo*>(data.group);
      return pdfGroup->containsLhapdfId(lhaid);
    });
    if (matchingPdfSet != pdfGroups.end()) {
      pdfGroups.push_back({matchingPdfSet->index, matchingPdfSet->group});
    }
  }

  return pdfGroups;
}

std::vector<int> GenWeightInfoProduct::weightGroupIndicesByType(gen::WeightType type) const {
  std::vector<int> matchingGroupIndices;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    if (weightGroupsInfo_[i]->weightType() == type)
      matchingGroupIndices.push_back(i);
  }
  return matchingGroupIndices;
}

void GenWeightInfoProduct::addWeightGroupInfo(gen::WeightGroupInfo info) {
  weightGroupsInfo_.push_back(std::make_unique<gen::WeightGroupInfo>(info));
}

void GenWeightInfoProduct::addWeightGroupInfo(std::unique_ptr<gen::WeightGroupInfo> info) {
  weightGroupsInfo_.push_back(std::move(info));
}
