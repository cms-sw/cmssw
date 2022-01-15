#include <vector>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"

GenWeightInfoProduct::GenWeightInfoProduct(edm::OwnVector<gen::WeightGroupInfo>& weightGroups) {
  weightGroupsInfo_ = weightGroups;
}

GenWeightInfoProduct& GenWeightInfoProduct::operator=(const GenWeightInfoProduct& other) {
  weightGroupsInfo_ = other.weightGroupsInfo_;
  return *this;
}

GenWeightInfoProduct& GenWeightInfoProduct::operator=(GenWeightInfoProduct&& other) {
  weightGroupsInfo_ = std::move(other.weightGroupsInfo_);
  return *this;
}

const edm::OwnVector<gen::WeightGroupInfo>& GenWeightInfoProduct::allWeightGroupsInfo() const {
  return weightGroupsInfo_;
}

const gen::WeightGroupInfo* GenWeightInfoProduct::containingWeightGroupInfo(int index) const {
  for (const auto& weightGroup : weightGroupsInfo_) {
    if (weightGroup.indexInRange(index))
      return &weightGroup;
  }
  throw std::domain_error("Failed to find containing weight group");
}

const gen::WeightGroupInfo* GenWeightInfoProduct::orderedWeightGroupInfo(int weightGroupIndex) const {
  if (weightGroupIndex >= static_cast<int>(weightGroupsInfo_.size()))
    throw std::range_error("Weight index out of range!");
  return &weightGroupsInfo_[weightGroupIndex];
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::weightGroupsAndIndicesByType(gen::WeightType type) const {
  std::vector<gen::WeightGroupData> matchingGroups;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    if (weightGroupsInfo_[i].weightType() == type)
      matchingGroups.push_back({i, weightGroupsInfo_[i].clone()});
  }
  return matchingGroups;
}

std::vector<gen::WeightGroupInfo*> GenWeightInfoProduct::weightGroupsByType(gen::WeightType type) const {
  std::vector<gen::WeightGroupInfo*> matchingGroups;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    if (weightGroupsInfo_[i].weightType() == type)
      matchingGroups.push_back(weightGroupsInfo_[i].clone());
  }
  return matchingGroups;
}

std::optional<gen::WeightGroupData> GenWeightInfoProduct::pdfGroupWithIndexByLHAID(int lhaid) const {
  auto pdfGroups = weightGroupsAndIndicesByType(gen::WeightType::kPdfWeights);

  auto matchingPdfSet = std::find_if(pdfGroups.begin(), pdfGroups.end(), [lhaid](gen::WeightGroupData& data) {
    auto pdfGroup = dynamic_cast<const gen::PdfWeightGroupInfo*>(data.group);
    return pdfGroup->containsLhapdfId(lhaid);
  });
  return matchingPdfSet != pdfGroups.end() ? std::optional<gen::WeightGroupData>(*matchingPdfSet) : std::nullopt;
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::pdfGroupsWithIndicesByLHAIDs(
    const std::vector<int>& lhaids) const {
  auto pdfGroups = weightGroupsAndIndicesByType(gen::WeightType::kPdfWeights);
  std::vector<gen::WeightGroupData> groups;

  for (auto lhaid : lhaids) {
    auto matchingPdfSet = std::find_if(pdfGroups.begin(), pdfGroups.end(), [lhaid](gen::WeightGroupData& data) {
      auto pdfGroup = dynamic_cast<const gen::PdfWeightGroupInfo*>(data.group);
      return pdfGroup->containsLhapdfId(lhaid);
    });
    if (matchingPdfSet != pdfGroups.end())
      pdfGroups.push_back(*matchingPdfSet);
  }

  return pdfGroups;
}

std::vector<int> GenWeightInfoProduct::weightGroupIndicesByType(gen::WeightType type) const {
  std::vector<int> matchingGroupIndices;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    if (weightGroupsInfo_[i].weightType() == type)
      matchingGroupIndices.push_back(i);
  }
  return matchingGroupIndices;
}

void GenWeightInfoProduct::addWeightGroupInfo(gen::WeightGroupInfo* info) { weightGroupsInfo_.push_back(info); }
