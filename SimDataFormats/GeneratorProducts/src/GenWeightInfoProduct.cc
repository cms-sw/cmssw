#include <vector>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "FWCore/Utilities/interface/Exception.h"

GenWeightInfoProduct::GenWeightInfoProduct(std::vector<std::unique_ptr<gen::WeightGroupInfo>>& weightGroups) {
  weightGroupsInfo_ = std::move(weightGroups);
}

GenWeightInfoProduct::GenWeightInfoProduct(std::vector<std::unique_ptr<gen::WeightGroupInfo>> weightGroups) {
  weightGroupsInfo_ = std::move(weightGroups);
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

const gen::WeightGroupInfo* GenWeightInfoProduct::containingWeightGroupInfo(int index) const {
  // The weight values in the product are arranged to match the order of the groups in the GenWeightInfoProduct
  int maxIdx = 0;
  for (const auto& weightGroup : weightGroupsInfo_) {
    int minIdx = maxIdx;
    maxIdx = maxIdx+weightGroup->nIdsContained();
    if (index >= minIdx && index < maxIdx)
      return weightGroup.get();
  }
  throw cms::Exception("GenWeightInfoProduct") << "No weight group found containing the weight index requested";
}

const gen::WeightGroupInfo* GenWeightInfoProduct::orderedWeightGroupInfo(int weightGroupIndex) const {
  if (weightGroupIndex >= static_cast<int>(weightGroupsInfo_.size()))
    throw cms::Exception("GenWeightInfoProduct")
        << "Weight index requested is outside the range of weights in the product";
  return weightGroupsInfo_[weightGroupIndex].get();
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::weightGroupsAndIndicesByType(gen::WeightType type) const {
  std::vector<gen::WeightGroupData> matchingGroups;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    const gen::WeightGroupInfo* group = weightGroupsInfo_[i].get();
    if (group->weightType() == type)
      matchingGroups.push_back({i, group});
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
