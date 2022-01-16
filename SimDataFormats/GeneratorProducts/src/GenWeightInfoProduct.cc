#include <vector>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "FWCore/Utilities/interface/Exception.h"

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

std::unique_ptr<const gen::WeightGroupInfo> GenWeightInfoProduct::containingWeightGroupInfo(int index) const {
  for (const auto& weightGroup : weightGroupsInfo_) {
    if (weightGroup.indexInRange(index))
      return std::unique_ptr<const gen::WeightGroupInfo>(weightGroup.clone());
  }
  throw cms::Exception("GenWeightInfoProduct") << "No weight group found containing the weight index requested";
}

std::unique_ptr<const gen::WeightGroupInfo> GenWeightInfoProduct::orderedWeightGroupInfo(int weightGroupIndex) const {
  if (weightGroupIndex >= static_cast<int>(weightGroupsInfo_.size()))
    throw cms::Exception("GenWeightInfoProduct")
        << "Weight index requested is outside the range of weights in the product";
  return std::unique_ptr<const gen::WeightGroupInfo>(weightGroupsInfo_[weightGroupIndex].clone());
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::weightGroupsAndIndicesByType(gen::WeightType type) const {
  std::vector<gen::WeightGroupData> matchingGroups;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    const gen::WeightGroupInfo& group = weightGroupsInfo_[i];
    if (weightGroupsInfo_[i].weightType() == type)
      matchingGroups.push_back({i, std::unique_ptr<const gen::WeightGroupInfo>(group.clone())});
  }
  return matchingGroups;
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::weightGroupsByType(gen::WeightType type) const {
  std::vector<gen::WeightGroupData> matchingGroups;
  for (size_t i = 0; i < weightGroupsInfo_.size(); i++) {
    const gen::WeightGroupInfo& group = weightGroupsInfo_[i];
    if (weightGroupsInfo_[i].weightType() == type)
      matchingGroups.push_back({i, std::unique_ptr<const gen::WeightGroupInfo>(group.clone())});
  }
  return matchingGroups;
}

std::optional<gen::WeightGroupData> GenWeightInfoProduct::pdfGroupWithIndexByLHAID(int lhaid) const {
  std::vector<gen::WeightGroupData> pdfGroups = weightGroupsAndIndicesByType(gen::WeightType::kPdfWeights);

  auto matchingPdfSet = std::find_if(pdfGroups.begin(), pdfGroups.end(), [lhaid](gen::WeightGroupData& data) {
    auto pdfGroup = std::unique_ptr<const gen::PdfWeightGroupInfo>(
        static_cast<const gen::PdfWeightGroupInfo*>(data.group.release()));
    return pdfGroup->containsLhapdfId(lhaid);
  });

  return matchingPdfSet == pdfGroups.end()
             ? std::nullopt
             : std::optional<gen::WeightGroupData>({matchingPdfSet->index, std::move(matchingPdfSet->group)});
}

std::vector<gen::WeightGroupData> GenWeightInfoProduct::pdfGroupsWithIndicesByLHAIDs(
    const std::vector<int>& lhaids) const {
  auto pdfGroups = weightGroupsAndIndicesByType(gen::WeightType::kPdfWeights);
  std::vector<gen::WeightGroupData> groups;

  for (auto lhaid : lhaids) {
    auto matchingPdfSet = std::find_if(pdfGroups.begin(), pdfGroups.end(), [lhaid](gen::WeightGroupData& data) {
      auto pdfGroup = std::unique_ptr<const gen::PdfWeightGroupInfo>(
          static_cast<const gen::PdfWeightGroupInfo*>(data.group.release()));
      return pdfGroup->containsLhapdfId(lhaid);
    });
    if (matchingPdfSet != pdfGroups.end()) {
      pdfGroups.push_back({matchingPdfSet->index, std::move(matchingPdfSet->group)});
    }
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

void GenWeightInfoProduct::addWeightGroupInfo(std::unique_ptr<gen::WeightGroupInfo> info) {
  weightGroupsInfo_.push_back(std::move(info));
}
