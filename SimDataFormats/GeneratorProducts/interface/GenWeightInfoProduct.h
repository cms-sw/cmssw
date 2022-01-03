#ifndef SimDataFormats_GeneratorProducts_GenWeightInfoProduct_h
#define SimDataFormats_GeneratorProducts_GenWeightInfoProduct_h

#include <iterator>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <numeric>
#include <memory>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"

namespace gen {
  struct WeightGroupData {
    size_t index;
    const gen::WeightGroupInfo* group;
  };

  struct SharedWeightGroupData {
    size_t index;
    std::shared_ptr<const gen::WeightGroupInfo> group;
  };
}  // namespace gen

class GenWeightInfoProduct {
public:
  GenWeightInfoProduct() {}
  GenWeightInfoProduct(std::vector<std::unique_ptr<gen::WeightGroupInfo>>& weightGroups);
  GenWeightInfoProduct(std::vector<std::unique_ptr<gen::WeightGroupInfo>> weightGroups);
  GenWeightInfoProduct(std::unique_ptr<GenWeightInfoProduct> other) { 
      for (auto& ptr : other->weightGroupsInfo_) {
        std::unique_ptr<gen::WeightGroupInfo> cloneptr(ptr->clone());
        weightGroupsInfo_.emplace_back(std::move(cloneptr));
      }
  }
  ~GenWeightInfoProduct() {}

  const std::vector<std::unique_ptr<gen::WeightGroupInfo>>& allWeightGroupsInfo() const;
  const std::vector<gen::WeightGroupData> allWeightGroupsInfoWithIndices() const;
  const gen::WeightGroupInfo* containingWeightGroupInfo(int index) const;
  const gen::WeightGroupInfo* orderedWeightGroupInfo(int index) const;
  std::vector<gen::WeightGroupData> weightGroupsByType(gen::WeightType type) const;
  std::vector<int> weightGroupIndicesByType(gen::WeightType type) const;
  std::vector<gen::WeightGroupData> weightGroupsAndIndicesByType(gen::WeightType type) const;
  std::optional<gen::WeightGroupData> pdfGroupWithIndexByLHAID(int lhaid) const;
  std::vector<gen::WeightGroupData> pdfGroupsWithIndicesByLHAIDs(const std::vector<int>& lhaids) const;
  void addWeightGroupInfo(gen::WeightGroupInfo info);
  void addWeightGroupInfo(std::unique_ptr<gen::WeightGroupInfo> info);
  const int numberOfGroups() const { return weightGroupsInfo_.size(); }
  // If there are unassociated weights, the number of filled groups will be less than the number 
  // of groups, because the unassociated group can't be filled. Likewise the number of weights
  // in the GenWeightInfoProduct product will be less than the number of weights in the event
  const int numberOfFilledGroups() const { 
      return std::accumulate(weightGroupsInfo_.begin(), weightGroupsInfo_.end(), 0,
                [](auto sum, auto& entry) { return sum + (entry->nIdsContained() > 0 ? 1 : 0); }); 
  }
  const int numberOfWeights() const { 
      return std::accumulate(weightGroupsInfo_.begin(), weightGroupsInfo_.end(), 0,
                [](auto sum, auto& entry) { return sum + entry->nIdsContained(); }); 
  }


private:
  std::vector<std::unique_ptr<gen::WeightGroupInfo>> weightGroupsInfo_;
};

#endif  // GeneratorWeightInfo_LHEInterface_GenWeightInfoProduct_h
