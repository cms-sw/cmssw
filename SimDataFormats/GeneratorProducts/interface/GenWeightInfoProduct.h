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

  typedef std::vector<std::unique_ptr<gen::WeightGroupInfo>> WeightGroupInfoContainer;
}  // namespace gen

class GenWeightInfoProduct {
public:
  GenWeightInfoProduct() {}
  GenWeightInfoProduct(gen::WeightGroupInfoContainer& weightGroups);
  GenWeightInfoProduct(std::unique_ptr<GenWeightInfoProduct> other) { copy(*other); }
  GenWeightInfoProduct(const GenWeightInfoProduct& other) { copy(other); }
  GenWeightInfoProduct& operator=(const GenWeightInfoProduct& other) {
    copy(other);
    return *this;
  }
  void copy(const GenWeightInfoProduct& other) {
    for (auto& ptr : other.weightGroupsInfo_) {
      std::unique_ptr<gen::WeightGroupInfo> cloneptr(ptr->clone());
      weightGroupsInfo_.emplace_back(std::move(cloneptr));
    }
    unassociatedIdx_ = other.unassociatedIdx_;
  }
  ~GenWeightInfoProduct() {}

  const gen::WeightGroupInfoContainer& allWeightGroupsInfo() const;
  const std::vector<gen::WeightGroupData> allWeightGroupsInfoWithIndices() const;
  gen::WeightGroupData containingWeightGroupInfo(int index, size_t startSearch = 0) const;
  const gen::WeightGroupInfo* orderedWeightGroupInfo(int index) const;
  std::vector<gen::WeightGroupData> weightGroupsByType(gen::WeightType type) const;
  std::vector<int> weightGroupIndicesByType(gen::WeightType type) const;
  std::vector<gen::WeightGroupData> weightGroupsAndIndicesByType(gen::WeightType type, int maxStore = -1) const;
  std::optional<gen::WeightGroupData> pdfGroupWithIndexByLHAID(int lhaid) const;
  std::vector<gen::WeightGroupData> pdfGroupsWithIndicesByLHAIDs(const std::vector<int>& lhaids) const;
  void addWeightGroupInfo(gen::WeightGroupInfo info);
  void addWeightGroupInfo(std::unique_ptr<gen::WeightGroupInfo> info);
  const int numberOfGroups() const { return weightGroupsInfo_.size(); }
  // If there are unassociated weights, the number of filled groups will be less than the number
  // of groups, because the unassociated group can't be filled. Likewise the number of weights
  // in the GenWeightInfoProduct product will be less than the number of weights in the event
  const int numberOfFilledGroups() const {
    return std::accumulate(weightGroupsInfo_.begin(), weightGroupsInfo_.end(), 0, [](auto sum, auto& entry) {
      return sum + (entry->nIdsContained() > 0 ? 1 : 0);
    });
  }
  const int numberOfWeights() const {
    return std::accumulate(weightGroupsInfo_.begin(), weightGroupsInfo_.end(), 0, [](auto sum, auto& entry) {
      return sum + entry->nIdsContained();
    });
  }
  const int unassociatedIdx() const { return unassociatedIdx_; }

private:
  gen::WeightGroupInfoContainer weightGroupsInfo_;
  int unassociatedIdx_ = -1;
};

#endif  // GeneratorWeightInfo_LHEInterface_GenWeightInfoProduct_h
