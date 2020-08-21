#ifndef SimDataFormats_GeneratorProducts_GenWeightInfoProduct_h
#define SimDataFormats_GeneratorProducts_GenWeightInfoProduct_h

#include <iterator>
#include <memory>
#include <vector>
#include <string>
#include <optional>

//#include <hepml.hpp>

#include "DataFormats/Common/interface/OwnVector.h"
#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"

namespace gen {
  struct WeightGroupData {
    size_t index;
    std::unique_ptr<gen::WeightGroupInfo> group;
  };

  struct SharedWeightGroupData {
    size_t index;
    std::shared_ptr<gen::WeightGroupInfo> group;
  };
}  // namespace gen

class GenWeightInfoProduct {
public:
  GenWeightInfoProduct() {}
  GenWeightInfoProduct(edm::OwnVector<gen::WeightGroupInfo>& weightGroups);
  GenWeightInfoProduct(const GenWeightInfoProduct& other);
  GenWeightInfoProduct(GenWeightInfoProduct&& other);
  ~GenWeightInfoProduct() {}
  GenWeightInfoProduct& operator=(const GenWeightInfoProduct& other);
  GenWeightInfoProduct& operator=(GenWeightInfoProduct&& other);

  const edm::OwnVector<gen::WeightGroupInfo>& allWeightGroupsInfo() const;
  const std::unique_ptr<gen::WeightGroupInfo> containingWeightGroupInfo(int index) const;
  const std::unique_ptr<gen::WeightGroupInfo> orderedWeightGroupInfo(int index) const;
  std::vector<std::unique_ptr<gen::WeightGroupInfo>> weightGroupsByType(gen::WeightType type) const;
  std::vector<int> weightGroupIndicesByType(gen::WeightType type) const;
  std::vector<gen::WeightGroupData> weightGroupsAndIndicesByType(gen::WeightType type) const;
  std::optional<gen::WeightGroupData> pdfGroupWithIndexByLHAID(int lhaid) const;
  std::vector<gen::WeightGroupData> pdfGroupsWithIndicesByLHAIDs(const std::vector<int>& lhaids) const;
  void addWeightGroupInfo(std::unique_ptr<gen::WeightGroupInfo> info);
  const int numberOfGroups() const { return weightGroupsInfo_.size(); }

private:
  edm::OwnVector<gen::WeightGroupInfo> weightGroupsInfo_;
};

#endif  // GeneratorWeightInfo_LHEInterface_GenWeightInfoProduct_h
