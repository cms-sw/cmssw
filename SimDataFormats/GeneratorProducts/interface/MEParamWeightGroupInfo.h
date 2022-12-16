#ifndef SimDataFormats_GeneratorProducts_MEParamWeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_MEParamWeightGroupInfo_h

#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"

namespace gen {
  class MEParamWeightGroupInfo : public WeightGroupInfo {
  public:
    MEParamWeightGroupInfo() : WeightGroupInfo() { weightType_ = WeightType::kMEParamWeights; }
    MEParamWeightGroupInfo(std::string header, std::string name) : WeightGroupInfo(header, name) {
      weightType_ = WeightType::kMEParamWeights;
    }
    MEParamWeightGroupInfo(std::string header) : MEParamWeightGroupInfo(header, header) {}
    ~MEParamWeightGroupInfo() override {}
    MEParamWeightGroupInfo(const MEParamWeightGroupInfo& other) : WeightGroupInfo(other) { copy(other); }
    void copy(const MEParamWeightGroupInfo& other);
    MEParamWeightGroupInfo* clone() const override;
  };
}  // namespace gen

#endif  // SimDataFormats_GeneratorProducts_MEParamWeightGroupInfo_h
