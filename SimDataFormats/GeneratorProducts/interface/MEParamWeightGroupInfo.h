#ifndef SimDataFormats_GeneratorProducts_MEParamWeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_MEParamWeightGroupInfo_h

#include <unordered_map>
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
    int getCentralIndex() { return centralIdx; }
    int getVariationIndex(int sig) { return massValue.at(numSigma + sig).second; }
    double getVariationValue(int sig) { return massValue.at(numSigma + sig).first; }
    void updateWeight(int globalIndex, std::string id, double weight);
    void updateWeight(int globalIndex, std::string id, std::vector<std::string>& content);

  private:
    std::unordered_map<size_t, std::vector<std::string>> splitContent;
    std::vector<std::pair<double, int>> massValue;
    double central;
    int centralIdx;
    int numSigma;
  };
}  // namespace gen

#endif  // SimDataFormats_GeneratorProducts_MEParamWeightGroupInfo_h
