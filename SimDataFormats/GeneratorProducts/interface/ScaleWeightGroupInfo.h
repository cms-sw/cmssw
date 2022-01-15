#ifndef SimDataFormats_GeneratorProducts_ScaleWeightGroupInfo_h
#define SimDataFormats_GeneratorProducts_ScaleWeightGroupInfo_h

#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"
#include <unordered_map>
#include <vector>

namespace gen {
  class ScaleWeightGroupInfo : public WeightGroupInfo {
  private:
    bool isFunctionalFormVar_;
    std::vector<size_t> muIndices_;
    bool containsCentral_ = false;
    int lhaid_ = -1;
    // Dyn_scale
    std::vector<std::string> dynNames_;
    std::vector<std::vector<size_t>> dynVec_;

    void setMuRMuFIndex(WeightMetaInfo& info, float muR, float muF);
    void setMuRMuFIndex(WeightMetaInfo& info, float muR, float muF, size_t dynNum, std::string dynName);
    inline int getIndex(int muR, int muF) const { return 3 * muR + muF; }
    inline bool isValidValue(float mu) const { return mu == 0.5 || mu == 1.0 || mu == 2.0; }

  public:
    ScaleWeightGroupInfo() : ScaleWeightGroupInfo("") {}
    ScaleWeightGroupInfo(std::string header, std::string name)
        : WeightGroupInfo(header, name), muIndices_(9, -1), dynVec_(9) {
      weightType_ = WeightType::kScaleWeights;
      isFunctionalFormVar_ = false;
    }
    ScaleWeightGroupInfo(std::string header) : ScaleWeightGroupInfo(header, header) {}
    ScaleWeightGroupInfo(const ScaleWeightGroupInfo& other) { copy(other); }
    virtual ~ScaleWeightGroupInfo() override {}
    void copy(const ScaleWeightGroupInfo& other);
    virtual ScaleWeightGroupInfo* clone() const override;
    bool containsCentralWeight() const { return containsCentral_; }

    void setMuRMuFIndex(
        int globalIndex, std::string id, float muR, float muF, size_t dynNum = -1, std::string dynName = "");
    void addContainedId(int weightEntry, std::string id, std::string label, float muR, float muF);
    int lhaid() { return lhaid_; }
    void setLhaid(int lhaid) { lhaid_ = lhaid; }
    // Is a variation of the functional form of the dynamic scale
    bool isFunctionalFormVariation();
    void setIsFunctionalFormVariation(bool functionalVar) { isFunctionalFormVar_ = functionalVar; }
    size_t centralIndex() const { return muIndices_.at(4); }
    size_t muR1muF2Index() const { return muIndices_.at(5); }
    size_t muR1muF05Index() const { return muIndices_.at(3); }
    size_t muR2muF05Index() const { return muIndices_.at(6); }
    size_t muR2muF1Index() const { return muIndices_.at(7); }
    size_t muR2muF2Index() const { return muIndices_.at(8); }
    size_t muR05muF05Index() const { return muIndices_.at(0); }
    size_t muR05muF1Index() const { return muIndices_.at(1); }
    size_t muR05muF2Index() const { return muIndices_.at(2); }
    // dynweight version
    size_t centralIndex(std::string& dynName) const { return getScaleIndex(4, dynName); }
    size_t muR1muF2Index(std::string& dynName) const { return getScaleIndex(5, dynName); }
    size_t muR1muF05Index(std::string& dynName) const { return getScaleIndex(3, dynName); }
    size_t muR2muF05Index(std::string& dynName) const { return getScaleIndex(6, dynName); }
    size_t muR2muF1Index(std::string& dynName) const { return getScaleIndex(7, dynName); }
    size_t muR2muF2Index(std::string& dynName) const { return getScaleIndex(8, dynName); }
    size_t muR05muF05Index(std::string& dynName) const { return getScaleIndex(0, dynName); }
    size_t muR05muF1Index(std::string& dynName) const { return getScaleIndex(1, dynName); }
    size_t muR05muF2Index(std::string& dynName) const { return getScaleIndex(2, dynName); }

    size_t getScaleIndex(float muR, float muF, size_t dynNum) const;
    size_t getScaleIndex(float muR, float muF, std::string& dynName) const;
    size_t getScaleIndex(int index, std::string& dynName) const;
    size_t getScaleIndex(float muR, float muF) const;

    size_t getScaleIndex(int index, size_t dynNum) const { return dynVec_.at(index).at(dynNum); }
    std::vector<std::string> getDynNames() const;
  };
}  // namespace gen

#endif
