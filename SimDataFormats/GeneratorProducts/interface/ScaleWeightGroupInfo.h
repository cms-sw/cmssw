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
    bool weightIsCorrupt_ = false;
    // Dyn_scale
    std::vector<std::string> dynNames_;
    std::vector<std::vector<size_t>> dynVec_;

    inline int indexFromMus(int muR, int muF) const { return 3 * muR + muF; }
    inline bool isValidValue(float mu) const { return mu == 0.5 || mu == 1.0 || mu == 2.0; }

    enum scaleIndices {
      muR0p5_muF0p5_idx = 0,
      muR0p5_muF1_idx = 1,
      muR0p5_muF2_idx = 2,
      muR1_muF0p5_idx = 3,
      Central_idx = 4,
      muR1_muF2_idx = 5,
      muR2_muF0p5_idx = 6,
      muR2_muF1_idx = 7,
      muR2_muF2_idx = 8
    };

  public:
    static const unsigned int MIN_SCALE_VARIATIONS = 9;
    ScaleWeightGroupInfo() : ScaleWeightGroupInfo("") {}
    ScaleWeightGroupInfo(std::string header, std::string name)
        : WeightGroupInfo(header, name), muIndices_(MIN_SCALE_VARIATIONS, -1), dynVec_(MIN_SCALE_VARIATIONS) {
      weightType_ = WeightType::kScaleWeights;
      isFunctionalFormVar_ = false;
    }
    ScaleWeightGroupInfo(std::string header) : ScaleWeightGroupInfo(header, header) {}
    ScaleWeightGroupInfo(const ScaleWeightGroupInfo& other) : WeightGroupInfo(other) { copy(other); }
    ~ScaleWeightGroupInfo() override {}
    void copy(const ScaleWeightGroupInfo& other);
    ScaleWeightGroupInfo* clone() const override;
    bool containsCentralWeight() const { return containsCentral_; }
    void addContainedId(int globalIndex, std::string id, std::string label, float muR, float muF);
    void setWeightIsCorrupt() {
      isWellFormed_ = false;
      weightIsCorrupt_ = true;
    }

    void setMuRMuFIndex(int globalIndex, std::string id, float muR, float muF);
    void setDyn(int globalIndex, std::string id, float muR, float muF, size_t dynNum, std::string_view dynName);
    int lhaid() { return lhaid_; }
    void setLhaid(int lhaid) { lhaid_ = lhaid; }
    // Is a variation of the functional form of the dynamic scale
    bool isFunctionalFormVariation();
    void setIsFunctionalFormVariation(bool functionalVar) { isFunctionalFormVar_ = functionalVar; }
    int centralIndex() const { return muIndices_.at(Central_idx); }
    int muR1muF2Index() const { return muIndices_.at(muR1_muF2_idx); }
    int muR1muF05Index() const { return muIndices_.at(muR1_muF0p5_idx); }
    int muR2muF05Index() const { return muIndices_.at(muR2_muF0p5_idx); }
    int muR2muF1Index() const { return muIndices_.at(muR2_muF1_idx); }
    int muR2muF2Index() const { return muIndices_.at(muR2_muF2_idx); }
    int muR05muF05Index() const { return muIndices_.at(muR0p5_muF0p5_idx); }
    int muR05muF1Index() const { return muIndices_.at(muR0p5_muF1_idx); }
    int muR05muF2Index() const { return muIndices_.at(muR0p5_muF2_idx); }

    // dynweight version
    size_t centralIndex(std::string_view dynName) const { return scaleIndex(Central_idx, dynName); }
    size_t muR1muF2Index(std::string_view dynName) const { return scaleIndex(muR1_muF2_idx, dynName); }
    size_t muR1muF05Index(std::string_view dynName) const { return scaleIndex(muR1_muF0p5_idx, dynName); }
    size_t muR2muF05Index(std::string_view dynName) const { return scaleIndex(muR2_muF0p5_idx, dynName); }
    size_t muR2muF1Index(std::string_view dynName) const { return scaleIndex(muR2_muF1_idx, dynName); }
    size_t muR2muF2Index(std::string_view dynName) const { return scaleIndex(muR2_muF2_idx, dynName); }
    size_t muR05muF05Index(std::string_view dynName) const { return scaleIndex(muR0p5_muF0p5_idx, dynName); }
    size_t muR05muF1Index(std::string_view dynName) const { return scaleIndex(muR0p5_muF1_idx, dynName); }
    size_t muR05muF2Index(std::string_view dynName) const { return scaleIndex(muR0p5_muF2_idx, dynName); }

    size_t scaleIndex(float muR, float muF, size_t dynNum) const;
    size_t scaleIndex(float muR, float muF, std::string_view dynName) const;
    size_t scaleIndex(int index, std::string_view dynName) const;
    size_t scaleIndex(float muR, float muF) const;

    size_t scaleIndex(int index, size_t dynNum) const { return dynVec_.at(index).at(dynNum); }
    std::vector<std::string> dynNames() const;
  };
}  // namespace gen

#endif
