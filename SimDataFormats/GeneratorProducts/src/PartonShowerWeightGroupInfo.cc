#include "SimDataFormats/GeneratorProducts/interface/PartonShowerWeightGroupInfo.h"
#include <exception>
#include <iostream>

namespace gen {
  PartonShowerWeightGroupInfo::PartonShowerWeightGroupInfo(std::string header, std::string name)
      : WeightGroupInfo(header, name) {
    weightType_ = WeightType::kPartonShowerWeights;
  }

  void PartonShowerWeightGroupInfo::copy(const PartonShowerWeightGroupInfo& other) {
    WeightGroupInfo::copy(other);
    nameIsPythiaSyntax_ = other.nameIsPythiaSyntax_;
  }

  PartonShowerWeightGroupInfo* PartonShowerWeightGroupInfo::clone() const {
    return new PartonShowerWeightGroupInfo(*this);
  }

  int PartonShowerWeightGroupInfo::variationIndex(bool isISR, bool isUp, PSVarType variationType) const {
    return variationIndex(isISR, isUp, variationType, PSSplittingType::combined);
  }

  int PartonShowerWeightGroupInfo::variationIndex(bool isISR,
                                                  bool isUp,
                                                  PSVarType variationType,
                                                  PSSplittingType splittingType) const {
    std::string varName = variationName(isISR, isUp, variationType, splittingType);
    int wgtIdx = weightIndexFromLabel(varName);
    // Guess PS idx if not in label list
    if (wgtIdx == -1 && guessPSWeightIdx_)
      wgtIdx = psWeightIdxGuess(varName);
    return wgtIdx;
  }

  std::string PartonShowerWeightGroupInfo::variationName(bool isISR,
                                                         bool isUp,
                                                         PSVarType variationType,
                                                         PSSplittingType splittingType) const {
    std::string label = isISR ? "isr" : "fsr";

    // if ((variationType == PSVarType::con || variationType == PSVarType::def || variationType == PSVarType::red) &&
    //     splittingType != PSSplittingType::combined)
    //   throw std::invalid_argument("VariationType must be muR or CNS if subprocess is specified");

    if (nameIsPythiaSyntax_) {
      // Splitting
      if (splittingType == PSSplittingType::g2gg)
        label += ":g2gg";
      else if (splittingType == PSSplittingType::g2qq)
        label += ":g2qq";
      else if (splittingType == PSSplittingType::x2xg)
        label += ":x2xg";
      else if (splittingType == PSSplittingType::q2qg)
        label += ":q2qg";
      // type
      if (variationType == PSVarType::con)
        label += isUp ? ":murfac=4.0" : ":murfac=0.25";
      else if (variationType == PSVarType::def || variationType == PSVarType::muR)
        label += isUp ? ":murfac=2.0" : ":murfac=0.5";
      else if (variationType == PSVarType::red)
        label += isUp ? ":murfac=1.414" : ":murfac=0.707";
      else if (variationType == PSVarType::cNS)
        label += isUp ? ":cns=2.0" : ":cns=-2.0";
    } else {
      // Splitting
      if (splittingType == PSSplittingType::g2gg)
        label += "_G2GG_";
      else if (splittingType == PSSplittingType::g2qq)
        label += "_G2QQ_";
      else if (splittingType == PSSplittingType::x2xg)
        label += "_X2XG_";
      else if (splittingType == PSSplittingType::q2qg)
        label += "_Q2QG_";
      // type
      if (variationType == PSVarType::con)
        label += "Con";
      else if (variationType == PSVarType::def)
        label += "Def";
      else if (variationType == PSVarType::muR)
        label += "muR";
      else if (variationType == PSVarType::red)
        label += "Red";
      else if (variationType == PSVarType::cNS)
        label += "cNS";
      // Up/Down
      if (splittingType != PSSplittingType::combined) {
        label += isUp ? "_up" : "_dn";
      } else
        label += isUp ? "Hi" : "Lo";
    }
    return label;
  }

  int PartonShowerWeightGroupInfo::psWeightIdxGuess(const std::string& varName) const {
    int wgtIdx;
    if (nameIsPythiaSyntax_) {
      auto wgtIter = std::find(expectedOrderPythiaSyntax_.begin(), expectedOrderPythiaSyntax_.end(), varName);
      wgtIdx = wgtIter - expectedOrderPythiaSyntax_.begin() + 2;
    } else {
      auto wgtIter = std::find(expectedOrder_.begin(), expectedOrder_.end(), varName);
      wgtIdx = wgtIter - expectedOrder_.begin() + 2;
    }
    if (wgtIdx >= static_cast<int>(nIdsContained()))
      wgtIdx = -1;
    return wgtIdx;
  }

  void PartonShowerWeightGroupInfo::printVariables() const {
    const auto& variations = (nameIsPythiaSyntax_) ? expectedOrderPythiaSyntax_ : expectedOrder_;
    for (const auto& varName : variations) {
      int wgtIdx = weightIndexFromLabel(varName);
      // Guess PS idx if not in label list
      if (wgtIdx == -1 && guessPSWeightIdx_)
        wgtIdx = psWeightIdxGuess(varName);
      if (wgtIdx != -1)
        std::cout << varName << " : " << wgtIdx << std::endl;
    }
  }

}  // namespace gen
