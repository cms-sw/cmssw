#include "SimDataFormats/GeneratorProducts/interface/MEParamWeightGroupInfo.h"

// To be expanded with more specific behaviours in the future
namespace gen {
  void MEParamWeightGroupInfo::copy(const MEParamWeightGroupInfo& other) { WeightGroupInfo::copy(other); }

  MEParamWeightGroupInfo* MEParamWeightGroupInfo::clone() const { return new MEParamWeightGroupInfo(*this); }
}  // namespace gen
