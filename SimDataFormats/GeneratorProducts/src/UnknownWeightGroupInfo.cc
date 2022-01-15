#include "SimDataFormats/GeneratorProducts/interface/UnknownWeightGroupInfo.h"

namespace gen {
  UnknownWeightGroupInfo* UnknownWeightGroupInfo::clone() const { return new UnknownWeightGroupInfo(*this); }
}  // namespace gen
