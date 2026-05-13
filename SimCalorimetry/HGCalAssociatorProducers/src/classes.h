#ifndef SimCalorimetry_HGCalAssociatorProducers_classes_h
#define SimCalorimetry_HGCalAssociatorProducers_classes_h

#include <unordered_map>

#include "DataFormats/Common/interface/Wrapper.h"

namespace SimCalorimetry_HGCalAssociatorProducers {
  struct dictionary {
    std::unordered_map<unsigned int, unsigned int> simHitToRecHitMap_;
    edm::Wrapper<std::unordered_map<unsigned int, unsigned int>> simHitToRecHitMapWrapper_;
  };
}  // namespace SimCalorimetry_HGCalAssociatorProducers

#endif
