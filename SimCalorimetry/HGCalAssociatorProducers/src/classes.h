// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef SimCalorimetry_HGCalAssociatorProducers_classes_h
#define SimCalorimetry_HGCalAssociatorProducers_classes_h

#include "DataFormats/Common/interface/Wrapper.h"
#include "SimCalorimetry/HGCalAssociatorProducers/interface/DetIdRecHitMap.h"

namespace SimCalorimetry_HGCalAssociatorProducers {
  struct dictionary {
    hgcal::DetIdRecHitMap detIdRecHitMap_;
    edm::Wrapper<hgcal::DetIdRecHitMap> detIdRecHitMapWrapper_;
  };
}  // namespace SimCalorimetry_HGCalAssociatorProducers

#endif
