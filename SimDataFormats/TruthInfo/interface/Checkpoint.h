// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef SimDataFormats_TruthInfo_interface_Checkpoint_h
#define SimDataFormats_TruthInfo_interface_Checkpoint_h

#include <cstdint>

#include "DataFormats/Math/interface/LorentzVector.h"

namespace truth {

  // A trajectory checkpoint of a logical particle (e.g. the calorimeter boundary
  // crossing): the position and momentum recorded at a labelled point.
  struct Checkpoint {
    uint32_t checkpointId = 0;
    math::XYZTLorentzVectorF position;
    math::XYZTLorentzVectorF momentum;
  };

}  // namespace truth

#endif
