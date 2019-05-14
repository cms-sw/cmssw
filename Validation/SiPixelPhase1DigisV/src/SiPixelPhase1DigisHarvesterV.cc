// -*- C++ -*-
//
// Package:     SiPixelPhase1DigisHarvesterV
// Class:       SiPixelPhase1DigisHarvesterV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/SiPixelPhase1DigisV/interface/SiPixelPhase1DigisV.h"

SiPixelPhase1DigisHarvesterV::SiPixelPhase1DigisHarvesterV(const edm::ParameterSet &iConfig)
    : SiPixelPhase1Harvester(iConfig) {}

DEFINE_FWK_MODULE(SiPixelPhase1DigisHarvesterV);
