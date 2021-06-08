// -*- C++ -*-
//
// Package:     SimG4Core/SensitiveDetector
// Class  :     SensitiveDetectorMakerBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 08 Jun 2021 13:25:09 GMT
//

// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

SensitiveDetectorMakerBase::~SensitiveDetectorMakerBase() = default;

//
// member functions
//
void SensitiveDetectorMakerBase::beginRun(edm::EventSetup const&) {}

//
// const member functions
//
std::unique_ptr<SensitiveDetector> SensitiveDetectorMakerBase::make(const std::string& iname,
                                                                    const edm::EventSetup& es,
                                                                    const SensitiveDetectorCatalog& clg,
                                                                    const edm::ParameterSet& p,
                                                                    const SimTrackManager* man,
                                                                    SimActivityRegistry& reg) const {
  return make(iname, clg, p, man, reg);
}

std::unique_ptr<SensitiveDetector> SensitiveDetectorMakerBase::make(const std::string& iname,
                                                                    const SensitiveDetectorCatalog& clg,
                                                                    const edm::ParameterSet& p,
                                                                    const SimTrackManager* man,
                                                                    SimActivityRegistry& reg) const {
  return std::unique_ptr<SensitiveDetector>();
}

//
// static member functions
//
