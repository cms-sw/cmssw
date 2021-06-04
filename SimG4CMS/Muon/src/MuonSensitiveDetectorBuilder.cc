// -*- C++ -*-
//
// Package:     SimG4CMS/Muon
// Class  :     MuonSensitiveDetectorBuilder
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 04 Jun 2021 18:18:17 GMT
//

// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "SimG4CMS/Muon/interface/MuonSensitiveDetector.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"

class MuonSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
  SensitiveDetector* make(const std::string& iname,
                          const edm::EventSetup& es,
                          const SensitiveDetectorCatalog& clg,
                          const edm::ParameterSet& p,
                          const SimTrackManager* man,
                          SimActivityRegistry& reg) const override {
    MuonSensitiveDetector* sd = new MuonSensitiveDetector(iname, es, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd);
    return sd;
  }
};

DEFINE_EDM_PLUGIN(SensitiveDetectorPluginFactory, MuonSensitiveDetectorBuilder, "MuonSensitiveDetector");
