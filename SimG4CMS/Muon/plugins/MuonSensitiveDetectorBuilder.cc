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

#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimG4CMS/Muon/interface/MuonSensitiveDetector.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MuonSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit MuonSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : offmap_{nullptr},
        mdc_{nullptr},
        offsetToken_{cc.esConsumes<edm::Transition::BeginRun>()},
        geomConstantsToken_{cc.esConsumes<edm::Transition::BeginRun>()} {
    edm::ParameterSet muonSD = p.getParameter<edm::ParameterSet>("MuonSD");
    ePersistentCutGeV_ = muonSD.getParameter<double>("EnergyThresholdForPersistency") / CLHEP::GeV;  //Default 1. GeV
    allMuonsPersistent_ = muonSD.getParameter<bool>("AllMuonsPersistent");
    printHits_ = muonSD.getParameter<bool>("PrintHits");
    dd4hep_ = p.getParameter<bool>("g4GeometryDD4hepSource");
  }

  void beginRun(const edm::EventSetup& es) final {
    edm::ESHandle<MuonOffsetMap> mom = es.getHandle(offsetToken_);
    offmap_ = (mom.isValid()) ? mom.product() : nullptr;
    edm::LogVerbatim("MuonSim") << "Finds the offset map at " << offmap_;
    mdc_ = &es.getData(geomConstantsToken_);
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<MuonSensitiveDetector>(
        iname, offmap_, *mdc_, clg, ePersistentCutGeV_, allMuonsPersistent_, printHits_, dd4hep_, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const MuonOffsetMap* offmap_;
  const MuonGeometryConstants* mdc_;
  const edm::ESGetToken<MuonOffsetMap, IdealGeometryRecord> offsetToken_;
  const edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> geomConstantsToken_;
  double ePersistentCutGeV_;
  bool allMuonsPersistent_;
  bool printHits_;
  bool dd4hep_;
};

DEFINE_SENSITIVEDETECTORBUILDER(MuonSensitiveDetectorBuilder, MuonSensitiveDetector);
