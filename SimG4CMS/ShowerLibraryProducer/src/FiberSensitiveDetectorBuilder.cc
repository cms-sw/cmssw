// -*- C++ -*-
//
// Package:     SimG4CMS/ShowerLibraryProducer
// Class  :     FiberSensitiveDetectorBuilder
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Sunanda Banerjee
//         Created: Tue, 13 Jun 2021 15:18:17 GMT
//

// system include files
#include <string>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "SimG4CMS/ShowerLibraryProducer/interface/FiberSensitiveDetector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FiberSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit FiberSensitiveDetectorBuilder(const edm::ParameterSet& p, edm::ConsumesCollector cc)
      : cspsToken_{cc.esConsumes<edm::Transition::BeginRun>()},
        cdcToken_{cc.esConsumes<edm::Transition::BeginRun>()},
        hcalSimCons_{nullptr},
        hcalDDCons_{nullptr} {}

  void beginRun(const edm::EventSetup& es) final {
    hcalSimCons_ = &es.getData(cspsToken_);
    hcalDDCons_ = &es.getData(cdcToken_);
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<FiberSensitiveDetector>(iname, hcalSimCons_, hcalDDCons_, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const edm::ESGetToken<HcalSimulationConstants, HcalSimNumberingRecord> cspsToken_;
  const edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> cdcToken_;
  const HcalSimulationConstants* hcalSimCons_;
  const HcalDDDSimConstants* hcalDDCons_;
};

DEFINE_SENSITIVEDETECTORBUILDER(FiberSensitiveDetectorBuilder, FiberSensitiveDetector);
