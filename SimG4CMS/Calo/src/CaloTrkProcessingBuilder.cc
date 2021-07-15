// -*- C++ -*-
//
// Package:     SimG4CMS/Calo
// Class  :     CaloTrkProcessingBuilder
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Sunanda Banerjee
//         Created:  Fri, 12 Jun 2021 23:18:17 GMT
//

// system include files
#include <string>
#include <vector>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "SimG4CMS/Calo/interface/CaloTrkProcessing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CaloTrkProcessingBuilder : public SensitiveDetectorMakerBase {
public:
  explicit CaloTrkProcessingBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : cspsToken_{cc.esConsumes<edm::Transition::BeginRun>()}, caloSimPar_{nullptr} {
    bool dd4hep = p.getParameter<bool>("g4GeometryDD4hepSource");
    addlevel_ = dd4hep ? 1 : 0;
    edm::ParameterSet csps = p.getParameter<edm::ParameterSet>("CaloTrkProcessing");
    testBeam_ = csps.getParameter<bool>("TestBeam");
    eMin_ = csps.getParameter<double>("EminTrack") * CLHEP::MeV;
    putHistory_ = csps.getParameter<bool>("PutHistory");
    doFineCalo_ = csps.getParameter<bool>("DoFineCalo");
    eMinFine_ = csps.getParameter<double>("EminFineTrack") * CLHEP::MeV;
    fineNames_ = csps.getParameter<std::vector<std::string> >("FineCaloNames");
    fineLevels_ = csps.getParameter<std::vector<int> >("FineCaloLevels");
    useFines_ = csps.getParameter<std::vector<int> >("UseFineCalo");
    for (auto& level : fineLevels_)
      level += addlevel_;
  }

  void beginRun(const edm::EventSetup& es) final { caloSimPar_ = &es.getData(cspsToken_); }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<CaloTrkProcessing>(iname,
                                                  *caloSimPar_,
                                                  clg,
                                                  testBeam_,
                                                  eMin_,
                                                  putHistory_,
                                                  doFineCalo_,
                                                  eMinFine_,
                                                  addlevel_,
                                                  fineNames_,
                                                  fineLevels_,
                                                  useFines_,
                                                  man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const edm::ESGetToken<CaloSimulationParameters, HcalParametersRcd> cspsToken_;
  const CaloSimulationParameters* caloSimPar_;
  bool testBeam_;
  double eMin_;
  bool putHistory_;
  bool doFineCalo_;
  double eMinFine_;
  int addlevel_;
  std::vector<std::string> fineNames_;
  std::vector<int> fineLevels_;
  std::vector<int> useFines_;
};

DEFINE_SENSITIVEDETECTORBUILDER(CaloTrkProcessingBuilder, CaloTrkProcessing);
