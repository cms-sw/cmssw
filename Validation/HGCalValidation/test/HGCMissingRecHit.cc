/// -*- C++ -*-
//
// Package:    HGCMissingRecHit
// Class:      HGCMissingRecHit
//
/**\class HGCMissingRecHit HGCMissingRecHit.cc Validation/HGCalValidation/test/HGCMissingRecHit.cc

 Description: [one line class summary]

 Implementation:
 	[Notes on implementation]
*/
//
// Original Author:  "Sunanda Banerjee"
//         Created:  Tue September 20 17:55:26 CDT 2022
// $Id$
//
//

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/transform.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <TH1D.h>

class HGCMissingRecHit : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCMissingRecHit(const edm::ParameterSet &);
  ~HGCMissingRecHit() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  typedef std::tuple<float, GlobalPoint> HGCHitTuple;

  void beginJob() override;
  void endJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
  void analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const &simHits,
                          int idet,
                          std::map<unsigned int, HGCHitTuple> &);
  template <class T1>
  void analyzeHGCalDigi(T1 const &theHits, int idet, std::map<unsigned int, HGCHitTuple> const &hitRefs);
  template <class T1>
  void analyzeHGCalRecHit(T1 const &theHits, int idet, std::map<unsigned int, HGCHitTuple> const &hitRefs);

private:
  //HGC Geometry
  const std::vector<std::string> geometrySource_, detectors_;
  const std::vector<int> ietaExcludeBH_;
  const std::vector<edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord>> tok_hgcal_;
  const std::vector<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>> tok_hgcalg_;
  std::vector<const HGCalDDDConstants *> hgcCons_;
  std::vector<const HGCalGeometry *> hgcGeometry_;

  const edm::InputTag eeSimHitSource, fhSimHitSource, bhSimHitSource;
  const edm::EDGetTokenT<std::vector<PCaloHit>> eeSimHitToken_;
  const edm::EDGetTokenT<std::vector<PCaloHit>> fhSimHitToken_;
  const edm::EDGetTokenT<std::vector<PCaloHit>> bhSimHitToken_;
  const edm::InputTag eeDigiSource, fhDigiSource, bhDigiSource;
  const edm::EDGetTokenT<HGCalDigiCollection> eeDigiToken_;
  const edm::EDGetTokenT<HGCalDigiCollection> fhDigiToken_;
  const edm::EDGetTokenT<HGCalDigiCollection> bhDigiToken_;
  const edm::InputTag eeRecHitSource, fhRecHitSource, bhRecHitSource;
  const edm::EDGetTokenT<HGCeeRecHitCollection> eeRecHitToken_;
  const edm::EDGetTokenT<HGChefRecHitCollection> fhRecHitToken_;
  const edm::EDGetTokenT<HGChebRecHitCollection> bhRecHitToken_;

  std::vector<TH1D *> goodHitsDE_, missedHitsDE_, goodHitsDT_, missedHitsDT_;
  std::vector<TH1D *> goodHitsRE_, missedHitsRE_, goodHitsRT_, missedHitsRT_;
};

HGCMissingRecHit::HGCMissingRecHit(const edm::ParameterSet &cfg)
    : geometrySource_(cfg.getParameter<std::vector<std::string>>("geometrySource")),
      detectors_(cfg.getParameter<std::vector<std::string>>("detectors")),
      ietaExcludeBH_(cfg.getParameter<std::vector<int>>("ietaExcludeBH")),
      tok_hgcal_{
          edm::vector_transform(geometrySource_,
                                [this](const std::string &name) {
                                  return esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
                                      edm::ESInputTag{"", name});
                                })},
      tok_hgcalg_{edm::vector_transform(
          geometrySource_,
          [this](const std::string &name) {
            return esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name});
          })},
      eeSimHitSource(cfg.getParameter<edm::InputTag>("eeSimHitSource")),
      fhSimHitSource(cfg.getParameter<edm::InputTag>("fhSimHitSource")),
      bhSimHitSource(cfg.getParameter<edm::InputTag>("bhSimHitSource")),
      eeSimHitToken_(consumes<std::vector<PCaloHit>>(eeSimHitSource)),
      fhSimHitToken_(consumes<std::vector<PCaloHit>>(fhSimHitSource)),
      bhSimHitToken_(consumes<std::vector<PCaloHit>>(bhSimHitSource)),
      eeDigiSource(cfg.getParameter<edm::InputTag>("eeDigiSource")),
      fhDigiSource(cfg.getParameter<edm::InputTag>("fhDigiSource")),
      bhDigiSource(cfg.getParameter<edm::InputTag>("bhDigiSource")),
      eeDigiToken_(consumes<HGCalDigiCollection>(eeDigiSource)),
      fhDigiToken_(consumes<HGCalDigiCollection>(fhDigiSource)),
      bhDigiToken_(consumes<HGCalDigiCollection>(bhDigiSource)),
      eeRecHitSource(cfg.getParameter<edm::InputTag>("eeRecHitSource")),
      fhRecHitSource(cfg.getParameter<edm::InputTag>("fhRecHitSource")),
      bhRecHitSource(cfg.getParameter<edm::InputTag>("bhRecHitSource")),
      eeRecHitToken_(consumes<HGCeeRecHitCollection>(eeRecHitSource)),
      fhRecHitToken_(consumes<HGChefRecHitCollection>(fhRecHitSource)),
      bhRecHitToken_(consumes<HGChebRecHitCollection>(bhRecHitSource)) {
  usesResource(TFileService::kSharedResource);

  edm::LogVerbatim("HGCalValid") << "Use " << geometrySource_.size() << " Geometry sources";
  for (unsigned int k = 0; k < geometrySource_.size(); k++)
    edm::LogVerbatim("HGCalValid") << "  " << detectors_[k] << ":" << geometrySource_[k];
  edm::LogVerbatim("HGCalValid") << "SimHit labels: " << eeSimHitSource << "  " << fhSimHitSource << "  "
                                 << bhSimHitSource;
  edm::LogVerbatim("HGCalValid") << "Digi labels: " << eeDigiSource << "  " << fhDigiSource << "  " << bhDigiSource;
  edm::LogVerbatim("HGCalValid") << "RecHit labels: " << eeRecHitSource << "  " << fhRecHitSource << "  "
                                 << bhRecHitSource;
  edm::LogVerbatim("HGCalValid") << "Exclude the following " << ietaExcludeBH_.size() << " ieta values from BH plots";
  for (unsigned int k = 0; k < ietaExcludeBH_.size(); ++k)
    edm::LogVerbatim("HGCalValid") << " [" << k << "] " << ietaExcludeBH_[k];
}

void HGCMissingRecHit::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> sources = {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"};
  std::vector<std::string> names = {"EE", "HE Silicon", "HE Scintillator"};
  std::vector<int> etas;
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("geometrySource", sources);
  desc.add<std::vector<std::string>>("detectors", names);
  desc.add<edm::InputTag>("eeSimHitSource", edm::InputTag("g4SimHits", "HGCHitsEE"));
  desc.add<edm::InputTag>("fhSimHitSource", edm::InputTag("g4SimHits", "HGCHitsHEfront"));
  desc.add<edm::InputTag>("bhSimHitSource", edm::InputTag("g4SimHits", "HGCHitsHEback"));
  desc.add<edm::InputTag>("eeDigiSource", edm::InputTag("simHGCalUnsuppressedDigis", "EE"));
  desc.add<edm::InputTag>("fhDigiSource", edm::InputTag("simHGCalUnsuppressedDigis", "HEfront"));
  desc.add<edm::InputTag>("bhDigiSource", edm::InputTag("simHGCalUnsuppressedDigis", "HEback"));
  desc.add<edm::InputTag>("eeRecHitSource", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("fhRecHitSource", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("bhRecHitSource", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  desc.add<std::vector<int>>("ietaExcludeBH", etas);
  descriptions.add("hgcMissingRecHit", desc);
}

void HGCMissingRecHit::beginJob() {
  //initiating fileservice
  edm::Service<TFileService> fs;
  for (unsigned int k = 0; k < detectors_.size(); ++k) {
    char name[50], title[100];
    sprintf(name, "GoodDE%s", geometrySource_[k].c_str());
    sprintf(title, "SimHit energy present among Digis in %s", detectors_[k].c_str());
    goodHitsDE_.emplace_back(fs->make<TH1D>(name, title, 1000, 0, 0.01));
    goodHitsDE_.back()->Sumw2();
    sprintf(name, "MissDE%s", geometrySource_[k].c_str());
    sprintf(title, "SimHit energy absent among Digis in %s", detectors_[k].c_str());
    missedHitsDE_.emplace_back(fs->make<TH1D>(name, title, 1000, 0, 0.01));
    missedHitsDE_.back()->Sumw2();
    sprintf(name, "GoodDT%s", geometrySource_[k].c_str());
    sprintf(title, "|#eta| of SimHits present among Digis in %s", detectors_[k].c_str());
    goodHitsDT_.emplace_back(fs->make<TH1D>(name, title, 320, 2.7, 3.1));
    goodHitsDT_.back()->Sumw2();
    sprintf(name, "MissDT%s", geometrySource_[k].c_str());
    sprintf(title, "|#eta| of SimHits absent among Digis in %s", detectors_[k].c_str());
    missedHitsDT_.emplace_back(fs->make<TH1D>(name, title, 320, 2.7, 3.1));
    missedHitsDT_.back()->Sumw2();
    sprintf(name, "GoodRE%s", geometrySource_[k].c_str());
    sprintf(title, "SimHit energy present among RecHits in %s", detectors_[k].c_str());
    goodHitsRE_.emplace_back(fs->make<TH1D>(name, title, 1000, 0, 0.01));
    goodHitsRE_.back()->Sumw2();
    sprintf(name, "MissRE%s", geometrySource_[k].c_str());
    sprintf(title, "SimHit energy absent among RecHits in %s", detectors_[k].c_str());
    missedHitsRE_.emplace_back(fs->make<TH1D>(name, title, 1000, 0, 0.01));
    missedHitsRE_.back()->Sumw2();
    sprintf(name, "GoodRT%s", geometrySource_[k].c_str());
    sprintf(title, "|#eta| of SimHits present among RecHits in %s", detectors_[k].c_str());
    goodHitsRT_.emplace_back(fs->make<TH1D>(name, title, 320, 2.7, 3.1));
    goodHitsRT_.back()->Sumw2();
    sprintf(name, "MissRT%s", geometrySource_[k].c_str());
    sprintf(title, "|#eta| of SimHits absent among RecHits in %s", detectors_[k].c_str());
    missedHitsRT_.emplace_back(fs->make<TH1D>(name, title, 320, 2.7, 3.1));
    missedHitsRT_.back()->Sumw2();
  }
}

void HGCMissingRecHit::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  for (size_t i = 0; i < geometrySource_.size(); i++) {
    edm::LogVerbatim("HGCalValid") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << i;
    const edm::ESHandle<HGCalDDDConstants> &hgcCons = iSetup.getHandle(tok_hgcal_[i]);
    if (hgcCons.isValid()) {
      hgcCons_.push_back(hgcCons.product());
    } else {
      edm::LogWarning("HGCalValid") << "Cannot initiate HGCalDDDConstants for " << geometrySource_[i] << std::endl;
    }
    const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcalg_[i]);
    if (hgcGeom.isValid()) {
      hgcGeometry_.push_back(hgcGeom.product());
    } else {
      edm::LogWarning("HGCalValid") << "Cannot initiate HGCalGeometry for " << geometrySource_[i] << std::endl;
    }
  }
}

void HGCMissingRecHit::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::map<unsigned int, HGCHitTuple> eeHitRefs, fhHitRefs, bhHitRefs;

  //Accesing ee simhits
  const edm::Handle<std::vector<PCaloHit>> &eeSimHits = iEvent.getHandle(eeSimHitToken_);
  if (eeSimHits.isValid()) {
    analyzeHGCalSimHit(eeSimHits, 0, eeHitRefs);
    for (std::map<unsigned int, HGCHitTuple>::iterator itr = eeHitRefs.begin(); itr != eeHitRefs.end(); ++itr) {
      int idx = std::distance(eeHitRefs.begin(), itr);
      edm::LogVerbatim("HGCalValid") << "EEHit[" << idx << "] " << std::hex << itr->first << std::dec << "; Energy "
                                     << std::get<0>(itr->second) << "; Position = " << std::get<1>(itr->second) << ")";
    }
  } else {
    edm::LogWarning("HGCalValid") << "No EE SimHit Found " << std::endl;
  }

  //Accesing fh simhits
  const edm::Handle<std::vector<PCaloHit>> &fhSimHits = iEvent.getHandle(fhSimHitToken_);
  if (fhSimHits.isValid()) {
    analyzeHGCalSimHit(fhSimHits, 1, fhHitRefs);
    for (std::map<unsigned int, HGCHitTuple>::iterator itr = fhHitRefs.begin(); itr != fhHitRefs.end(); ++itr) {
      int idx = std::distance(fhHitRefs.begin(), itr);
      edm::LogVerbatim("HGCalValid") << "FHHit[" << idx << "] " << std::hex << itr->first << std::dec << "; Energy "
                                     << std::get<0>(itr->second) << "; Position = " << std::get<1>(itr->second) << ")";
    }
  } else {
    edm::LogWarning("HGCalValid") << "No FH SimHit Found " << std::endl;
  }

  //Accessing bh simhits
  const edm::Handle<std::vector<PCaloHit>> &bhSimHits = iEvent.getHandle(bhSimHitToken_);
  if (bhSimHits.isValid()) {
    analyzeHGCalSimHit(bhSimHits, 2, bhHitRefs);
    for (std::map<unsigned int, HGCHitTuple>::iterator itr = bhHitRefs.begin(); itr != bhHitRefs.end(); ++itr) {
      int idx = std::distance(bhHitRefs.begin(), itr);
      edm::LogVerbatim("HGCalValid") << "BHHit[" << idx << "] " << std::hex << itr->first << std::dec << "; Energy "
                                     << std::get<0>(itr->second) << "; Position = (" << std::get<1>(itr->second) << ")";
    }
  } else {
    edm::LogWarning("HGCalValid") << "No BH SimHit Found " << std::endl;
  }

  //accessing EE Digi information
  const edm::Handle<HGCalDigiCollection> &eeDigi = iEvent.getHandle(eeDigiToken_);
  if (eeDigi.isValid()) {
    const HGCalDigiCollection *theHits = (eeDigi.product());
    analyzeHGCalDigi(theHits, 0, eeHitRefs);
  } else {
    edm::LogWarning("HGCalValid") << "No EE Digi Found " << std::endl;
  }

  //accessing FH Digi information
  const edm::Handle<HGCalDigiCollection> &fhDigi = iEvent.getHandle(fhDigiToken_);
  if (fhDigi.isValid()) {
    const HGCalDigiCollection *theHits = (fhDigi.product());
    analyzeHGCalDigi(theHits, 1, fhHitRefs);
  } else {
    edm::LogWarning("HGCalValid") << "No FH Digi Found " << std::endl;
  }

  //accessing BH Digi information
  const edm::Handle<HGCalDigiCollection> &bhDigi = iEvent.getHandle(bhDigiToken_);
  if (bhDigi.isValid()) {
    const HGCalDigiCollection *theHits = (bhDigi.product());
    analyzeHGCalDigi(theHits, 2, bhHitRefs);
  } else {
    edm::LogWarning("HGCalValid") << "No BH Digi Found " << std::endl;
  }

  //accessing EE Rechit information
  const edm::Handle<HGCeeRecHitCollection> &eeRecHit = iEvent.getHandle(eeRecHitToken_);
  if (eeRecHit.isValid()) {
    const HGCeeRecHitCollection *theHits = (eeRecHit.product());
    analyzeHGCalRecHit(theHits, 0, eeHitRefs);
  } else {
    edm::LogWarning("HGCalValid") << "No EE RecHit Found " << std::endl;
  }

  //accessing FH Rechit information
  const edm::Handle<HGChefRecHitCollection> &fhRecHit = iEvent.getHandle(fhRecHitToken_);
  if (fhRecHit.isValid()) {
    const HGChefRecHitCollection *theHits = (fhRecHit.product());
    analyzeHGCalRecHit(theHits, 1, fhHitRefs);
  } else {
    edm::LogWarning("HGCalValid") << "No FH RecHit Found " << std::endl;
  }

  //accessing BH Rechit information
  const edm::Handle<HGChebRecHitCollection> &bhRecHit = iEvent.getHandle(bhRecHitToken_);
  if (bhRecHit.isValid()) {
    const HGChebRecHitCollection *theHits = (bhRecHit.product());
    analyzeHGCalRecHit(theHits, 2, bhHitRefs);
  } else {
    edm::LogWarning("HGCalValid") << "No BH RecHit Found " << std::endl;
  }
}

void HGCMissingRecHit::analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const &simHits,
                                          int idet,
                                          std::map<unsigned int, HGCHitTuple> &hitRefs) {
  for (auto const &simHit : *simHits) {
    DetId id(simHit.id());
    bool ok(true), valid2(false);
    GlobalPoint p;
    bool valid1 = (hgcGeometry_[idet]->topology()).valid(id);
    std::ostringstream st1;
    if (id.det() == DetId::HGCalHSc) {
      st1 << HGCScintillatorDetId(id);
    } else if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
      st1 << HGCSiliconDetId(id);
    } else {
      st1 << "Not a Standard One";
    }
    if ((hgcCons_[idet]->waferHexagon8()) && ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi))) {
      p = hgcGeometry_[idet]->getPosition(id);
      HGCSiliconDetId hid(id);
      valid2 = hgcCons_[idet]->isValidHex8(hid.layer(), hid.waferU(), hid.waferV(), hid.cellU(), hid.cellV(), true);
    } else if ((hgcCons_[idet]->tileTrapezoid()) && (id.det() == DetId::HGCalHSc)) {
      p = hgcGeometry_[idet]->getPosition(id);
      HGCScintillatorDetId hid(id);
      valid2 = hgcCons_[idet]->isValidTrap(hid.zside(), hid.layer(), hid.ring(), hid.iphi());
      edm::LogVerbatim("HGCalGeom") << "Scint " << HGCScintillatorDetId(id) << " position (" << p.x() << ", " << p.y()
                                    << ", " << p.z() << ") " << valid1 << ":" << valid2;
    } else {
      // This is an invalid cell
      ok = false;
      edm::LogVerbatim("HGCalError") << "Hit " << std::hex << id.rawId() << std::dec << " " << st1.str()
                                     << " in the wrong collection for detector " << idet << ":" << geometrySource_[idet]
                                     << " ***** ERROR *****";
    }

    edm::LogVerbatim("HGCalValid") << "SimHit: " << std::hex << id.rawId() << std::dec << " " << st1.str() << " Flag "
                                   << ok;

    if (ok) {
      float energy = simHit.energy();

      float energySum(energy);
      if (hitRefs.count(id) != 0)
        energySum += std::get<0>(hitRefs[id]);
      hitRefs[id] = std::make_tuple(energySum, p);
      edm::LogVerbatim("HGCalValid") << "Position = " << p << " Energy " << simHit.energy() << ":" << energySum;
    }
    if ((!valid1) || (!valid2))
      edm::LogVerbatim("HGCalMiss") << "Invalid SimHit " << st1.str() << " position = " << p << " perp " << p.perp()
                                    << " Validity flags " << valid1 << ":" << valid2;
  }
}

template <class T1>
void HGCMissingRecHit::analyzeHGCalDigi(T1 const &theHits,
                                        int idet,
                                        std::map<unsigned int, HGCHitTuple> const &hitRefs) {
  std::vector<unsigned int> ids;
  for (auto it = theHits->begin(); it != theHits->end(); ++it) {
    ids.emplace_back((it->id().rawId()));
    if (!(hgcGeometry_[idet]->topology()).valid(it->id())) {
      std::ostringstream st1;
      if ((it->id()).det() == DetId::HGCalHSc)
        st1 << HGCScintillatorDetId(it->id());
      else
        st1 << HGCSiliconDetId(it->id());
      edm::LogVerbatim("HGCalError") << "Invalid Hit " << st1.str();
    }
  }
  for (auto it = hitRefs.begin(); it != hitRefs.end(); ++it) {
    double eta = std::get<1>(it->second).eta();
    auto itr = std::find(ids.begin(), ids.end(), it->first);
    if (itr == ids.end()) {
      bool ok = (hgcGeometry_[idet]->topology()).valid(DetId(it->first));
      missedHitsDE_[idet]->Fill(std::get<0>(it->second));
      missedHitsDT_[idet]->Fill(eta);
      std::ostringstream st1;
      if (DetId(it->first).det() == DetId::HGCalHSc)
        st1 << HGCScintillatorDetId(it->first);
      else
        st1 << HGCSiliconDetId(it->first);
      edm::LogVerbatim("HGCalMiss") << "Hit: " << std::hex << (it->first) << std::dec << " " << st1.str()
                                    << " SimHit (E = " << std::get<0>(it->second)
                                    << ", Position = " << std::get<1>(it->second) << ") Valid " << ok
                                    << " is missing in the Digi collection";
    } else {
      goodHitsDE_[idet]->Fill(std::get<0>(it->second));
      goodHitsDT_[idet]->Fill(eta);
    }
  }
}

template <class T1>
void HGCMissingRecHit::analyzeHGCalRecHit(T1 const &theHits,
                                          int idet,
                                          std::map<unsigned int, HGCHitTuple> const &hitRefs) {
  std::vector<unsigned int> ids;
  for (auto it = theHits->begin(); it != theHits->end(); ++it) {
    ids.emplace_back((it->id().rawId()));
    if (!(hgcGeometry_[idet]->topology()).valid(it->id())) {
      std::ostringstream st1;
      if ((it->id()).det() == DetId::HGCalHSc)
        st1 << HGCScintillatorDetId(it->id());
      else
        st1 << HGCSiliconDetId(it->id());
      edm::LogVerbatim("HGCalError") << "Invalid Hit " << st1.str();
    }
  }
  for (auto it = hitRefs.begin(); it != hitRefs.end(); ++it) {
    double eta = std::get<1>(it->second).eta();
    auto itr = std::find(ids.begin(), ids.end(), it->first);
    if (itr == ids.end()) {
      bool ok = (hgcGeometry_[idet]->topology()).valid(DetId(it->first));
      missedHitsRE_[idet]->Fill(std::get<0>(it->second));
      missedHitsRT_[idet]->Fill(eta);
      std::ostringstream st1;
      if (DetId(it->first).det() == DetId::HGCalHSc)
        st1 << HGCScintillatorDetId(it->first);
      else
        st1 << HGCSiliconDetId(it->first);
      edm::LogVerbatim("HGCalMiss") << "Hit: " << std::hex << (it->first) << std::dec << " " << st1.str()
                                    << " SimHit (E = " << std::get<0>(it->second)
                                    << ", Position = " << std::get<1>(it->second) << ") Valid " << ok
                                    << " is missing in the RecHit collection";
    } else {
      goodHitsRE_[idet]->Fill(std::get<0>(it->second));
      goodHitsRT_[idet]->Fill(eta);
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCMissingRecHit);
