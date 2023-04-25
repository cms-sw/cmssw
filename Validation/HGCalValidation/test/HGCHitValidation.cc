/// -*- C++ -*-
//
// Package:    HGCHitValidation
// Class:      HGCHitValidation
//
/**\class HGCHitValidation HGCHitValidation.cc Validation/HGCalValidation/test/HGCHitValidation.cc

 Description: [one line class summary]

 Implementation:
 	[Notes on implementation]
*/
//
// Original Author:  "Maksat Haytmyradov"
//         Created:  Fri March 19 13:32:26 CDT 2016
// $Id$
//
//

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/transform.h"

#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include <TH1.h>
#include <TH2.h>
#include <TTree.h>
#include <TVector3.h>
#include <TSystem.h>
#include <TFile.h>

#include <cmath>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

class HGCHitValidation : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCHitValidation(const edm::ParameterSet &);
  ~HGCHitValidation() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  typedef std::tuple<float, float, float, float> HGCHitTuple;

  void beginJob() override;
  void endJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
  void analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const &simHits,
                          int idet,
                          TH1F *,
                          std::map<unsigned int, HGCHitTuple> &);
  template <class T1>
  void analyzeHGCalRecHit(T1 const &theHits, std::map<unsigned int, HGCHitTuple> const &hitRefs);

private:
  //HGC Geometry
  const std::vector<std::string> geometrySource_;
  const std::vector<int> ietaExcludeBH_;
  const bool makeTree_;
  const std::vector<edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord>> tok_hgcal_;
  const std::vector<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>> tok_hgcalg_;
  std::vector<const HGCalDDDConstants *> hgcCons_;
  std::vector<const HGCalGeometry *> hgcGeometry_;

  const edm::InputTag eeSimHitSource, fhSimHitSource, bhSimHitSource;
  const edm::EDGetTokenT<std::vector<PCaloHit>> eeSimHitToken_;
  const edm::EDGetTokenT<std::vector<PCaloHit>> fhSimHitToken_;
  const edm::EDGetTokenT<std::vector<PCaloHit>> bhSimHitToken_;
  const edm::InputTag eeRecHitSource, fhRecHitSource, bhRecHitSource;
  const edm::EDGetTokenT<HGCeeRecHitCollection> eeRecHitToken_;
  const edm::EDGetTokenT<HGChefRecHitCollection> fhRecHitToken_;
  const edm::EDGetTokenT<HGChebRecHitCollection> bhRecHitToken_;

  TTree *hgcHits_;
  std::vector<float> *heeRecX_, *heeRecY_, *heeRecZ_, *heeRecEnergy_;
  std::vector<float> *hefRecX_, *hefRecY_, *hefRecZ_, *hefRecEnergy_;
  std::vector<float> *hebRecX_, *hebRecY_, *hebRecZ_, *hebRecEnergy_;
  std::vector<float> *heeSimX_, *heeSimY_, *heeSimZ_, *heeSimEnergy_;
  std::vector<float> *hefSimX_, *hefSimY_, *hefSimZ_, *hefSimEnergy_;
  std::vector<float> *hebSimX_, *hebSimY_, *hebSimZ_, *hebSimEnergy_;
  std::vector<float> *hebSimEta_, *hebRecEta_, *hebSimPhi_, *hebRecPhi_;
  std::vector<unsigned int> *heeDetID_, *hefDetID_, *hebDetID_;

  TH2F *heedzVsZ_, *heedyVsY_, *heedxVsX_;
  TH2F *hefdzVsZ_, *hefdyVsY_, *hefdxVsX_;
  TH2F *hebdzVsZ_, *hebdPhiVsPhi_, *hebdEtaVsEta_;
  TH2F *heeRecVsSimZ_, *heeRecVsSimY_, *heeRecVsSimX_;
  TH2F *hefRecVsSimZ_, *hefRecVsSimY_, *hefRecVsSimX_;
  TH2F *hebRecVsSimZ_, *hebRecVsSimY_, *hebRecVsSimX_;
  TH2F *heeEnSimRec_, *hefEnSimRec_, *hebEnSimRec_;
  TH1F *hebEnRec_, *hebEnSim_, *hefEnRec_;
  TH1F *hefEnSim_, *heeEnRec_, *heeEnSim_;
};

HGCHitValidation::HGCHitValidation(const edm::ParameterSet &cfg)
    : geometrySource_(cfg.getUntrackedParameter<std::vector<std::string>>("geometrySource")),
      ietaExcludeBH_(cfg.getParameter<std::vector<int>>("ietaExcludeBH")),
      makeTree_(cfg.getUntrackedParameter<bool>("makeTree", true)),
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
      eeRecHitSource(cfg.getParameter<edm::InputTag>("eeRecHitSource")),
      fhRecHitSource(cfg.getParameter<edm::InputTag>("fhRecHitSource")),
      bhRecHitSource(cfg.getParameter<edm::InputTag>("bhRecHitSource")),
      eeRecHitToken_(consumes<HGCeeRecHitCollection>(eeRecHitSource)),
      fhRecHitToken_(consumes<HGChefRecHitCollection>(fhRecHitSource)),
      bhRecHitToken_(consumes<HGChebRecHitCollection>(bhRecHitSource)) {
  usesResource(TFileService::kSharedResource);

  hgcHits_ = nullptr;
  heeRecX_ = heeRecY_ = heeRecZ_ = heeRecEnergy_ = nullptr;
  hefRecX_ = hefRecY_ = hefRecZ_ = hefRecEnergy_ = nullptr;
  hebRecX_ = hebRecY_ = hebRecZ_ = hebRecEnergy_ = nullptr;
  heeSimX_ = heeSimY_ = heeSimZ_ = heeSimEnergy_ = nullptr;
  hefSimX_ = hefSimY_ = hefSimZ_ = hefSimEnergy_ = nullptr;
  hebSimX_ = hebSimY_ = hebSimZ_ = hebSimEnergy_ = nullptr;
  hebSimEta_ = hebRecEta_ = hebSimPhi_ = hebRecPhi_ = nullptr;
  heeDetID_ = hefDetID_ = hebDetID_ = nullptr;
  heedzVsZ_ = heedyVsY_ = heedxVsX_ = nullptr;
  hefdzVsZ_ = hefdyVsY_ = hefdxVsX_ = nullptr;
  hebdzVsZ_ = hebdPhiVsPhi_ = hebdEtaVsEta_ = nullptr;
  heeRecVsSimZ_ = heeRecVsSimY_ = heeRecVsSimX_ = nullptr;
  hefRecVsSimZ_ = hefRecVsSimY_ = hefRecVsSimX_ = nullptr;
  hebRecVsSimZ_ = hebRecVsSimY_ = hebRecVsSimX_ = nullptr;
  heeEnSimRec_ = hefEnSimRec_ = hebEnSimRec_ = nullptr;
  hebEnRec_ = hebEnSim_ = hefEnRec_ = nullptr;
  hefEnSim_ = heeEnRec_ = heeEnSim_ = nullptr;

  edm::LogVerbatim("HGCalValid") << "MakeTree Flag set to " << makeTree_ << " and use " << geometrySource_.size()
                                 << " Geometry sources";
  for (auto const &s : geometrySource_)
    edm::LogVerbatim("HGCalValid") << "  " << s;
  edm::LogVerbatim("HGCalValid") << "SimHit labels: " << eeSimHitSource << "  " << fhSimHitSource << "  "
                                 << bhSimHitSource;
  edm::LogVerbatim("HGCalValid") << "RecHit labels: " << eeRecHitSource << "  " << fhRecHitSource << "  "
                                 << bhRecHitSource;
  edm::LogVerbatim("HGCalValid") << "Exclude the following " << ietaExcludeBH_.size() << " ieta values from BH plots";
  for (unsigned int k = 0; k < ietaExcludeBH_.size(); ++k)
    edm::LogVerbatim("HGCalValid") << " [" << k << "] " << ietaExcludeBH_[k];
}

void HGCHitValidation::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"};
  std::vector<int> etas;
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("makeTree", true);
  desc.addUntracked<std::vector<std::string>>("geometrySource", names);
  desc.add<edm::InputTag>("eeSimHitSource", edm::InputTag("g4SimHits", "HGCHitsEE"));
  desc.add<edm::InputTag>("fhSimHitSource", edm::InputTag("g4SimHits", "HGCHitsHEfront"));
  desc.add<edm::InputTag>("bhSimHitSource", edm::InputTag("g4SimHits", "HGCHitsHEback"));
  desc.add<edm::InputTag>("eeRecHitSource", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("fhRecHitSource", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("bhRecHitSource", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  desc.add<std::vector<int>>("ietaExcludeBH", etas);
  descriptions.add("hgcHitAnalysis", desc);
}

void HGCHitValidation::beginJob() {
  //initiating fileservice
  edm::Service<TFileService> fs;
  if (makeTree_) {
    hgcHits_ = fs->make<TTree>("hgcHits", "Hit Collection");
    hgcHits_->Branch("heeRecX", &heeRecX_);
    hgcHits_->Branch("heeRecY", &heeRecY_);
    hgcHits_->Branch("heeRecZ", &heeRecZ_);
    hgcHits_->Branch("heeRecEnergy", &heeRecEnergy_);
    hgcHits_->Branch("hefRecX", &hefRecX_);
    hgcHits_->Branch("hefRecY", &hefRecY_);
    hgcHits_->Branch("hefRecZ", &hefRecZ_);
    hgcHits_->Branch("hefRecEnergy", &hefRecEnergy_);
    hgcHits_->Branch("hebRecX", &hebRecX_);
    hgcHits_->Branch("hebRecY", &hebRecY_);
    hgcHits_->Branch("hebRecZ", &hebRecZ_);
    hgcHits_->Branch("hebRecEta", &hebRecEta_);
    hgcHits_->Branch("hebRecPhi", &hebRecPhi_);
    hgcHits_->Branch("hebRecEnergy", &hebRecEnergy_);

    hgcHits_->Branch("heeSimX", &heeSimX_);
    hgcHits_->Branch("heeSimY", &heeSimY_);
    hgcHits_->Branch("heeSimZ", &heeSimZ_);
    hgcHits_->Branch("heeSimEnergy", &heeSimEnergy_);
    hgcHits_->Branch("hefSimX", &hefSimX_);
    hgcHits_->Branch("hefSimY", &hefSimY_);
    hgcHits_->Branch("hefSimZ", &hefSimZ_);
    hgcHits_->Branch("hefSimEnergy", &hefSimEnergy_);
    hgcHits_->Branch("hebSimX", &hebSimX_);
    hgcHits_->Branch("hebSimY", &hebSimY_);
    hgcHits_->Branch("hebSimZ", &hebSimZ_);
    hgcHits_->Branch("hebSimEta", &hebSimEta_);
    hgcHits_->Branch("hebSimPhi", &hebSimPhi_);
    hgcHits_->Branch("hebSimEnergy", &hebSimEnergy_);

    hgcHits_->Branch("heeDetID", &heeDetID_);
    hgcHits_->Branch("hefDetID", &hefDetID_);
    hgcHits_->Branch("hebDetID", &hebDetID_);
  } else {
    heedzVsZ_ = fs->make<TH2F>("heedzVsZ", "", 7200, -360, 360, 100, -0.1, 0.1);
    heedyVsY_ = fs->make<TH2F>("heedyVsY", "", 400, -200, 200, 100, -0.02, 0.02);
    heedxVsX_ = fs->make<TH2F>("heedxVsX", "", 400, -200, 200, 100, -0.02, 0.02);
    heeRecVsSimZ_ = fs->make<TH2F>("heeRecVsSimZ", "", 7200, -360, 360, 7200, -360, 360);
    heeRecVsSimY_ = fs->make<TH2F>("heeRecVsSimY", "", 400, -200, 200, 400, -200, 200);
    heeRecVsSimX_ = fs->make<TH2F>("heeRecVsSimX", "", 400, -200, 200, 400, -200, 200);
    hefdzVsZ_ = fs->make<TH2F>("hefdzVsZ", "", 8200, -410, 410, 100, -0.1, 0.1);
    hefdyVsY_ = fs->make<TH2F>("hefdyVsY", "", 400, -200, 200, 100, -0.02, 0.02);
    hefdxVsX_ = fs->make<TH2F>("hefdxVsX", "", 400, -200, 200, 100, -0.02, 0.02);
    hefRecVsSimZ_ = fs->make<TH2F>("hefRecVsSimZ", "", 8200, -410, 410, 8200, -410, 410);
    hefRecVsSimY_ = fs->make<TH2F>("hefRecVsSimY", "", 400, -200, 200, 400, -200, 200);
    hefRecVsSimX_ = fs->make<TH2F>("hefRecVsSimX", "", 400, -200, 200, 400, -200, 200);
    hebdzVsZ_ = fs->make<TH2F>("hebdzVsZ", "", 1080, -540, 540, 100, -1.0, 1.0);
    hebdPhiVsPhi_ = fs->make<TH2F>("hebdPhiVsPhi", "", M_PI * 100, -0.5, M_PI + 0.5, 200, -0.2, 0.2);
    hebdEtaVsEta_ = fs->make<TH2F>("hebdEtaVsEta", "", 1000, -5, 5, 200, -0.1, 0.1);
    hebRecVsSimZ_ = fs->make<TH2F>("hebRecVsSimZ", "", 1080, -540, 540, 1080, -540, 540);
    hebRecVsSimY_ = fs->make<TH2F>("hebRecVsSimY", "", 400, -200, 200, 400, -200, 200);
    hebRecVsSimX_ = fs->make<TH2F>("hebRecVsSimX", "", 400, -200, 200, 400, -200, 200);
    heeEnRec_ = fs->make<TH1F>("heeEnRec", "", 1000, 0, 10);
    heeEnSim_ = fs->make<TH1F>("heeEnSim", "", 1000, 0, 0.01);
    heeEnSimRec_ = fs->make<TH2F>("heeEnSimRec", "", 200, 0, 0.002, 200, 0, 0.2);
    hefEnRec_ = fs->make<TH1F>("hefEnRec", "", 1000, 0, 10);
    hefEnSim_ = fs->make<TH1F>("hefEnSim", "", 1000, 0, 0.01);
    hefEnSimRec_ = fs->make<TH2F>("hefEnSimRec", "", 200, 0, 0.001, 200, 0, 0.5);
    hebEnRec_ = fs->make<TH1F>("hebEnRec", "", 1000, 0, 15);
    hebEnSim_ = fs->make<TH1F>("hebEnSim", "", 1000, 0, 0.01);
    hebEnSimRec_ = fs->make<TH2F>("hebEnSimRec", "", 200, 0, 0.02, 200, 0, 4);
  }
}

void HGCHitValidation::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  for (size_t i = 0; i < geometrySource_.size(); i++) {
    edm::LogVerbatim("HGCalValid") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << i;
    edm::ESHandle<HGCalDDDConstants> hgcCons = iSetup.getHandle(tok_hgcal_[i]);
    if (hgcCons.isValid()) {
      hgcCons_.push_back(hgcCons.product());
    } else {
      edm::LogWarning("HGCalValid") << "Cannot initiate HGCalDDDConstants for " << geometrySource_[i] << std::endl;
    }
    edm::ESHandle<HGCalGeometry> hgcGeom = iSetup.getHandle(tok_hgcalg_[i]);
    if (hgcGeom.isValid()) {
      hgcGeometry_.push_back(hgcGeom.product());
    } else {
      edm::LogWarning("HGCalValid") << "Cannot initiate HGCalGeometry for " << geometrySource_[i] << std::endl;
    }
  }
}

void HGCHitValidation::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::map<unsigned int, HGCHitTuple> eeHitRefs, fhHitRefs, bhHitRefs;

  //Accesing ee simhits
  const edm::Handle<std::vector<PCaloHit>> &eeSimHits = iEvent.getHandle(eeSimHitToken_);

  if (eeSimHits.isValid()) {
    analyzeHGCalSimHit(eeSimHits, 0, heeEnSim_, eeHitRefs);
    for (std::map<unsigned int, HGCHitTuple>::iterator itr = eeHitRefs.begin(); itr != eeHitRefs.end(); ++itr) {
      int idx = std::distance(eeHitRefs.begin(), itr);
      edm::LogVerbatim("HGCalValid") << "EEHit[" << idx << "] " << std::hex << itr->first << std::dec << "; Energy "
                                     << std::get<0>(itr->second) << "; Position"
                                     << " (" << std::get<1>(itr->second) << ", " << std::get<2>(itr->second) << ", "
                                     << std::get<3>(itr->second) << ")";
    }
  } else {
    edm::LogWarning("HGCalValid") << "No EE SimHit Found " << std::endl;
  }

  //Accesing fh simhits
  const edm::Handle<std::vector<PCaloHit>> &fhSimHits = iEvent.getHandle(fhSimHitToken_);
  if (fhSimHits.isValid()) {
    analyzeHGCalSimHit(fhSimHits, 1, hefEnSim_, fhHitRefs);
    for (std::map<unsigned int, HGCHitTuple>::iterator itr = fhHitRefs.begin(); itr != fhHitRefs.end(); ++itr) {
      int idx = std::distance(fhHitRefs.begin(), itr);
      edm::LogVerbatim("HGCalValid") << "FHHit[" << idx << "] " << std::hex << itr->first << std::dec << "; Energy "
                                     << std::get<0>(itr->second) << "; Position"
                                     << " (" << std::get<1>(itr->second) << ", " << std::get<2>(itr->second) << ", "
                                     << std::get<3>(itr->second) << ")";
    }
  } else {
    edm::LogWarning("HGCalValid") << "No FH SimHit Found " << std::endl;
  }

  //Accessing bh simhits
  const edm::Handle<std::vector<PCaloHit>> &bhSimHits = iEvent.getHandle(bhSimHitToken_);
  if (bhSimHits.isValid()) {
    analyzeHGCalSimHit(bhSimHits, 2, hebEnSim_, bhHitRefs);
    for (std::map<unsigned int, HGCHitTuple>::iterator itr = bhHitRefs.begin(); itr != bhHitRefs.end(); ++itr) {
      int idx = std::distance(bhHitRefs.begin(), itr);
      edm::LogVerbatim("HGCalValid") << "BHHit[" << idx << "] " << std::hex << itr->first << std::dec << "; Energy "
                                     << std::get<0>(itr->second) << "; Position (" << std::get<1>(itr->second) << ", "
                                     << std::get<2>(itr->second) << ", " << std::get<3>(itr->second) << ")";
    }
  } else {
    edm::LogWarning("HGCalValid") << "No BH SimHit Found " << std::endl;
  }

  //accessing EE Rechit information
  const edm::Handle<HGCeeRecHitCollection> &eeRecHit = iEvent.getHandle(eeRecHitToken_);
  if (eeRecHit.isValid()) {
    const HGCeeRecHitCollection *theHits = (eeRecHit.product());
    for (auto it = theHits->begin(); it != theHits->end(); ++it) {
      double energy = it->energy();
      if (!makeTree_)
        heeEnRec_->Fill(energy);
      std::map<unsigned int, HGCHitTuple>::const_iterator itr = eeHitRefs.find(it->id().rawId());
      if (itr != eeHitRefs.end()) {
        GlobalPoint xyz = hgcGeometry_[0]->getPosition(it->id());
        if (makeTree_) {
          heeRecX_->push_back(xyz.x());
          heeRecY_->push_back(xyz.y());
          heeRecZ_->push_back(xyz.z());
          heeRecEnergy_->push_back(energy);
          heeSimX_->push_back(std::get<1>(itr->second));
          heeSimY_->push_back(std::get<2>(itr->second));
          heeSimZ_->push_back(std::get<3>(itr->second));
          heeSimEnergy_->push_back(std::get<0>(itr->second));
          heeDetID_->push_back(itr->first);
        } else {
          heeRecVsSimX_->Fill(std::get<1>(itr->second), xyz.x());
          heeRecVsSimY_->Fill(std::get<2>(itr->second), xyz.y());
          heeRecVsSimZ_->Fill(std::get<3>(itr->second), xyz.z());
          heedxVsX_->Fill(std::get<1>(itr->second), (xyz.x() - std::get<1>(itr->second)));
          heedyVsY_->Fill(std::get<2>(itr->second), (xyz.y() - std::get<2>(itr->second)));
          heedzVsZ_->Fill(std::get<3>(itr->second), (xyz.z() - std::get<3>(itr->second)));
          heeEnSimRec_->Fill(std::get<0>(itr->second), energy);
        }
        edm::LogVerbatim("HGCalValid") << "EEHit: " << std::hex << it->id().rawId() << std::dec << " Sim ("
                                       << std::get<0>(itr->second) << ", " << std::get<1>(itr->second) << ", "
                                       << std::get<2>(itr->second) << ", " << std::get<3>(itr->second) << ") Rec ("
                                       << energy << ", " << xyz.x() << ", " << xyz.y() << ", " << xyz.z();
      }
    }
  } else {
    edm::LogWarning("HGCalValid") << "No EE RecHit Found " << std::endl;
  }

  //accessing FH Rechit information
  const edm::Handle<HGChefRecHitCollection> &fhRecHit = iEvent.getHandle(fhRecHitToken_);
  if (fhRecHit.isValid()) {
    const HGChefRecHitCollection *theHits = (fhRecHit.product());
    for (auto it = theHits->begin(); it != theHits->end(); ++it) {
      double energy = it->energy();
      if (!makeTree_)
        hefEnRec_->Fill(energy);
      std::map<unsigned int, HGCHitTuple>::const_iterator itr = fhHitRefs.find(it->id().rawId());
      if (itr != fhHitRefs.end()) {
        GlobalPoint xyz = hgcGeometry_[1]->getPosition(it->id());
        if (makeTree_) {
          hefRecX_->push_back(xyz.x());
          hefRecY_->push_back(xyz.y());
          hefRecZ_->push_back(xyz.z());
          hefRecEnergy_->push_back(energy);
          hefSimX_->push_back(std::get<1>(itr->second));
          hefSimY_->push_back(std::get<2>(itr->second));
          hefSimZ_->push_back(std::get<3>(itr->second));
          hefSimEnergy_->push_back(std::get<0>(itr->second));
          hefDetID_->push_back(itr->first);
        } else {
          hefRecVsSimX_->Fill(std::get<1>(itr->second), xyz.x());
          hefRecVsSimY_->Fill(std::get<2>(itr->second), xyz.y());
          hefRecVsSimZ_->Fill(std::get<3>(itr->second), xyz.z());
          hefdxVsX_->Fill(std::get<1>(itr->second), (xyz.x() - std::get<1>(itr->second)));
          hefdyVsY_->Fill(std::get<2>(itr->second), (xyz.y() - std::get<2>(itr->second)));
          hefdzVsZ_->Fill(std::get<3>(itr->second), (xyz.z() - std::get<3>(itr->second)));
          hefEnSimRec_->Fill(std::get<0>(itr->second), energy);
        }
        edm::LogVerbatim("HGCalValid") << "FHHit: " << std::hex << it->id().rawId() << std::dec << " Sim ("
                                       << std::get<0>(itr->second) << ", " << std::get<1>(itr->second) << ", "
                                       << std::get<2>(itr->second) << ", " << std::get<3>(itr->second) << ") Rec ("
                                       << energy << "," << xyz.x() << ", " << xyz.y() << ", " << xyz.z();
      }
    }
  } else {
    edm::LogWarning("HGCalValid") << "No FH RecHit Found " << std::endl;
  }

  //accessing BH Rechit information
  const edm::Handle<HGChebRecHitCollection> &bhRecHit = iEvent.getHandle(bhRecHitToken_);
  if (bhRecHit.isValid()) {
    const HGChebRecHitCollection *theHits = (bhRecHit.product());
    analyzeHGCalRecHit(theHits, bhHitRefs);
  } else {
    edm::LogWarning("HGCalValid") << "No BH RecHit Found " << std::endl;
  }

  if (makeTree_) {
    hgcHits_->Fill();

    heeRecX_->clear();
    heeRecY_->clear();
    heeRecZ_->clear();
    hefRecX_->clear();
    hefRecY_->clear();
    hefRecZ_->clear();
    hebRecX_->clear();
    hebRecY_->clear();
    hebRecZ_->clear();
    heeRecEnergy_->clear();
    hefRecEnergy_->clear();
    hebRecEnergy_->clear();
    heeSimX_->clear();
    heeSimY_->clear();
    heeSimZ_->clear();
    hefSimX_->clear();
    hefSimY_->clear();
    hefSimZ_->clear();
    hebSimX_->clear();
    hebSimY_->clear();
    hebSimZ_->clear();
    heeSimEnergy_->clear();
    hefSimEnergy_->clear();
    hebSimEnergy_->clear();
    hebSimEta_->clear();
    hebRecEta_->clear();
    hebSimPhi_->clear();
    hebRecPhi_->clear();
    heeDetID_->clear();
    hefDetID_->clear();
    hebDetID_->clear();
  }
}

void HGCHitValidation::analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const &simHits,
                                          int idet,
                                          TH1F *hist,
                                          std::map<unsigned int, HGCHitTuple> &hitRefs) {
  const HGCalTopology &hTopo = hgcGeometry_[idet]->topology();
  for (auto const &simHit : *simHits) {
    unsigned int id = simHit.id();
    std::pair<float, float> xy;
    bool ok(true);
    int subdet(0), zside, layer, wafer, celltype, cell, wafer2(0), cell2(0);
    if (hgcCons_[idet]->waferHexagon8()) {
      HGCSiliconDetId detId = HGCSiliconDetId(id);
      subdet = (int)(detId.det());
      cell = detId.cellU();
      cell2 = detId.cellV();
      wafer = detId.waferU();
      wafer2 = detId.waferV();
      celltype = detId.type();
      layer = detId.layer();
      zside = detId.zside();
      xy = hgcCons_[idet]->locateCell(zside, layer, wafer, wafer2, cell, cell2, false, true, false, false);
    } else if (hgcCons_[idet]->tileTrapezoid()) {
      HGCScintillatorDetId detId = HGCScintillatorDetId(id);
      subdet = (int)(detId.det());
      cell = detId.ietaAbs();
      wafer = detId.iphi();
      celltype = detId.type();
      layer = detId.layer();
      zside = detId.zside();
      xy = hgcCons_[idet]->locateCellTrap(zside, layer, wafer, cell, false, false);
    } else {
      HGCalTestNumbering::unpackHexagonIndex(simHit.id(), subdet, zside, layer, wafer, celltype, cell);
      xy = hgcCons_[idet]->locateCell(cell, layer, wafer, false);

      //skip this hit if after ganging it is not valid
      std::pair<int, int> recoLayerCell = hgcCons_[idet]->simToReco(cell, layer, wafer, hTopo.detectorType());
      ok = !(recoLayerCell.second < 0 || recoLayerCell.first < 0);
      id = HGCalDetId((ForwardSubdetector)(subdet), zside, layer, celltype, wafer, cell).rawId();
    }

    edm::LogVerbatim("HGCalValid") << "SimHit: " << std::hex << id << std::dec << " (" << subdet << ":" << zside << ":"
                                   << layer << ":" << celltype << ":" << wafer << ":" << wafer2 << ":" << cell << ":"
                                   << cell2 << ") Flag " << ok;

    if (ok) {
      float zp = hgcCons_[idet]->waferZ(layer, false);
      if (zside < 0)
        zp = -zp;
      float xp = (zside < 0) ? -xy.first / 10 : xy.first / 10;
      float yp = xy.second / 10.0;
      float energy = simHit.energy();

      float energySum(energy);
      if (hitRefs.count(id) != 0)
        energySum += std::get<0>(hitRefs[id]);
      hitRefs[id] = std::make_tuple(energySum, xp, yp, zp);
      if (hist != nullptr)
        hist->Fill(energy);
      edm::LogVerbatim("HGCalValid") << "Position (" << xp << ", " << yp << ", " << zp << ") "
                                     << " Energy " << simHit.energy() << ":" << energySum;
    }
  }
}

template <class T1>
void HGCHitValidation::analyzeHGCalRecHit(T1 const &theHits, std::map<unsigned int, HGCHitTuple> const &hitRefs) {
  for (auto it = theHits->begin(); it != theHits->end(); ++it) {
    DetId id = it->id();
    double energy = it->energy();
    if (!makeTree_)
      hebEnRec_->Fill(energy);
    GlobalPoint xyz = hgcGeometry_[2]->getPosition(id);

    std::map<unsigned int, HGCHitTuple>::const_iterator itr = hitRefs.find(id.rawId());
    if (itr != hitRefs.end()) {
      float ang3 = xyz.phi().value();  // returns the phi in radians
      double fac = sinh(std::get<1>(itr->second));
      double pT = std::get<3>(itr->second) / fac;
      double xp = pT * cos(std::get<2>(itr->second));
      double yp = pT * sin(std::get<2>(itr->second));
      if (makeTree_) {
        hebRecX_->push_back(xyz.x());
        hebRecY_->push_back(xyz.y());
        hebRecZ_->push_back(xyz.z());
        hebRecEnergy_->push_back(energy);
        hebSimX_->push_back(xp);
        hebSimY_->push_back(yp);
        hebSimZ_->push_back(std::get<3>(itr->second));
        hebSimEnergy_->push_back(std::get<0>(itr->second));
        hebSimEta_->push_back(std::get<1>(itr->second));
        hebRecEta_->push_back(xyz.eta());
        hebSimPhi_->push_back(std::get<2>(itr->second));
        hebRecPhi_->push_back(ang3);
        hebDetID_->push_back(itr->first);
      } else {
        hebRecVsSimX_->Fill(xp, xyz.x());
        hebRecVsSimY_->Fill(yp, xyz.y());
        hebRecVsSimZ_->Fill(std::get<3>(itr->second), xyz.z());
        hebdEtaVsEta_->Fill(std::get<1>(itr->second), (xyz.eta() - std::get<1>(itr->second)));
        hebdPhiVsPhi_->Fill(std::get<2>(itr->second), (ang3 - std::get<2>(itr->second)));
        hebdzVsZ_->Fill(std::get<3>(itr->second), (xyz.z() - std::get<3>(itr->second)));
        hebEnSimRec_->Fill(std::get<0>(itr->second), energy);
      }
      edm::LogVerbatim("HGCalValid") << "BHHit: " << std::hex << id.rawId() << std::dec << " Sim ("
                                     << std::get<0>(itr->second) << ", " << std::get<1>(itr->second) << ", "
                                     << std::get<2>(itr->second) << ", " << std::get<3>(itr->second) << ") Rec ("
                                     << energy << ", " << xyz.eta() << ", " << ang3 << ", " << xyz.z() << ")";
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCHitValidation);
