// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "TH2D.h"

class HGCalWaferStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalWaferStudy(const edm::ParameterSet&);
  ~HGCalWaferStudy() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  void analyzeHits(int, const std::string&, const std::vector<PCaloHit>&);

  // ----------member data ---------------------------
  const std::vector<std::string> nameDetectors_, caloHitSources_;
  const std::vector<edm::InputTag> digiSources_;
  const std::vector<edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord>> ddTokens_;
  const std::vector<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>> geomTokens_;
  const int verbosity_, nBinHit_, nBinDig_;
  const double xyMinHit_, xyMaxHit_;
  const double xyMinDig_, xyMaxDig_;
  const bool ifNose_;
  std::vector<int> layerMnSim_, layerMxSim_;
  std::vector<int> layerMnDig_, layerMxDig_;
  std::vector<int> layerSim_, layerDig_, layerFront_;
  std::vector<const HGCalDDDConstants*> hgcons_;
  std::vector<const HGCalGeometry*> hgeoms_;
  std::vector<edm::EDGetTokenT<edm::PCaloHitContainer>> tok_hits_;
  std::vector<edm::EDGetToken> tok_digi_;

  //histogram related stuff
  static const int nType = 2;
  std::vector<TH2D*> h_XYsi1_[nType], h_XYsi2_[nType];
  std::vector<TH2D*> h_XYdi1_[nType], h_XYdi2_[nType];
};

HGCalWaferStudy::HGCalWaferStudy(const edm::ParameterSet& iConfig)
    : nameDetectors_(iConfig.getParameter<std::vector<std::string>>("detectorNames")),
      caloHitSources_(iConfig.getParameter<std::vector<std::string>>("caloHitSources")),
      digiSources_(iConfig.getParameter<std::vector<edm::InputTag>>("digiSources")),
      ddTokens_{
          edm::vector_transform(nameDetectors_,
                                [this](const std::string& name) {
                                  return esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
                                      edm::ESInputTag{"", name});
                                })},
      geomTokens_{edm::vector_transform(
          nameDetectors_,
          [this](const std::string& name) {
            return esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name});
          })},
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      nBinHit_(iConfig.getUntrackedParameter<int>("nBinHit", 600)),
      nBinDig_(iConfig.getUntrackedParameter<int>("nBinDig", 600)),
      xyMinHit_(iConfig.getUntrackedParameter<double>("xyMinHit", -3000.0)),
      xyMaxHit_(iConfig.getUntrackedParameter<double>("xyMaxHit", 3000.0)),
      xyMinDig_(iConfig.getUntrackedParameter<double>("xyMinDig", -300.0)),
      xyMaxDig_(iConfig.getUntrackedParameter<double>("xyMaxDig", 300.0)),
      ifNose_(iConfig.getUntrackedParameter<bool>("ifNose", false)),
      layerMnSim_(iConfig.getUntrackedParameter<std::vector<int>>("layerMinSim")),
      layerMxSim_(iConfig.getUntrackedParameter<std::vector<int>>("layerMaxSim")),
      layerMnDig_(iConfig.getUntrackedParameter<std::vector<int>>("layerMinDig")),
      layerMxDig_(iConfig.getUntrackedParameter<std::vector<int>>("layerMaxDig")) {
  usesResource(TFileService::kSharedResource);

  for (auto const& source : caloHitSources_) {
    tok_hits_.emplace_back(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", source)));
    edm::LogVerbatim("HGCalValidation") << "SimHitSource: " << source;
  }
  for (auto const& source : digiSources_) {
    tok_digi_.emplace_back(consumes<HGCalDigiCollection>(source));
    edm::LogVerbatim("HGCalValidation") << "DigiSource: " << source;
  }
}

void HGCalWaferStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  std::vector<std::string> simSources = {"HGCHitsEE", "HGCHitsHEfront"};
  std::vector<edm::InputTag> digSources = {edm::InputTag("simHGCalUnsuppressedDigis", "EE"),
                                           edm::InputTag("simHGCalUnsuppressedDigis", "HEfront")};
  std::vector<int> layers = {28, 24};
  std::vector<int> layerMin = {1, 1};
  desc.add<std::vector<std::string>>("detectorNames", names);
  desc.add<std::vector<std::string>>("caloHitSources", simSources);
  desc.add<std::vector<edm::InputTag>>("digiSources", digSources);
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<int>("nBinHit", 600);
  desc.addUntracked<int>("nBinDig", 600);
  desc.addUntracked<double>("xyMinHit", -3000.0);
  desc.addUntracked<double>("xyMaxHit", 3000.0);
  desc.addUntracked<double>("xyMinDig", -300.0);
  desc.addUntracked<double>("xyMaxDig", 300.0);
  desc.addUntracked<bool>("ifNose", false);
  desc.addUntracked<std::vector<int>>("layerMaxSim", layers);
  desc.addUntracked<std::vector<int>>("layerMaxDig", layers);
  desc.addUntracked<std::vector<int>>("layerMinSim", layerMin);
  desc.addUntracked<std::vector<int>>("layerMinDig", layerMin);
  descriptions.add("hgcalWaferStudy", desc);
}

void HGCalWaferStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //First the hits
  for (unsigned int k = 0; k < tok_hits_.size(); ++k) {
    edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
    iEvent.getByToken(tok_hits_[k], theCaloHitContainers);
    if (theCaloHitContainers.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation")
            << " PcalohitItr = " << theCaloHitContainers->size() << " Geometry Pointer " << hgcons_[k] << " # of Hists "
            << h_XYsi1_[k].size() << ":" << h_XYsi2_[k].size();
      int kount(0);
      for (auto const& hit : *(theCaloHitContainers)) {
        unsigned int id = hit.id();
        std::pair<float, float> xy;
        int layer(0), zside(1);
        bool wvtype(true);
        if (ifNose_) {
          HFNoseDetId detId = HFNoseDetId(id);
          layer = detId.layer();
          zside = detId.zside();
          wvtype = hgcons_[k]->waferVirtual(layer, detId.waferU(), detId.waferV());
          xy = hgcons_[k]->locateCell(layer, detId.waferU(), detId.waferV(), detId.cellU(), detId.cellV(), false, true);
        } else if (hgcons_[k]->waferHexagon8()) {
          HGCSiliconDetId detId = HGCSiliconDetId(id);
          layer = detId.layer();
          zside = detId.zside();
          wvtype = hgcons_[k]->waferVirtual(layer, detId.waferU(), detId.waferV());
          xy = hgcons_[k]->locateCell(layer, detId.waferU(), detId.waferV(), detId.cellU(), detId.cellV(), false, true);
        } else {
          int subdet, sector, type, cell;
          HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector, type, cell);
          xy = hgcons_[k]->locateCell(cell, layer, sector, false);
          wvtype = hgcons_[k]->waferVirtual(layer, sector, 0);
        }
        double xp = (zside < 0) ? -xy.first : xy.first;
        double yp = xy.second;
        int ll = layer - layerMnSim_[k];
        if (verbosity_ > 1)
          edm::LogVerbatim("HGCalValidation")
              << "Hit [" << kount << "] Layer " << layer << ":" << ll << " x|y " << xp << ":" << yp << " Type "
              << wvtype << " Hist pointers " << h_XYsi1_[k][ll] << ":" << h_XYsi2_[k][ll];
        if (layer >= layerMnSim_[k] && layer <= layerMxSim_[k]) {
          if (wvtype)
            h_XYsi2_[k][ll]->Fill(xp, yp);
          else
            h_XYsi1_[k][ll]->Fill(xp, yp);
        }
        ++kount;
      }
    } else if (verbosity_ > 0) {
      edm::LogVerbatim("HGCalValidation") << "PCaloHitContainer does not "
                                          << "exist for " << nameDetectors_[k];
    }
  }

  //Then the digis
  for (unsigned int k = 0; k < tok_digi_.size(); ++k) {
    edm::Handle<HGCalDigiCollection> theHGCDigiContainer;
    iEvent.getByToken(tok_digi_[k], theHGCDigiContainer);
    if (theHGCDigiContainer.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation")
            << nameDetectors_[k] << " with " << theHGCDigiContainer->size() << " element(s) Geometry Pointer "
            << hgeoms_[k] << " # of Hists " << h_XYdi1_[k].size() << ":" << h_XYdi2_[k].size();
      int kount(0);
      for (const auto& it : *(theHGCDigiContainer.product())) {
        DetId id = it.id();
        int layer(0);
        bool wvtype(true);
        if (ifNose_) {
          HFNoseDetId detId = HFNoseDetId(id);
          layer = detId.layer();
          wvtype = hgcons_[k]->waferVirtual(layer, detId.waferU(), detId.waferV());
        } else if (hgcons_[k]->waferHexagon8()) {
          HGCSiliconDetId detId = HGCSiliconDetId(id);
          layer = detId.layer();
          wvtype = hgcons_[k]->waferVirtual(layer, detId.waferU(), detId.waferV());
        } else {
          HGCalDetId detId = HGCalDetId(id);
          layer = detId.layer();
          int wafer = detId.wafer();
          wvtype = hgcons_[k]->waferVirtual(layer, wafer, 0);
        }
        int ll = layer - layerMnDig_[k];
        const GlobalPoint& gcoord = hgeoms_[k]->getPosition(id);
        double xp = gcoord.x();
        double yp = gcoord.y();
        if (verbosity_ > 1)
          edm::LogVerbatim("HGCalValidation")
              << "Digi [" << kount << "] Layer " << layer << ":" << ll << " x|y " << xp << ":" << yp << " Type "
              << wvtype << " Hist pointers " << h_XYdi1_[k][ll] << ":" << h_XYdi2_[k][ll];
        if (layer >= layerMnDig_[k] && layer <= layerMxDig_[k]) {
          if (wvtype)
            h_XYdi2_[k][ll]->Fill(xp, yp);
          else
            h_XYdi1_[k][ll]->Fill(xp, yp);
        }
        ++kount;
      }
    } else {
      edm::LogVerbatim("HGCalValidation")
          << "DigiCollection handle " << digiSources_[k] << " does not exist for " << nameDetectors_[k] << " !!!";
    }
  }
}

// ------------ method called when starting to processes a run  ------------
void HGCalWaferStudy::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  for (unsigned int k = 0; k < nameDetectors_.size(); ++k) {
    const auto& pHGDC = iSetup.getData(ddTokens_[k]);
    hgcons_.emplace_back(&pHGDC);
    layerSim_.emplace_back(hgcons_.back()->layers(false));
    layerDig_.emplace_back(hgcons_.back()->layers(true));
    layerFront_.emplace_back(hgcons_.back()->firstLayer());
    layerMnSim_[k] = std::max(layerMnSim_[k], layerFront_[k]);
    layerMnDig_[k] = std::max(layerMnDig_[k], layerFront_[k]);
    layerMxSim_[k] = std::min((layerFront_[k] + layerSim_[k] - 1), layerMxSim_[k]);
    layerMxDig_[k] = std::min((layerFront_[k] + layerDig_[k] - 1), layerMxDig_[k]);

    const auto& geom = iSetup.getData(geomTokens_[k]);
    hgeoms_.emplace_back(&geom);
    if (verbosity_ > 0)
      edm::LogVerbatim("HGCalValidation")
          << nameDetectors_[k] << " defined with " << layerFront_[k] << ":" << layerSim_[k] << ":" << layerDig_[k]
          << " Layers which gives limits " << layerMnSim_[k] << ":" << layerMxSim_[k] << " for Sim " << layerMnDig_[k]
          << ":" << layerMxDig_[k] << " for Digi "
          << "\nLimits for plots " << nBinHit_ << ":" << xyMinHit_ << ":" << xyMaxHit_ << " (Sim) " << nBinDig_ << ":"
          << xyMinDig_ << ":" << xyMaxDig_ << " (Digi) with Pointers " << hgcons_.back() << ":" << hgeoms_.back();
  }

  // Now define the histograms
  edm::Service<TFileService> fs;
  std::ostringstream name, title;
  for (unsigned int ih = 0; ih <= nameDetectors_.size(); ++ih) {
    for (int i = layerMnSim_[ih]; i <= layerMxSim_[ih]; ++i) {
      name.str("");
      title.str("");
      name << "XY_" << nameDetectors_[ih] << "L" << i << "SimR";
      title << "y vs x (Layer " << i << ") real wafers for " << nameDetectors_[ih] << " at SimHit level";
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << i << " book " << name.str() << ":" << title.str();
      h_XYsi1_[ih].emplace_back(fs->make<TH2D>(
          name.str().c_str(), title.str().c_str(), nBinHit_, xyMinHit_, xyMaxHit_, nBinHit_, xyMinHit_, xyMaxHit_));
      name.str("");
      title.str("");
      name << "XY_" << nameDetectors_[ih] << "L" << i << "SimV";
      title << "y vs x (Layer " << i << ") virtual wafers for " << nameDetectors_[ih] << " at SimHit level";
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << i << " book " << name.str() << ":" << title.str();
      h_XYsi2_[ih].emplace_back(fs->make<TH2D>(
          name.str().c_str(), title.str().c_str(), nBinHit_, xyMinHit_, xyMaxHit_, nBinHit_, xyMinHit_, xyMaxHit_));
    }
    edm::LogVerbatim("HGCalValidation") << "Complete booking of Sim Plots for " << nameDetectors_[ih];

    for (int i = layerMnDig_[ih]; i <= layerMxDig_[ih]; ++i) {
      name.str("");
      title.str("");
      name << "XY_" << nameDetectors_[ih] << "L" << i << "DigR";
      title << "y vs x (Layer " << i << ") real wafers for " << nameDetectors_[ih] << " at Digi level";
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << i << " book " << name.str() << ":" << title.str();
      h_XYdi1_[ih].emplace_back(fs->make<TH2D>(
          name.str().c_str(), title.str().c_str(), nBinDig_, xyMinDig_, xyMaxDig_, nBinDig_, xyMinDig_, xyMaxDig_));
      name.str("");
      title.str("");
      name << "XY_" << nameDetectors_[ih] << "L" << i << "DigV";
      title << "y vs x (Layer " << i << ") virtual wafers for " << nameDetectors_[ih] << " at Digi level";
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << i << " book " << name.str() << ":" << title.str();
      h_XYdi2_[ih].emplace_back(fs->make<TH2D>(
          name.str().c_str(), title.str().c_str(), nBinDig_, xyMinDig_, xyMaxDig_, nBinDig_, xyMinDig_, xyMaxDig_));
    }
    edm::LogVerbatim("HGCalValidation") << "Complete booking of Digi Plots for " << nameDetectors_[ih];
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferStudy);
