// system include files
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

//#define EDM_ML_DEBUG

class HGCalTimingAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalTimingAnalyzer(edm::ParameterSet const&);
  ~HGCalTimingAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void analyzeSimHits(int type, std::vector<PCaloHit> const& hits);
  void analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk,
                        edm::Handle<edm::SimVertexContainer> const& SimVtx);

  edm::Service<TFileService> fs_;
  const std::vector<int> idBeamDef_ = {1001};
  const std::string detectorEE_, detectorBeam_;
  const bool groupHits_;
  const double timeUnit_;
  const bool doTree_;
  const std::vector<int> idBeams_;
  const edm::ESGetToken<HGCalTBDDDConstants, IdealGeometryRecord> tokDDD_;
  const HGCalTBDDDConstants* hgcons_;
  const edm::InputTag labelGen_;
  const std::string labelHitEE_, labelHitBeam_;
  const edm::EDGetTokenT<edm::HepMCProduct> tok_hepMC_;
  const edm::EDGetTokenT<edm::SimTrackContainer> tok_simTk_;
  const edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_hitsEE_, tok_hitsBeam_;
  TTree* tree_;
  std::vector<uint32_t> simHitCellIdEE_, simHitCellIdBeam_;
  std::vector<float> simHitCellEnEE_, simHitCellEnBeam_;
  std::vector<float> simHitCellTmEE_, simHitCellTmBeam_;
  double xBeam_, yBeam_, zBeam_, pBeam_;
};

HGCalTimingAnalyzer::HGCalTimingAnalyzer(const edm::ParameterSet& iConfig)
    : detectorEE_(iConfig.getParameter<std::string>("DetectorEE")),
      detectorBeam_(iConfig.getParameter<std::string>("DetectorBeam")),
      groupHits_(iConfig.getParameter<bool>("GroupHits")),
      timeUnit_((!groupHits_) ? 0.000001 : (iConfig.getParameter<double>("TimeUnit"))),
      doTree_(iConfig.getUntrackedParameter<bool>("DoTree", false)),
      idBeams_((iConfig.getParameter<std::vector<int>>("IDBeams")).empty()
                   ? idBeamDef_
                   : (iConfig.getParameter<std::vector<int>>("IDBeams"))),
      tokDDD_(esConsumes<HGCalTBDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", detectorEE_))),
      labelGen_(iConfig.getParameter<edm::InputTag>("GeneratorSrc")),
      labelHitEE_(iConfig.getParameter<std::string>("CaloHitSrcEE")),
      labelHitBeam_(iConfig.getParameter<std::string>("CaloHitSrcBeam")),
      tok_hepMC_(consumes<edm::HepMCProduct>(labelGen_)),
      tok_simTk_(consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"))),
      tok_simVtx_(consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"))),
      tok_hitsEE_(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", labelHitEE_))),
      tok_hitsBeam_(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", labelHitBeam_))) {
  usesResource("TFileService");

  // now do whatever initialization is needed
  // Group hits (if groupHits_ = true) if hits come within timeUnit_
  // Only look into the beam counters with ID's as in idBeams_
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  st1 << "HGCalTimingAnalyzer:: Group Hits " << groupHits_ << " in " << timeUnit_ << " IdBeam " << idBeams_.size()
      << ":";
  for (const auto& id : idBeams_)
    st1 << " " << id;
  edm::LogVerbatim("HGCSim") << st1.str();

  edm::LogVerbatim("HGCSim") << "HGCalTimingAnalyzer:: GeneratorSource = " << labelGen_;
  edm::LogVerbatim("HGCSim") << "HGCalTimingAnalyzer:: Detector " << detectorEE_ << " with tags " << labelHitEE_;
  edm::LogVerbatim("HGCSim") << "HGCalTimingAnalyzer:: Detector " << detectorBeam_ << " with tags " << labelHitBeam_;
#endif
}

void HGCalTimingAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("DetectorEE", "HGCalEESensitive");
  desc.add<std::string>("DetectorBeam", "HcalTB06BeamDetector");
  desc.add<bool>("GroupHits", false);
  desc.add<double>("TimeUnit", 0.001);
  std::vector<int> ids = {1001, 1002, 1003, 1004, 1005};
  desc.add<std::vector<int>>("IDBeams", ids);
  desc.addUntracked<bool>("DoTree", true);
  desc.add<edm::InputTag>("GeneratorSrc", edm::InputTag("generatorSmeared"));
  desc.add<std::string>("CaloHitSrcEE", "HGCHitsEE");
  desc.add<std::string>("CaloHitSrcBeam", "HcalTB06BeamHits");
  descriptions.add("HGCalTimingAnalyzer", desc);
}

void HGCalTimingAnalyzer::beginJob() {
  std::string det(detectorEE_);
  if (doTree_) {
    tree_ = fs_->make<TTree>("HGCTB", "SimHitEnergy");
    tree_->Branch("xBeam", &xBeam_, "xBeam/D");
    tree_->Branch("yBeam", &yBeam_, "yBeam/D");
    tree_->Branch("zBeam", &zBeam_, "zBeam/D");
    tree_->Branch("pBeam", &pBeam_, "pBeam/D");
    tree_->Branch("simHitCellIdEE_", &simHitCellIdEE_);
    tree_->Branch("simHitCellEnEE_", &simHitCellEnEE_);
    tree_->Branch("simHitCellTmEE_", &simHitCellTmEE_);
    tree_->Branch("simHitCellIdBeam_", &simHitCellIdBeam_);
    tree_->Branch("simHitCellEnBeam_", &simHitCellEnBeam_);
    tree_->Branch("simHitCellTmBeam_", &simHitCellTmBeam_);
  }
}

void HGCalTimingAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  hgcons_ = &iSetup.getData(tokDDD_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalTimingAnalyzer::" << detectorEE_ << " defined with " << hgcons_->layers(false)
                             << " layers";
#endif
}

void HGCalTimingAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
#ifdef EDM_ML_DEBUG
  // Generator input
  const edm::Handle<edm::HepMCProduct>& evtMC = iEvent.getHandle(tok_hepMC_);
  if (!evtMC.isValid()) {
    edm::LogWarning("HGCal") << "no HepMCProduct found";
  } else {
    const HepMC::GenEvent* myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p, ++k) {
      edm::LogVerbatim("HGCSim") << "Particle[" << k << "] with p " << (*p)->momentum().rho() << " theta "
                                 << (*p)->momentum().theta() << " phi " << (*p)->momentum().phi();
    }
  }
#endif

  // Now the Simhits
  const edm::Handle<edm::SimTrackContainer>& SimTk = iEvent.getHandle(tok_simTk_);
  const edm::Handle<edm::SimVertexContainer>& SimVtx = iEvent.getHandle(tok_simVtx_);
  analyzeSimTracks(SimTk, SimVtx);

  simHitCellIdEE_.clear();
  simHitCellIdBeam_.clear();
  simHitCellEnEE_.clear();
  simHitCellEnBeam_.clear();
  simHitCellTmEE_.clear();
  simHitCellTmBeam_.clear();

  std::vector<PCaloHit> caloHits;
  const edm::Handle<edm::PCaloHitContainer>& theCaloHitContainers = iEvent.getHandle(tok_hitsEE_);
  if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "PcalohitContainer for " << detectorEE_ << " has " << theCaloHitContainers->size()
                               << " hits";
#endif
    caloHits.clear();
    caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), theCaloHitContainers->end());
    analyzeSimHits(0, caloHits);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "PCaloHitContainer does not exist for " << detectorEE_ << " !!!";
#endif
  }

  const edm::Handle<edm::PCaloHitContainer>& caloHitContainerBeam = iEvent.getHandle(tok_hitsBeam_);
  if (caloHitContainerBeam.isValid()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "PcalohitContainer for " << detectorBeam_ << " has " << caloHitContainerBeam->size()
                               << " hits";
#endif
    caloHits.clear();
    caloHits.insert(caloHits.end(), caloHitContainerBeam->begin(), caloHitContainerBeam->end());
    analyzeSimHits(1, caloHits);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "PCaloHitContainer does not exist for " << detectorBeam_ << " !!!";
#endif
  }
  if (doTree_)
    tree_->Fill();
}

void HGCalTimingAnalyzer::analyzeSimHits(int type, std::vector<PCaloHit> const& hits) {
#ifdef EDM_ML_DEBUG
  unsigned int i(0);
#endif
  std::map<std::pair<uint32_t, uint64_t>, std::pair<double, double>> map_hits;
  for (const auto& hit : hits) {
    double energy = hit.energy();
    double time = hit.time();
    uint32_t id = hit.id();
    if (type == 0) {
      int subdet, zside, layer, sector, subsector, cell;
      HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector, subsector, cell);
      std::pair<int, int> recoLayerCell = hgcons_->simToReco(cell, layer, sector, true);
      id = HGCalDetId((ForwardSubdetector)(subdet), zside, recoLayerCell.second, subsector, sector, recoLayerCell.first)
               .rawId();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "SimHit:Hit[" << i << "] Id " << subdet << ":" << zside << ":" << layer << ":"
                                 << sector << ":" << subsector << ":" << recoLayerCell.first << ":"
                                 << recoLayerCell.second << " Energy " << energy << " Time " << time;
#endif
    } else {
#ifdef EDM_ML_DEBUG
      int subdet, layer, x, y;
      HcalTestBeamNumbering::unpackIndex(id, subdet, layer, x, y);
      edm::LogVerbatim("HGCSim") << "SimHit:Hit[" << i << "] Beam Subdet " << subdet << " Layer " << layer << " x|y "
                                 << x << ":" << y << " Energy " << energy << " Time " << time;
#endif
    }
    uint64_t tid = (uint64_t)((time + 50.0) / timeUnit_);
    std::pair<uint32_t, uint64_t> key(id, tid);
    auto itr = map_hits.find(key);
    if (itr == map_hits.end()) {
      map_hits[key] = std::pair<double, double>(time, 0.0);
      itr = map_hits.find(key);
    }
    energy += (itr->second).second;
    map_hits[key] = std::pair<double, double>((itr->second).first, energy);
#ifdef EDM_ML_DEBUG
    ++i;
#endif
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "analyzeSimHits: Finds " << map_hits.size() << " hits "
                             << " from the Hit Vector of size " << hits.size() << " for type " << type;
#endif
  for (const auto& itr : map_hits) {
    uint32_t id = (itr.first).first;
    double time = (itr.second).first;
    double energy = (itr.second).second;
    if (type == 0) {
      simHitCellIdEE_.push_back(id);
      simHitCellEnEE_.push_back(energy);
      simHitCellTmEE_.push_back(time);
    } else {
      simHitCellIdBeam_.push_back(id);
      simHitCellEnBeam_.push_back(energy);
      simHitCellTmBeam_.push_back(time);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "SimHit::ID: " << std::hex << id << std::dec << " T: " << time << " E: " << energy;
#endif
  }
}

void HGCalTimingAnalyzer::analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk,
                                           edm::Handle<edm::SimVertexContainer> const& SimVtx) {
  xBeam_ = yBeam_ = zBeam_ = pBeam_ = -1000000;
  int vertIndex(-1);
  for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Track " << simTrkItr->trackId() << " Vertex " << simTrkItr->vertIndex() << " Type "
                               << simTrkItr->type() << " Charge " << simTrkItr->charge() << " momentum "
                               << simTrkItr->momentum() << " " << simTrkItr->momentum().P();
#endif
    if (vertIndex == -1) {
      vertIndex = simTrkItr->vertIndex();
      pBeam_ = simTrkItr->momentum().P();
    }
  }
  if (vertIndex != -1 && vertIndex < (int)SimVtx->size()) {
    edm::SimVertexContainer::const_iterator simVtxItr = SimVtx->begin();
    for (int iv = 0; iv < vertIndex; iv++)
      simVtxItr++;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Vertex " << vertIndex << " position " << simVtxItr->position();
#endif
    xBeam_ = simVtxItr->position().X();
    yBeam_ = simVtxItr->position().Y();
    zBeam_ = simVtxItr->position().Z();
  }
}

// define this as a plug-in

DEFINE_FWK_MODULE(HGCalTimingAnalyzer);
