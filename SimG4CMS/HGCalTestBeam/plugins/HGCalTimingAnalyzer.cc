// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalDetId.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

//#define EDM_ML_DEBUG

class HGCalTimingAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HGCalTimingAnalyzer(edm::ParameterSet const&);
  ~HGCalTimingAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override ;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void analyzeSimHits(int type, std::vector<PCaloHit> const& hits);
  void analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk, 
			edm::Handle<edm::SimVertexContainer> const& SimVtx);

  edm::Service<TFileService>                fs_;
  const HGCalDDDConstants                  *hgcons_;
  bool                                      doTree_, groupHits_;
  std::string                               detectorEE_, detectorBeam_;
  double                                    timeUnit_;
  std::vector<int>                          idBeams_;
  edm::EDGetTokenT<edm::PCaloHitContainer>  tok_hitsEE_, tok_hitsBeam_;
  edm::EDGetTokenT<edm::SimTrackContainer>  tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;
  edm::EDGetTokenT<edm::HepMCProduct>       tok_hepMC_;
  TTree                                    *tree_;
  std::vector<uint32_t>                     simHitCellIdEE_, simHitCellIdBeam_;
  std::vector<float>                        simHitCellEnEE_, simHitCellEnBeam_;
  std::vector<float>                        simHitCellTmEE_, simHitCellTmBeam_;
  double                                    xBeam_, yBeam_, zBeam_, pBeam_;
};

HGCalTimingAnalyzer::HGCalTimingAnalyzer(const edm::ParameterSet& iConfig) {

  usesResource("TFileService");

  //now do whatever initialization is needed
  detectorEE_  = iConfig.getParameter<std::string>("DetectorEE");
  detectorBeam_= iConfig.getParameter<std::string>("DetectorBeam");
  // Group hits (if groupHits_ = true) if hits come within timeUnit_
  groupHits_   = iConfig.getParameter<bool>("GroupHits");
  timeUnit_    = iConfig.getParameter<double>("TimeUnit");
  // Only look into the beam counters with ID's as in idBeams_
  idBeams_     = iConfig.getParameter<std::vector<int>>("IDBeams");  
  doTree_      = iConfig.getUntrackedParameter<bool>("DoTree",false);
  if (!groupHits_) timeUnit_ = 0.000001;
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalTimingAnalyzer:: Group Hits " << groupHits_ << " in "
	    << timeUnit_ << " IdBeam " << idBeams_.size() << ":";
  for (const auto& id : idBeams_) std::cout << " " << id;
  std::cout << std::endl;
#endif
  if (idBeams_.empty()) idBeams_.push_back(1001);

  edm::InputTag tmp0 = iConfig.getParameter<edm::InputTag>("GeneratorSrc");
  tok_hepMC_   = consumes<edm::HepMCProduct>(tmp0);
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalTimingAnalyzer:: GeneratorSource = " << tmp0 << std::endl;
#endif
  tok_simTk_   = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  tok_simVtx_  = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  std::string   tmp1 = iConfig.getParameter<std::string>("CaloHitSrcEE");
  tok_hitsEE_  = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",tmp1));
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalTimingAnalyzer:: Detector " << detectorEE_ 
	    << " with tags " << tmp1 << std::endl;
#endif
  tmp1         = iConfig.getParameter<std::string>("CaloHitSrcBeam");
  tok_hitsBeam_= consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",tmp1));
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalTimingAnalyzer:: Detector " << detectorBeam_ 
	    << " with tags " << tmp1 << std::endl;
#endif
}

HGCalTimingAnalyzer::~HGCalTimingAnalyzer() {}

void HGCalTimingAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("DetectorEE","HGCalEESensitive");
  desc.add<std::string>("DetectorBeam","HcalTB06BeamDetector");
  desc.add<bool>("GroupHits",false);
  desc.add<double>("TimeUnit",0.001);
  std::vector<int> ids = {1001,1002,1003,1004,1005};
  desc.add<std::vector<int>>("IDBeams",ids);
  desc.addUntracked<bool>("DoTree",true);
  desc.add<edm::InputTag>("GeneratorSrc",edm::InputTag("generatorSmeared"));
  desc.add<std::string>("CaloHitSrcEE","HGCHitsEE");
  desc.add<std::string>("CaloHitSrcBeam","HcalTB06BeamHits");
  descriptions.add("HGCalTimingAnalyzer",desc);
}

void HGCalTimingAnalyzer::beginJob() {
  std::string det(detectorEE_); 
  if (doTree_) {
    tree_ = fs_->make<TTree>("HGCTB","SimHitEnergy");
    tree_->Branch("xBeam",             &xBeam_,           "xBeam/D");
    tree_->Branch("yBeam",             &yBeam_,           "yBeam/D");
    tree_->Branch("zBeam",             &zBeam_,           "zBeam/D");
    tree_->Branch("pBeam",             &pBeam_,           "pBeam/D");
    tree_->Branch("simHitCellIdEE_",   &simHitCellIdEE_);
    tree_->Branch("simHitCellEnEE_",   &simHitCellEnEE_);
    tree_->Branch("simHitCellTmEE_",   &simHitCellTmEE_);
    tree_->Branch("simHitCellIdBeam_", &simHitCellIdBeam_);
    tree_->Branch("simHitCellEnBeam_", &simHitCellEnBeam_);
    tree_->Branch("simHitCellTmBeam_", &simHitCellTmBeam_);
  }
}

void HGCalTimingAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  edm::ESHandle<HGCalDDDConstants>  pHGDC;
  iSetup.get<IdealGeometryRecord>().get(detectorEE_, pHGDC);
  hgcons_ = &(*pHGDC);
    
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalTimingAnalyzer::" << detectorEE_ << " defined with "
	    << hgcons_->layers(false) << " layers" << std::endl;
#endif
  
}

void HGCalTimingAnalyzer::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) {

#ifdef EDM_ML_DEBUG
  //Generator input
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_hepMC_,evtMC);
  if (!evtMC.isValid()) {
    edm::LogWarning("HGCal") << "no HepMCProduct found";
  } else { 
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
         p != myGenEvent->particles_end(); ++p, ++k) {
      std::cout << "Particle[" << k << "] with p " << (*p)->momentum().rho() 
		<< " theta " << (*p)->momentum().theta() << " phi "
		<< (*p)->momentum().phi() << std::endl;
    }
  }
#endif

  //Now the Simhits
  edm::Handle<edm::SimTrackContainer>  SimTk;
  iEvent.getByToken(tok_simTk_, SimTk);
  edm::Handle<edm::SimVertexContainer> SimVtx;
  iEvent.getByToken(tok_simVtx_, SimVtx);
  analyzeSimTracks(SimTk, SimVtx);

  simHitCellIdEE_.clear(); simHitCellIdBeam_.clear(); 
  simHitCellEnEE_.clear(); simHitCellEnBeam_.clear();
  simHitCellTmEE_.clear(); simHitCellTmBeam_.clear();

  edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
  std::vector<PCaloHit>               caloHits;
  iEvent.getByToken(tok_hitsEE_, theCaloHitContainers);
  if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
    std::cout << "PcalohitContainer for " << detectorEE_ << " has "
	      << theCaloHitContainers->size() << " hits" << std::endl;
#endif
    caloHits.clear();
    caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
		    theCaloHitContainers->end());
    analyzeSimHits(0, caloHits);
  } else {
#ifdef EDM_ML_DEBUG
    std::cout << "PCaloHitContainer does not exist for " << detectorEE_ 
	      << " !!!" << std::endl;
#endif
  }
    
  iEvent.getByToken(tok_hitsBeam_, theCaloHitContainers);
  if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
    std::cout << "PcalohitContainer for " << detectorBeam_ << " has "
	      << theCaloHitContainers->size() << " hits" << std::endl;
#endif
    caloHits.clear();
    caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
		    theCaloHitContainers->end());
    analyzeSimHits(1, caloHits);
  } else {
#ifdef EDM_ML_DEBUG
    std::cout << "PCaloHitContainer does not exist for " << detectorBeam_ 
	      << " !!!" << std::endl;
#endif
  }
  if (doTree_) tree_->Fill();

}

void HGCalTimingAnalyzer::analyzeSimHits (int type, 
					  std::vector<PCaloHit> const& hits) {

#ifdef EDM_ML_DEBUG
  unsigned int i(0);
#endif
  std::map<std::pair<uint32_t,uint64_t>,std::pair<double,double> > map_hits;
  for (const auto& hit : hits) {
    double energy      = hit.energy();
    double time        = hit.time();
    uint32_t id        = hit.id();
    if (type == 0) {
      int      subdet, zside, layer, sector, subsector, cell;
      HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector,
					     subsector, cell);
      std::pair<int,int> recoLayerCell = hgcons_->simToReco(cell,layer,sector,true);
      id               = HGCalDetId((ForwardSubdetector)(subdet),zside,
				    recoLayerCell.second,subsector,sector,
				    recoLayerCell.first).rawId();
#ifdef EDM_ML_DEBUG
      std::cout << "SimHit:Hit[" << i << "] Id " << subdet << ":" << zside 
		<< ":" << layer << ":" << sector << ":" << subsector << ":" 
		<< recoLayerCell.first << ":" << recoLayerCell.second
		<< " Energy " << energy << " Time " << time << std::endl;
#endif
    } else {
#ifdef EDM_ML_DEBUG
      int      subdet, layer, x, y;
      HcalTestBeamNumbering::unpackIndex(id, subdet, layer, x, y);
      std::cout << "SimHit:Hit[" << i << "] Beam Subdet " << subdet 
		<< " Layer " << layer << " x|y " << x << ":" << y
		<< " Energy " << energy << " Time " << time << std::endl;
#endif
    }
    uint64_t tid = (uint64_t)((time+50.0)/timeUnit_);
    std::pair<uint32_t,uint64_t> key(id,tid);
    auto itr = map_hits.find(key);
    if (itr == map_hits.end()) {
      map_hits[key] = std::pair<double,double>(time,0.0);
      itr  = map_hits.find(key);
    }
    energy += (itr->second).second;
    map_hits[key] = std::pair<double,double>((itr->second).first,energy);
#ifdef EDM_ML_DEBUG
    ++i;
#endif
  }
    
#ifdef EDM_ML_DEBUG
  std::cout << "analyzeSimHits: Finds " << map_hits.size() << " hits "
	    << " from the Hit Vector of size " << hits.size() << " for type "
	    << type << std::endl;
#endif
  for (const auto& itr: map_hits) {
    uint32_t id   = (itr.first).first;
    double time   = (itr.second).first;
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
    std::cout << "SimHit::ID: " << std::hex << id << std::dec << " T: " << time
	      << " E: " << energy << std::endl;
#endif
  }
}


void HGCalTimingAnalyzer::analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk, 
					   edm::Handle<edm::SimVertexContainer> const& SimVtx) {

  xBeam_ = yBeam_ = zBeam_ = pBeam_ = -1000000;
  int vertIndex(-1);
  for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); 
       simTrkItr!= SimTk->end(); simTrkItr++) {
#ifdef EDM_ML_DEBUG
    std::cout << "Track " << simTrkItr->trackId() << " Vertex "
	      << simTrkItr->vertIndex() << " Type " << simTrkItr->type()
	      << " Charge " << simTrkItr->charge() << " momentum "
	      << simTrkItr->momentum() << " " << simTrkItr->momentum().P()
	      << std::endl;
#endif
    if (vertIndex == -1) {
      vertIndex = simTrkItr->vertIndex();
      pBeam_    = simTrkItr->momentum().P();
    }
  }
  if (vertIndex != -1 && vertIndex < (int)SimVtx->size()) {
    edm::SimVertexContainer::const_iterator simVtxItr= SimVtx->begin();
    for (int iv=0; iv<vertIndex; iv++) simVtxItr++;
#ifdef EDM_ML_DEBUG
    std::cout << "Vertex " << vertIndex << " position "
	      << simVtxItr->position() << std::endl;
#endif
    xBeam_ = simVtxItr->position().X();
    yBeam_ = simVtxItr->position().Y();
    zBeam_ = simVtxItr->position().Z();
  }

}
  
//define this as a plug-in

DEFINE_FWK_MODULE(HGCalTimingAnalyzer);
