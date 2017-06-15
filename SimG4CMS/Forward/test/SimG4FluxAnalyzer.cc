// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"

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

#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"

//#define EDM_ML_DEBUG

class SimG4FluxAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit SimG4FluxAnalyzer(edm::ParameterSet const&);
  ~SimG4FluxAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;

  edm::Service<TFileService>                   fs_;
  std::vector<std::string>                     lvNames_;
  std::vector<edm::EDGetTokenT<ParticleFlux> > tok_PF_;
  TTree                                       *tree_;

  std::vector<std::string>                     detName;
  std::vector<int>                             detId, pdgId, vxType;
  std::vector<float>                           tof, vtxX, vtxY, vtxZ;
  std::vector<float>                           hitPtX, hitPtY, hitPtZ;
  std::vector<float>                           momX, momY, momZ;
};

SimG4FluxAnalyzer::SimG4FluxAnalyzer(const edm::ParameterSet& iConfig) {

  usesResource("TFileService");

  //now do whatever initialization is needed
  lvNames_  = iConfig.getParameter<std::vector<std::string> >("LVNames");
#ifdef EDM_ML_DEBUG
  std::cout << "SimG4FluxAnalyzer:: for " << lvNames_.size() << " names:";
  for (auto name : lvNames_) std::cout << " " << name;
  std::cout << std::endl;
#endif

  for (auto name : lvNames_) {
    std::string tagn = name+"ParticleFlux";
    tok_PF_.push_back(consumes<ParticleFlux>(edm::InputTag("g4SimHits",tagn)));
#ifdef EDM_ML_DEBUG
    std::cout << "Flux source " << edm::InputTag("g4SimHits",tagn) <<std::endl;
#endif
  }
}

SimG4FluxAnalyzer::~SimG4FluxAnalyzer() {}

void SimG4FluxAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  std::vector<std::string> lvnames = {"TotemT1Part1","TotemT1Part2","TotemT1Part3","TotemT2Part1","TotemT2Part2","TotemT2Part3"};
  desc.add<std::vector<std::string>>("LVNames",lvnames);
  descriptions.add("SimG4FluxAnalyzer",desc);
}

void SimG4FluxAnalyzer::beginJob() {

  tree_ = fs_->make<TTree>("Flux","ParticleFlux");
  tree_->Branch("DetectorName", &detName);
  tree_->Branch("DetectorID",   &detId);
  tree_->Branch("ParticleCode", &pdgId);
  tree_->Branch("VertexType",   &vxType);
  tree_->Branch("ParticleTOF",  &tof);
  tree_->Branch("VertexX",      &vtxX);
  tree_->Branch("VertexY",      &vtxY);
  tree_->Branch("VertexZ",      &vtxZ);
  tree_->Branch("HitPointX",    &hitPtX);
  tree_->Branch("HitPointY",    &hitPtY);
  tree_->Branch("HitPointZ",    &hitPtZ);
  tree_->Branch("MomentumX",    &momX);
  tree_->Branch("MomentumY",    &momY);
  tree_->Branch("MomentumZ",    &momZ);
}

void SimG4FluxAnalyzer::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {

  //Loop over all flux containers
  detName.clear(); detId.clear();  pdgId.clear(); vxType.clear();
  tof.clear();     vtxX.clear();   vtxY.clear();  vtxZ.clear();
  hitPtX.clear();  hitPtY.clear(); hitPtZ.clear();
  momX.clear();    momY.clear();   momZ.clear();
#ifdef EDM_ML_DEBUG
  unsigned int k(0);
#endif
  for (auto token : tok_PF_) {
    edm::Handle<ParticleFlux>  pFlux;
    iEvent.getByToken(token, pFlux);
    if (pFlux.isValid()) {
      const ParticleFlux             *pflux = pFlux.product();
      std::string                     name = pflux->getName();
      int                             id   = pflux->getId();
      std::vector<ParticleFlux::flux> flux = pflux->getFlux();
#ifdef EDM_ML_DEBUG
      std::cout << "SimG4FluxAnalyzer:: ParticleFlux for " << lvNames_[k] 
		<< " has " << pflux->getComponents() << " entries" <<std::endl;
      ++k;
      unsigned k1(0);
#endif
      for (auto element : flux) {
	detName.push_back(name); detId.push_back(id);
	pdgId.push_back(element.pdgId);
	vxType.push_back(element.vxType);
	tof.push_back(element.tof);
	vtxX.push_back(element.vertex.X());
	vtxY.push_back(element.vertex.Y());
	vtxZ.push_back(element.vertex.Z());
	hitPtX.push_back(element.hitPoint.X());
	hitPtY.push_back(element.hitPoint.Y());
	hitPtZ.push_back(element.hitPoint.Z());
	momX.push_back(element.momentum.X());
	momY.push_back(element.momentum.Y());
	momZ.push_back(element.momentum.Z());
#ifdef EDM_ML_DEBUG
	std::cout << "Flux[" << k1 << "] PDGId " << element.pdgId << " VT "
		  << element.vxType << " ToF " << element.tof << " Vertex "
		  << element.vertex << " Hit " << element.hitPoint << " p "
		  << element.momentum << std::endl;
	++k1;
#endif
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "All flux compnents have " << detName.size() << " entries" 
	    << std::endl;
#endif
  if (detName.size() > 0) tree_->Fill();

}
  
//define this as a plug-in
DEFINE_FWK_MODULE(SimG4FluxAnalyzer);
