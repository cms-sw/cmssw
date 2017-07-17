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

  std::vector<std::string>                     detName_;
  std::vector<int>                             detId_, pdgId_, vxType_;
  std::vector<float>                           tof_, vtxX_, vtxY_, vtxZ_;
  std::vector<float>                           hitPtX_, hitPtY_, hitPtZ_;
  std::vector<float>                           momX_, momY_, momZ_;
};

SimG4FluxAnalyzer::SimG4FluxAnalyzer(const edm::ParameterSet& iConfig) {

  usesResource("TFileService");

  //now do whatever initialization is needed
  lvNames_  = iConfig.getParameter<std::vector<std::string> >("LVNames");
#ifdef EDM_ML_DEBUG
  std::cout << "SimG4FluxAnalyzer:: for " << lvNames_.size() << " names:";
  for (const auto& name : lvNames_) std::cout << " " << name;
  std::cout << std::endl;
#endif

  for (const auto& name : lvNames_) {
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
  tree_->Branch("DetectorName", &detName_);
  tree_->Branch("DetectorID",   &detId_);
  tree_->Branch("ParticleCode", &pdgId_);
  tree_->Branch("VertexType",   &vxType_);
  tree_->Branch("ParticleTOF",  &tof_);
  tree_->Branch("VertexX",      &vtxX_);
  tree_->Branch("VertexY",      &vtxY_);
  tree_->Branch("VertexZ",      &vtxZ_);
  tree_->Branch("HitPointX",    &hitPtX_);
  tree_->Branch("HitPointY",    &hitPtY_);
  tree_->Branch("HitPointZ",    &hitPtZ_);
  tree_->Branch("MomentumX",    &momX_);
  tree_->Branch("MomentumY",    &momY_);
  tree_->Branch("MomentumZ",    &momZ_);
}

void SimG4FluxAnalyzer::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {

  //Loop over all flux containers
  detName_.clear(); detId_.clear();  pdgId_.clear(); vxType_.clear();
  tof_.clear();     vtxX_.clear();   vtxY_.clear();  vtxZ_.clear();
  hitPtX_.clear();  hitPtY_.clear(); hitPtZ_.clear();
  momX_.clear();    momY_.clear();   momZ_.clear();
#ifdef EDM_ML_DEBUG
  unsigned int k(0);
#endif
  for (const auto& token : tok_PF_) {
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
      for (const auto& element : flux) {
	detName_.push_back(name); detId_.push_back(id);
	pdgId_.push_back(element.pdgId);
	vxType_.push_back(element.vxType);
	tof_.push_back(element.tof);
	vtxX_.push_back(element.vertex.X());
	vtxY_.push_back(element.vertex.Y());
	vtxZ_.push_back(element.vertex.Z());
	hitPtX_.push_back(element.hitPoint.X());
	hitPtY_.push_back(element.hitPoint.Y());
	hitPtZ_.push_back(element.hitPoint.Z());
	momX_.push_back(element.momentum.X());
	momY_.push_back(element.momentum.Y());
	momZ_.push_back(element.momentum.Z());
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
  std::cout << "All flux compnents have " << detName_.size() << " entries" 
	    << std::endl;
#endif
  if (detName_.size() > 0) tree_->Fill();

}
  
//define this as a plug-in
DEFINE_FWK_MODULE(SimG4FluxAnalyzer);
