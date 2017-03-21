// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

//#define EDM_ML_DEBUG

class FTSimHitTest: public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit FTSimHitTest(const edm::ParameterSet& ps);
  ~FTSimHitTest();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  virtual void beginJob() override {}
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  void plotHits(const edm::Handle<edm::PSimHitContainer>&, const int);

private:

  edm::Service<TFileService>                 fs_;
  std::string                                g4Label_, barrelHit_, endcapHit_;
  edm::EDGetTokenT<edm::PSimHitContainer>    tok_hitBarrel_, tok_hitEndcap_;
  const FastTimeDDDConstants                *ftcons_; 
  TH1D                                      *hsimE_[2], *hsimT_[2], *hcell_[2];
  TH2D                                      *hsimP_[2], *hsimM_[2];
};


FTSimHitTest::FTSimHitTest(const edm::ParameterSet& ps) : ftcons_(0) {

  usesResource("TFileService");

  g4Label_   = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  barrelHit_ = ps.getUntrackedParameter<std::string>("HitCollection","FastTimerHitsBarrel");
  endcapHit_ = ps.getUntrackedParameter<std::string>("HitCollection","FastTimerHitsEndcap");

  tok_hitBarrel_ = consumes<edm::PSimHitContainer>(edm::InputTag(g4Label_,barrelHit_));
  tok_hitEndcap_ = consumes<edm::PSimHitContainer>(edm::InputTag(g4Label_,endcapHit_));
#ifdef EDM_ML_DEBUG
  std::cout << "FTSimHitTest::Input for SimHit for Barrel: " 
	    << edm::InputTag(g4Label_,barrelHit_) << " and Endcap: " 
	    << edm::InputTag(g4Label_,endcapHit_) << std::endl;
#endif
}

FTSimHitTest::~FTSimHitTest() {}

void FTSimHitTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void FTSimHitTest::beginRun(edm::Run const&, edm::EventSetup const& es) {
  
  edm::ESHandle<FastTimeDDDConstants> fdc;
  es.get<IdealGeometryRecord>().get(fdc);
  if (fdc.isValid()) ftcons_ = &(*fdc);

  //Histograms for Sim Hits
  std::string detector[2] = {"Barrel", "Endcap"};
  char name[80], title[120];
  for (unsigned int k=0; k<2; ++k) {
    sprintf(name, "SimHitEn%d", k);
    sprintf(title,"Sim Hit Energy (%s)", detector[k].c_str());
    hsimE_[k] = fs_->make<TH1D>(name, title,1000,0.0,1.0);
    sprintf(name, "SimHitTm%d", k);
    sprintf(title,"Sim Hit Time (%s)", detector[k].c_str());
    hsimT_[k] = fs_->make<TH1D>(name, title,1000,0.0,500.0);
    sprintf(name, "SimHitOc%d", k);
    sprintf(title,"# Cells with Sim Hit (%s)", detector[k].c_str());
    hcell_[k] = fs_->make<TH1D>(name, title,1000,0.0,1000.0);
    sprintf(name, "SimHitPos%d", k);
    sprintf(title,"Sim Hit Eta(z)-Phi (%s) for +z", detector[k].c_str());
    hsimP_[k] = fs_->make<TH2D>(name, title,200,0,400.0,360,0,720.0);
    sprintf(name, "SimHitNeg%d", k);
    sprintf(title,"Sim Hit Eta(z)-Phi (%s) for -z", detector[k].c_str());
    hsimM_[k] = fs_->make<TH2D>(name, title,200,0,400.0,360,0,720.0);
  }
}

void FTSimHitTest::analyze(const edm::Event& e, const edm::EventSetup& ) {
  
#ifdef EDM_ML_DEBUG  
  std::cout << "FTSimHitTest:Run = " << e.id().run() << " Event = "
	    << e.id().event();
#endif

  //Barrel
  edm::Handle<edm::PSimHitContainer> hits;
  e.getByToken(tok_hitBarrel_,hits); 
#ifdef EDM_ML_DEBUG  
  std::cout << "FTSimHitTest.: PSimHitContainer for Barrel obtained with flag "
	    << hits.isValid() << std::endl;
#endif
  if (hits.isValid()) {
#ifdef EDM_ML_DEBUG 
    std::cout << "FTSimHitTest: PSimHit buffer for Barrel " << hits->size()
	      << std::endl;
#endif
    plotHits(hits,0);
  }

  //Endcap
  e.getByToken(tok_hitEndcap_,hits); 
#ifdef EDM_ML_DEBUG  
  std::cout << "FTSimHitTest.: PSimHitContainer for Endcap obtained with flag "
	    << hits.isValid() << std::endl;
#endif
  if (hits.isValid()) {
#ifdef EDM_ML_DEBUG 
    std::cout << "FTSimHitTest: PSimHit buffer for Endcap " << hits->size()
	      << std::endl;
#endif
    plotHits(hits,1);
  }
}

void FTSimHitTest::plotHits(const edm::Handle<edm::PSimHitContainer>& hits,
			    const int indx) {
#ifdef EDM_ML_DEBUG
  unsigned kount(0);
#endif
  std::vector<unsigned int> ids;
  for (edm::PSimHitContainer::const_iterator it = hits->begin();
       it != hits->end(); ++it) {
    unsigned int id  = it->detUnitId();
    double energy    = it->energyLoss();
    double time      = it->tof();
    int    etaz      = FastTimeDetId(id).ieta();
    int    phi       = FastTimeDetId(id).iphi();
    int    zside     = FastTimeDetId(id).zside();
    hsimE_[indx]->Fill(energy);
    hsimT_[indx]->Fill(time);
    if (zside > 0) hsimP_[indx]->Fill(etaz,phi);
    else           hsimM_[indx]->Fill(etaz,phi);
    if (std::find(ids.begin(),ids.end(),id) == ids.end()) ids.push_back(id);
#ifdef EDM_ML_DEBUG
    ++kount;
    std::cout << "FTSimHitTest[" << kount << "] ID " << std::hex << " " << id 
	      << std::dec << " " << FastTimeDetId(id) << " E " << energy
	      << " time " << time << std::endl;
#endif
  }
  hcell_[indx]->Fill(double(ids.size()));
#ifdef EDM_ML_DEBUG
  std::cout << "FTSimHitTest: " << ids.size() << " cells fired for type "
	    << indx << std::endl;
#endif
}

DEFINE_FWK_MODULE(FTSimHitTest);

