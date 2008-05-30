
// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <TH1F.h>

class XtalDedxAnalysis : public edm::EDAnalyzer {

public:

  explicit XtalDedxAnalysis(const edm::ParameterSet&);
  virtual ~XtalDedxAnalysis() {}

protected:

  virtual void beginJob(const edm::EventSetup&) {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

  void analyzeHits (std::vector<PCaloHit>&);

private:
  edm::InputTag  caloHitSource_;

  TH1F           *meNHit_, *meE1T0_, *meE9T0_, *meE1T1_, *meE9T1_;

};

XtalDedxAnalysis::XtalDedxAnalysis(const edm::ParameterSet& ps) {

  caloHitSource_   = ps.getParameter<edm::InputTag>("caloHitSource");
  double energyMax = ps.getParameter<double>("EnergyMax");
  edm::LogInfo("CherenkovAnalysis") << "XtalDedxAnalysis::Source "
				    << caloHitSource_ << " Energy Max "
				    << energyMax;

  // Book histograms
  edm::Service<TFileService> tfile;

  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  //Histograms for Hits
  meNHit_= tfile->make<TH1F>("Hits","Number of Hits",1000,0.,5000.);
  meNHit_->GetXaxis()->SetTitle("Number of hits");
  meE1T0_= tfile->make<TH1F>("E1T0","Energy Max (all time)",1000,0,energyMax);
  meE1T0_->GetXaxis()->SetTitle("E1 (GeV)");
  meE9T0_= tfile->make<TH1F>("E9T0","Energy All (all time)",1000,0,energyMax);
  meE9T0_->GetXaxis()->SetTitle("E9 (GeV)");
  meE1T1_= tfile->make<TH1F>("E1T1","Energy Max (t<400 ns)",1000,0,energyMax);
  meE1T1_->GetXaxis()->SetTitle("E1 (GeV) (t < 400 ns)");
  meE9T1_= tfile->make<TH1F>("E9T1","Energy All (t<400 ns)",1000,0,energyMax);
  meE9T1_->GetXaxis()->SetTitle("E9 (GeV) (t < 400 ns)");
}

void XtalDedxAnalysis::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("CherenkovAnalysis") << "Run = " << e.id().run() << " Event = "
				<< e.id().event();

  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> pCaloHits;
  e.getByLabel(caloHitSource_, pCaloHits);

  if (pCaloHits.isValid()) {
    caloHits.insert(caloHits.end(),pCaloHits->begin(),pCaloHits->end());
    LogDebug("CherenkovAnalysis") << "HcalValidation: Hit buffer "
				  << caloHits.size();
    analyzeHits (caloHits);
  }
}

void XtalDedxAnalysis::analyzeHits (std::vector<PCaloHit>& hits) {

  int nHit = hits.size();
  double e10=0, e90=0, e11=0, e91=0;
  for (int i=0; i<nHit; i++) {
    double energy    = hits[i].energy();
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    e90  += energy;
    if (time < 400) e91 += energy;
    if (id_ == 22) {
      e10 += energy;
      if (time < 400) e11 += energy;
    }
    LogDebug("CherenkovAnalysis") << "Hit[" << i << "] ID " << id_ << " E " 
				  << energy << " time " << time;
  }
  meNHit_->Fill(double(nHit));
  meE1T0_->Fill(e10);
  meE9T0_->Fill(e90);
  meE1T1_->Fill(e11);
  meE9T1_->Fill(e91);
}

//define this as a plug-in
DEFINE_FWK_MODULE(XtalDedxAnalysis);
