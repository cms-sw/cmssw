
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
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <TH1F.h>

class XtalDedxAnalysis : public edm::EDAnalyzer {

public:

  explicit XtalDedxAnalysis(const edm::ParameterSet&);
  virtual ~XtalDedxAnalysis() {}

protected:

  virtual void beginJob(const edm::EventSetup&) {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

  void analyzeHits (std::vector<PCaloHit>&, std::vector<SimTrack>&);

private:
  edm::InputTag  caloHitSource_;
  std::string    simTkLabel_;

  TH1F           *meNHit_[4], *meE1T0_[4], *meE9T0_[4], *meE1T1_[4], *meE9T1_[4];

};

XtalDedxAnalysis::XtalDedxAnalysis(const edm::ParameterSet& ps) {

  caloHitSource_   = ps.getParameter<edm::InputTag>("caloHitSource");
  simTkLabel_      = ps.getUntrackedParameter<std::string>("moduleLabelTk","g4SimHits");
  double energyMax = ps.getParameter<double>("EnergyMax");
  edm::LogInfo("CherenkovAnalysis") << "XtalDedxAnalysis::Source "
				    << caloHitSource_ << " Track Label "
				    << simTkLabel_ << " Energy Max "
				    << energyMax;

  // Book histograms
  edm::Service<TFileService> tfile;

  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  //Histograms for Hits
  std::string types[4] = {"total", "by dE/dx", "by delta-ray", "by bremms"};
  char name[20], title[80];
  for (int i=0; i<4; i++) {
    sprintf (name,  "Hits%d", i);
    sprintf (title, "Number of hits (%s)", types[i].c_str());
    meNHit_[i]= tfile->make<TH1F>(name, title, 1000, 0., 5000.);
    meNHit_[i]->GetXaxis()->SetTitle(title);
    meNHit_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name,  "E1T0%d", i);
    sprintf (title, "E1 (Loss %s) in GeV", types[i].c_str());
    meE1T0_[i] = tfile->make<TH1F>(name, title, 1000, 0, energyMax);
    meE1T0_[i]->GetXaxis()->SetTitle(title);
    meE1T0_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name,  "E9T0%d", i);
    sprintf (title, "E9 (Loss %s) in GeV", types[i].c_str());
    meE9T0_[i] = tfile->make<TH1F>(name, title, 1000, 0, energyMax);
    meE9T0_[i]->GetXaxis()->SetTitle(title);
    meE9T0_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name,  "E1T1%d", i);
    sprintf (title, "E1 (Loss %s with t < 400 ns) in GeV", types[i].c_str());
    meE1T1_[i] = tfile->make<TH1F>(name, title, 1000, 0, energyMax);
    meE1T1_[i]->GetXaxis()->SetTitle(title);
    meE1T1_[i]->GetYaxis()->SetTitle("Events");
    sprintf (name,  "E9T1%d", i);
    sprintf (title, "E9 (Loss %s with t < 400 ns) in GeV", types[i].c_str());
    meE9T1_[i]= tfile->make<TH1F>(name, title, 1000, 0, energyMax);
    meE9T1_[i]->GetXaxis()->SetTitle(title);
    meE9T1_[i]->GetYaxis()->SetTitle("Events");
  }
}

void XtalDedxAnalysis::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("CherenkovAnalysis") << "Run = " << e.id().run() << " Event = "
				<< e.id().event();

  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> pCaloHits;
  e.getByLabel(caloHitSource_, pCaloHits);

  std::vector<SimTrack> theSimTracks;
  edm::Handle<edm::SimTrackContainer> simTk;
  e.getByLabel(simTkLabel_,simTk);
  theSimTracks.insert(theSimTracks.end(),simTk->begin(),simTk->end());

  if (pCaloHits.isValid()) {
    caloHits.insert(caloHits.end(),pCaloHits->begin(),pCaloHits->end());
    LogDebug("CherenkovAnalysis") << "HcalValidation: Hit buffer "
				  << caloHits.size();
    analyzeHits (caloHits, theSimTracks);
  }
}

void XtalDedxAnalysis::analyzeHits (std::vector<PCaloHit>& hits,
				    std::vector<SimTrack>& tracks) {

  int nHit = hits.size();
  double e10[4], e90[4], e11[4], e91[4], hit[4];
  for (int i=0; i<4; i++)
    e10[i] = e90[i] = e11[i] = e91[i] = hit[i] = 0;
  for (int i=0; i<nHit; i++) {
    double energy    = hits[i].energy();
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    int    trackID   = hits[i].geantTrackId();
    int    type      = 1;
    for (unsigned int k=0; k<tracks.size(); k++) {
      if (trackID == (int)(tracks[k].trackId())) {
	int thePID = tracks[k].type();
	if      (thePID == -11 || thePID == 11) type = 2;
	else if (thePID != -13 && thePID != 13) type = 3;
	break;
      }
    }
    hit[0]++;
    hit[type]++;
    e90[0]    += energy;
    e90[type] += energy;
    if (time < 400) {
      e91[0]    += energy;
      e91[type] += energy;
    }
    if (id_ == 22) {
      e10[0]    += energy;
      e10[type] += energy;
      if (time < 400) {
	e11[0]    += energy;
	e11[type] += energy;
      }
    }
    LogDebug("CherenkovAnalysis") << "Hit[" << i << "] ID " << id_ << " E " 
				  << energy << " time " << time << " track "
				  << trackID << " type " << type;
  }
  for (int i=0; i<4; i++) {
    LogDebug("CherenkovAnalysis") << "Type[" << i << "] Hit " << hit[i] 
				  << " E10 " << e10[i] << " E11 " << e11[i]
				  << " E90 " << e90[i] << " E91 " << e91[i];
    meNHit_[i]->Fill(hit[i]);
    meE1T0_[i]->Fill(e10[i]);
    meE9T0_[i]->Fill(e90[i]);
    meE1T1_[i]->Fill(e11[i]);
    meE9T1_[i]->Fill(e91[i]);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(XtalDedxAnalysis);
