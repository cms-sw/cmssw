#ifndef MultiTrackValidator_h
#define MultiTrackValidator_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Validation/RecoTrack/interface/TrackEfficiencySelector.h"
#include "Validation/RecoTrack/interface/TPEfficiencySelector.h"
#include "Validation/RecoTrack/interface/TrackFakeRateSelector.h"

#include <iostream>
#include <string>
#include <TH1F.h>

using namespace edm;
using namespace std;

class MultiTrackValidator : public edm::EDAnalyzer {
 public:

  MultiTrackValidator(const edm::ParameterSet& pset):
    dbe_(0),
    sim(pset.getParameter<string>("sim")),
    label(pset.getParameter< vector<string> >("label")),
    associators(pset.getParameter< vector<string> >("associators")),
    out(pset.getParameter<string>("out")),
    min(pset.getParameter<double>("min")),
    max(pset.getParameter<double>("max")),
    nint(pset.getParameter<int>("nint")),
    selectTracks4Efficiency(pset.getParameter<edm::ParameterSet>("TracksEfficCuts")),
    selectTPs4Efficiency(pset.getParameter<edm::ParameterSet>("TPEfficCuts")),
    selectTracks4FakeRate(pset.getParameter<edm::ParameterSet>("TracksFakeRateCuts")),
    selectTPs4FakeRate(pset.getParameter<edm::ParameterSet>("TPFakeRateCuts"))
    {
      dbe_ = Service<DaqMonitorBEInterface>().operator->();
    }
  
  ~MultiTrackValidator(){ }

  void beginJob( const EventSetup &);
  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  void endJob();

 private:

  DaqMonitorBEInterface* dbe_;

  string sim;
  vector<string> label, associators;
  string out;
  double  min, max;
  int nint;
  
  vector<MonitorElement*> h_ptSIM, h_etaSIM, h_tracksSIM, h_vertposSIM;
  vector<MonitorElement*> h_tracks, h_fakes, h_nchi2, h_nchi2_prob, h_hits,  h_ptrmsh, h_d0rmsh, h_charge;
  vector<MonitorElement*> h_effic, h_fakerate, h_reco, h_assoc, h_assoc2, h_simul;
  vector<MonitorElement*> h_pt, h_eta, h_pullTheta,h_pullPhi0,h_pullD0,h_pullDz,h_pullK;
  vector<MonitorElement*> chi2_vs_nhits, chi2_vs_eta, nhits_vs_eta, ptres_vs_eta, etares_vs_eta, nrec_vs_nsim;
  vector<MonitorElement*> h_assochi2, h_assochi2_prob, h_hits_eta;
  
  vector< vector<double> > etaintervals;
  vector< vector<double> > hitseta;
  vector< vector<int> > totSIM,totREC,totASS,totASS2;
  
  vector< vector<TH1F*> > ptdistrib;
  vector< vector<TH1F*> > d0distrib;

  edm::ESHandle<MagneticField> theMF;

  vector<TrackAssociatorBase*> associator;
  TrackAssociatorByChi2 * associatorForParamAtPca;
  TrackEfficiencySelector selectTracks4Efficiency;
  TPEfficiencySelector selectTPs4Efficiency;
  TrackFakeRateSelector   selectTracks4FakeRate;
  TPEfficiencySelector selectTPs4FakeRate;
};


#endif
