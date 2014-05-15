#ifndef NeutRadtuple_h
#define NeutRadtuple_h


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleSelector.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

class TTree;
class TFile;

class NeutRadtuple : public edm::EDAnalyzer
{
 public:

  explicit NeutRadtuple(const edm::ParameterSet& conf);
  virtual ~NeutRadtuple();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 protected:
  void fillEvt(const int numfs, const edm::Event& E);
  void fillTrk(const int pdgid, const int nlyrs, const float theta,
	       const float phi, const float eta, const float zee,
	       const float mom, const float eng);
  void fillLyr(const int laynm, const float radln, const float layRpos, const float layZpos);
  
 private:
  edm::ParameterSet conf_;

  void init();

  //--- Structures for ntupling:                                                                                             
  struct evt
  {
    int run, evtnum;
    int numfs;
    void init();
  } evt_;

  struct trk
  {
    int pdgid;
    int nlyrs;
    float theta;
    float phi;
    float eta;
    float zee;
    float mom;
    float eng;
    void init();
  } trk_;

  struct lyr
  {
    int laynm;
    float radln;
    float layRpos;
    float layZpos;
    void init();
  } lyr_;

  TFile * tfile_;
  TTree * tptree_;

};

#endif
