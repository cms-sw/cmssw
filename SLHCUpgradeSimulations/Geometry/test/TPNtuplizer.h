#ifndef TPNtuplizer_h
#define TPNtuplizer_h

/** \class TPNtuplizer
 * 
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

class TTree;
class TFile;

class TPNtuplizer : public edm::EDAnalyzer
{
 public:
  
  explicit TPNtuplizer(const edm::ParameterSet& conf);
  virtual ~TPNtuplizer();
  /// Method called before the event loop
  void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 protected:

  void fillEvt(const int numtp, const int nseltp, const int nfdtp,
               const int numtk, const int nasstk, const edm::Event& );
  void fillTP(const int num, const int matched_hit, const float quality, 
              const int selected, const TrackingParticle* tp );

 private:
  edm::ParameterSet conf_;
  std::vector<edm::InputTag> label_;
  edm::InputTag label_tp_effic_;
  edm::InputTag label_tp_fake_;
  bool UseAssociators_;
  std::vector<std::string> associators_;
  TrackingParticleSelector tpSelector_;
  std::vector<const reco::TrackToTrackingParticleAssociator*> associator_;

  void init();
  
  //--- Structures for ntupling:
  struct evt
  {
    int run, evtnum;
    int numtp, nseltp, nfdtp, numtk, nasstk;
    void init();
  } evt_;
  
  struct myTp 
  {
    // signal is really in-time (with signal crossing)
    int tpn, bcross, tevt, charge, stable, status, pdgid, mathit, signal, llived, sel, gpsz, gpstat;
    float pt, eta, tip, lip;
    float p, e, phi, theta, rap;
    float qual;
    void init();
  } tp_;

  TFile * tfile_;
  TTree * tptree_;
};

#endif
