#ifndef MuonToSimAssociatorBase_h
#define MuonToSimAssociatorBase_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"


class TrackerTopology;

class MuonToSimAssociatorBase  {
  
 public:
  
  MuonToSimAssociatorBase ();
  virtual ~MuonToSimAssociatorBase();
  
  enum MuonTrackType { InnerTk, OuterTk, GlobalTk, Segments };

  struct RefToBaseSort { 
    template<typename T> bool operator()(const edm::RefToBase<T> &r1, const edm::RefToBase<T> &r2) const { 
        return (r1.id() == r2.id() ? r1.key() < r2.key() : r1.id() < r2.id()); 
    }
  };
  typedef std::map<edm::RefToBase<reco::Muon>, std::vector<std::pair<TrackingParticleRef, double> >, RefToBaseSort> MuonToSimCollection;
  typedef std::map<TrackingParticleRef, std::vector<std::pair<edm::RefToBase<reco::Muon>, double> > >               SimToMuonCollection;


  virtual void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                              const edm::RefToBaseVector<reco::Muon> &, MuonTrackType ,
                              const edm::RefVector<TrackingParticleCollection>&,
                              const edm::Event * event = 0, const edm::EventSetup * setup = 0) const = 0; 

  virtual void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                              const edm::Handle<edm::View<reco::Muon> > &, MuonTrackType , 
                              const edm::Handle<TrackingParticleCollection>&,
                              const edm::Event * event = 0, const edm::EventSetup * setup = 0) const = 0;

};

#endif
