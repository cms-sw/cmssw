#ifndef TrackAssociator_h
#define TrackAssociator_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociation.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class Track;
class ParticleTrack;

class TrackAssociator  {
  
 public:
  explicit TrackAssociator(const edm::Event&, const edm::ParameterSet&);  
  ~TrackAssociator();
  
/* Associate SimTracks to RecoTracks By Hits */
  reco::RecoToSimCollection  AssociateByHitsRecoTrack(const float minFractionOfHits = 0.) const;

/* Associate SimTracks to RecoTracks By Pulls */
  reco::RecoToSimCollection  AssociateByPullsRecoTrack() const;

 private:
  // ----- member data
  const edm::Event& myEvent_; 
  const edm::ParameterSet& conf_;
  const float theMinHitFraction;    
  TrackerHitAssociator* associate;
};

#endif
