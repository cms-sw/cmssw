#ifndef TrackAssociatorByHits_h
#define TrackAssociatorByHits_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

/* class Track; */
/* class ParticleTrack; */

class TrackAssociatorByHits : public TrackAssociatorBase {
  
 public:
  explicit TrackAssociatorByHits( const edm::ParameterSet& );  
  ~TrackAssociatorByHits();
  
/* Associate SimTracks to RecoTracks By Hits */

  reco::RecoToSimCollection associateRecoToSim (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>&, 
						const edm::Event * event = 0) const;
  
  reco::SimToRecoCollection associateSimToReco (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>&, 
						const edm::Event * event = 0) const;

 
 private:
  // ----- member data
  const edm::ParameterSet& conf_;
  const bool AbsoluteNumberOfHits;    
  const double theMinHitCut;    
  int LayerFromDetid(const DetId&) const;
};

#endif
