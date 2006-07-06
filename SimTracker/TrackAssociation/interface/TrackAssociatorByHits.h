#ifndef TrackAssociation_TrackAssociatorByHits_h
#define TrackAssociation_TrackAssociatorByHits_h

#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociator.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/EDProduct.h"

class TrackAssociator;
class Track;
class SimTrack;

class TrackAssociatorByHits : public TrackAssociator {
  
 public:
  typedef edm::AssociationMap<edm::OneToMany<edm::SimTrackContainer, reco::TrackCollection, unsigned int> > SimToRecoCollection;  
  typedef edm::AssociationMap<edm::OneToMany<reco::TrackCollection, edm::SimTrackContainer, unsigned int> > RecoToSimCollection;  
  
  /* Constructor */
  /* Need to pass the event in the constructor. Will be modified once TrackingParticle is final */
  TrackAssociatorByHits(const edm::Event& e);
  
  virtual ~TrackAssociatorByHits (){} 
  
  //method
 void  AssociateByHitsRecoTrack(const edm::SimTrackContainer&,
				const reco::TrackCollection&,
				const float minFractionOfHits = 0.);
  
 private:
  const edm::Event& myEvent_; 
  const float theMinHitFraction;    
  TrackerHitAssociator* associate;
  typedef std::map<unsigned int, std::vector<SimTrack> > simtrk_map;
  typedef simtrk_map::iterator simtrk_map_iterator;
  simtrk_map  SimTrackIdMap;
};
			      
#endif // TrackAssociation_TrackAssociatorByHits_h
			      
