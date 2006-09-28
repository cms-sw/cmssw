//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

using namespace reco;

/* Constructor */
TrackAssociatorByHits::TrackAssociatorByHits (const edm::ParameterSet& conf) :  conf_(conf)  
{
}


/* Destructor */
TrackAssociatorByHits::~TrackAssociatorByHits()
{
  //do cleanup here
}

//
//---member functions
//

RecoToSimCollection  
TrackAssociatorByHits::associateRecoToSim(edm::Handle<reco::TrackCollection>& trackCollectionH,
					  edm::Handle<TrackingParticleCollection>&  TPCollectionH,     
					  const edm::Event * e ){

  const float minHitFraction = 0;
  int nshared =0;
  //  float fraction=0;
  std::vector<unsigned int> SimTrackIds;
  std::vector<unsigned int> matchedIds; 
  RecoToSimCollection  outputCollection;
  
  TrackerHitAssociator * associate = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);

  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 


  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++)
    {
      matchedIds.clear();
      int ri=0;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	if ((*it)->isValid()){
	  ri++;
	  SimTrackIds.clear();	  
	  SimTrackIds = associate->associateHitId((**it));
	  //save all the id of matched simtracks
	  if(!SimTrackIds.empty()){
	    for(size_t j=0; j<SimTrackIds.size(); j++){
	      //std::cout << " hit # " << ri << " SimId " << SimTrackIds[j] << std::endl; 
	      matchedIds.push_back(SimTrackIds[j]);
	    }
	  }
	}else{
	  std::cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<std::endl;
	}
      }
      //save id for the track
      if(!matchedIds.empty()){
	unsigned int simtrackid_cache = 9999999;
	nshared =0;
	for(size_t j=0; j<matchedIds.size(); j++){
	  if(matchedIds[j] != simtrackid_cache) {
	    //only the first time we see this ID 
	    simtrackid_cache = matchedIds[j];
	    int tpindex =0;
	    for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) 
	      {
		for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();
		     g4T !=  t -> g4Track_end(); ++g4T) {
		  if((*g4T)->trackId() == matchedIds[j]){
		    // 		    std::cout << " Match ID = " << (*g4T)->trackId() 
		    // 			      << " Nrh = " << ri << " Nshared = " << n << std::endl;
		    // 		    std::cout << " G4  Track Momentum " << (*g4T)->momentum() << std::endl;   
		    // 		    std::cout << " reco Track Momentum " << track->momentum() << std::endl;  
		    //		    if(ri!=0) fraction = n/ri;
		    nshared = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
		    //std::cout << " sim ID = " << matchedIds[j] << " Occurrence = " << nshared << std::endl; 
		    //for now save the number of shared hits between the reco and sim track
		    //cut on the fraction
		    outputCollection.insert(reco::TrackRef(trackCollectionH,tindex), 
					    std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex),
							   nshared));
		  }
		}
	      }
	  }
	}
      }
    }
  delete associate;
  return outputCollection;
}


SimToRecoCollection  
TrackAssociatorByHits::associateSimToReco(edm::Handle<reco::TrackCollection>& trackCollectionH,
					  edm::Handle<TrackingParticleCollection>&  
					  TPCollectionH, 
					  const edm::Event * e ){
  
  const float minHitFraction = 0;
  //float fraction=0;
  int nshared = 0;
  std::vector<unsigned int> SimTrackIds;
  std::vector<unsigned int> matchedIds; 
  SimToRecoCollection  outputCollection;

  TrackerHitAssociator * associate = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());
//   std::cout << "Found " << tPC.size() << " TrackingParticles" << std::endl;
  
  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 
//   std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;


  //get the ID of the recotrack  by hits 

  int tindex=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++)
    {
      matchedIds.clear();
      int ri=0;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	if ((*it)->isValid()){
	  ri++;
	  SimTrackIds.clear();	  
	  SimTrackIds = associate->associateHitId((**it));
	  if(!SimTrackIds.empty()){
	    for(size_t j=0; j<SimTrackIds.size(); j++){
	      //std::cout << " hit # " << ri << " SimId " << SimTrackIds[j] << std::endl; 
	      matchedIds.push_back(SimTrackIds[j]);
	    }
	  }
	}else{
	  std::cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<std::endl;
	}
      }
      //save id for the track
      if(!matchedIds.empty()){
	unsigned int simtrackid_cache = 9999999;
	nshared =0;
	for(size_t j=0; j<matchedIds.size(); j++){
	  if(matchedIds[j] != simtrackid_cache) {
	    //only the first time we see this ID 
	    simtrackid_cache = matchedIds[j];	  
	    int tpindex =0;
	    for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	      for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();
		   g4T !=  t -> g4Track_end(); ++g4T) {
		if((*g4T)->trackId() == matchedIds[j]){
		  nshared = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
		  int nsimhit = t->trackPSimHit().size(); 
		  //	  if(nsimhit!=0) fraction = n/nsimhit;
		  // 		  std::cout << " Match ID = " << (*g4T)->trackId() 
		  // 			    << " Nrh = " << ri << " Nshared = " << n << " Nsim = " << nsimhit << std::endl;
		  // 		  std::cout << " G4  Track Momentum " << (*g4T)->momentum() << std::endl;   
		  // 		  std::cout << " reco Track Momentum " << track->momentum() << std::endl;  
		  //NOTE: not sorted for quality yet
		  //for now save the number of shared hits: n instead of the fraction 
		  //Until we solve the problem with the number of Simhits associated to the TP
		  outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex), 
					  std::make_pair(reco::TrackRef(trackCollectionH,tindex),nshared));
		}
	      }
	    }
	  }
	}
      }
    }
  delete associate;
  return outputCollection;
}
  
