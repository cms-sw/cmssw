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
#include "SimTracker/TrackAssociation/interface/TrackAssociator.h"
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
TrackAssociator::TrackAssociator(const edm::Event& e, const edm::ParameterSet& conf) : myEvent_(e), conf_(conf)  
{
  //  std::cout << "\nEvent ID = "<< e.id() << std::endl ;
  
  std::cout << "TrackAssociator Constructor " << std::endl;

  //prepare hit based association
  associate = new TrackerHitAssociator::TrackerHitAssociator(e, conf);
}


/* Destructor */

TrackAssociator::~TrackAssociator()
{
  //do cleanup here
}

//
//---member functions
//

RecoToSimCollection  TrackAssociator::associateRecoToSim(edm::Handle<reco::TrackCollection>& trackCollectionH,
							       edm::Handle<TrackingParticleCollection>&  TPCollectionH)
{    

  const float minHitFraction = 0;
  std::vector<unsigned int> SimTrackIds;
  std::vector<unsigned int> matchedIds; 
  RecoToSimCollection  outputCollection;
  
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());
  std::cout << "Found " << tPC.size() << " TrackingParticles" << std::endl;

  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 
  std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;


  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++)
    {
      std::cout <<"\n Track # " << tindex << "\tNumber of RecHits "<<track->recHitsSize()<<std::endl;
      matchedIds.clear();
      int nmax=0;
      int idmax=-1;
      int ri=0;
      int tnmax=0;
      int tidmax=-1;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	if ((*it)->isValid()){
	  ri++;
	  SimTrackIds.clear();	  
	  SimTrackIds = associate->associateHitId((**it));
	  nmax = 0;
	  idmax = -1;
	  if(!SimTrackIds.empty()){
	    for(size_t j=0; j<SimTrackIds.size(); j++){
	      int n =0;
	      n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	      if(n>nmax){
		nmax = n;
		idmax = SimTrackIds[j];
	      }
	    }
	  }
	}else{
	  std::cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<std::endl;
	}
	matchedIds.push_back(idmax);
      }
      //save id for the track
      if(!matchedIds.empty()){
	tnmax = 0;
	tidmax = -1;
	//	std::cout << " track = " << i << " totrh = " << ri << " tot matches = " << matchedIds.size() << std::endl;
	for(size_t j=0; j<matchedIds.size(); j++){
	  int n =0;
	  n = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
	  //std::cout << " sim ID = " << matchedIds[j] << " Occurrence = " << n << std::endl; 
	  if(n>tnmax){
	    tnmax = n;
	    tidmax = matchedIds[j];
	  }
	}
	float fraction = tnmax/ri ;
	std::cout << " bestID = " << tidmax << " Occurrence = " << tnmax << " fraction = " << fraction << std::endl; 
	//note simtrackId == geantID+1
      }
      //now loop over TPcollection and save the appropriate tracks
      int tpindex =0;
      for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();
	     g4T !=  t -> g4Track_end(); ++g4T) {
	  if((*g4T)->trackId() == (tidmax-1)){
	    //	    std::cout << " found match " << std::endl;
	    std::cout << "  G4  Track Momentum " << (*g4T)->momentum() << std::endl;   
	    std::cout << "  reco Track Momentum " << track->momentum() << std::endl;   
	    outputCollection.insert(reco::TrackRef(trackCollectionH,tindex), 
				    std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex),0));
	  }
	}
      }
    }
  return outputCollection;
}

