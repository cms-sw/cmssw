#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
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

/* Associate SimTracks to RecoTracks By Hits */

void TrackAssociatorByHits::AssociateByHitsRecoTrack(const reco::TrackCollection& tC,
						     const float minHitFraction){    
  
  SimToRecoCollection result;
  std::vector<unsigned int> SimTrackIds;
  //get the ID of the recotrack  by hits 
  int i=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++)
    {
      i++;
      std::cout <<"\n Track # " << i << "\tNumber of RecHits "<<track->recHitsSize()<<std::endl;
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
	      //    std::cout << " Rechit = " << ri << " matches = " << SimTrackIds.size() 
	      //		<< " sim ID = " << SimTrackIds[j] << " Occurrence = " << n << std::endl; 
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
      RecoToSimCollection outputCollection;
      for (TrackingParticleCollection::const_iterator t = tPC -> begin(); 
	   t != tPC -> end(); ++t) {
	for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();
	     g4T !=  t -> g4Track_end(); ++g4T) {
	  if((*g4T)->trackId() == (tidmax-1)){
	    std::cout << " found match " << std::endl;
	    std::cout << "  G4  Track Momentum " << (*g4T)->momentum() << std::endl;   
	    std::cout << "  reco Track Momentum " << track->momentum() << std::endl;   
	    // outputCollection.insert(edm::Ref<TrackCollection>(tC,track), edm::Ref<TrackingParticleCollection>(tPC, t));
	  }
	}
      }
    }
  
}

/* Constructor */
TrackAssociatorByHits::TrackAssociatorByHits(const edm::Event& e, const edm::ParameterSet& conf) : myEvent_(e), conf_(conf)  
{
  
  std::cout << "\nEvent ID = "<< e.id() << std::endl ;
  
  //get TrackingParticle from the event
  edm::Handle<TrackingParticleCollection>  TruthTrackContainer ;
  e.getByType(TruthTrackContainer );
  tPC   = TruthTrackContainer.product();
  std::cout << "Found " << tPC->size() << " TP tracks" <<std::endl;
  int count = 0; 
  std::cout << "Dumping out sample track info" << std::endl;
  for (TrackingParticleCollection::const_iterator t = tPC -> begin(); 
       t != tPC -> end(); ++t, ++count) {
    
    //     for (TrackingParticle::genp_iterator hepT = t -> genParticle_begin();
    //          hepT !=  t -> genParticle_end(); ++hepT) {
    //       std::cout << " Gen Track PDG ID   " <<  (*hepT)->pdg_id() << std::endl;    
    //       std::cout << "  Gen Track Momentum " << (*hepT)->momentum() << std::endl;    
    //     }
    for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();
	 g4T !=  t -> g4Track_end(); ++g4T) {
      std::cout << "  G4  Track Momentum " << (*g4T)->momentum() << std::endl;   
      std::cout << "  G4  Track Id       " << (*g4T)->trackId() << std::endl;   
    }
  }  
  
  //prepare hit based association
  associate = new TrackerHitAssociator::TrackerHitAssociator(e, conf_);
}
