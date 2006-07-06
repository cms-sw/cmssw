#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociator.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"


/* Associate SimTracks to RecoTracks By Hits */

void TrackAssociatorByHits::AssociateByHitsRecoTrack(const edm::SimTrackContainer& SimTk,
						     const reco::TrackCollection& tracks,
						     const float minHitFraction){    
  
  SimToRecoCollection result;
  std::vector<unsigned int> SimTrackIds;
  //get the ID of the recotrack  by hits 
  int i=1;
  for (reco::TrackCollection::const_iterator track=tracks.begin(); track!=tracks.end(); track++)
    {
      std::cout <<"\t\tNumber of RecHits "<<track->recHitsSize()<<std::endl;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	if ((*it)->isValid()){
	  SimTrackIds.clear();	  
	  SimTrackIds = associate->associateHitId((**it));
	  if(!SimTrackIds.empty()) std::cout << "\t\t\tMatched Ids =   " << SimTrackIds.size() << std::endl;
	}else{
	  std::cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<std::endl;
	}
      }
      int nmax = 0;
      int idmax = -1;
      for(size_t j=0; j<SimTrackIds.size(); j++){
	int n =0;
	n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	std::cout << " Tracks # of rechits = " << track->recHitsSize() << " found match = " << SimTrackIds.size() << std::endl;
	std::cout << " rechit = " << i << " sim ID = " << SimTrackIds[i] << " Occurrence = " << n << std::endl; 
	if(n>nmax){
	  nmax = n;
	  idmax = SimTrackIds[i];
	}
      }
      float totsim = nmax;
      float tothits = track->recHitsSize();//include pixel as well...
      float fraction = totsim/tothits ;
      
      std::cout << " Track id # " << i << "# of rechits = " << track->recHitsSize() << " matched simtrack id= " << idmax 
		<< " fraction = " << fraction << std::endl;
      i++;
    }
}

/* Constructor */
TrackAssociatorByHits::TrackAssociatorByHits(const edm::Event& e)  : myEvent_(e)  {
  
  std::cout << "\nEvent ID = "<< e.id() << std::endl ;
  
  //this are not the proper ttracks and they do not have an id
  //get simtracks
  edm::Handle<edm::SimTrackContainer> SimTk;
  e.getByType(SimTk); 
  //fill map of simtrack and id
  SimTrackIdMap.clear();
  edm::SimTrackContainer::const_iterator itTrk;
  for (itTrk = SimTk->begin(); itTrk != SimTk->end();++itTrk) {
    SimTrackIdMap[(*itTrk).genpartIndex()].push_back((*itTrk));
  }    
  
  
  //get reco tracks
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByType(trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product()); 
  std::cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl ;
  
  //prepare hit based association
  associate = new TrackerHitAssociator::TrackerHitAssociator(e);
}
