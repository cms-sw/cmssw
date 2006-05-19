
#include "SimTracker/TrackerHitAssociation/test/myTrackAnalyzer.h"
#include "Math/GenVector/BitReproducible.h"

#include <memory>
#include <iostream>
#include <string>

using namespace edm;
class TrackerHitAssociator;

namespace cms
{
  void myTrackAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
  {
    //
    // extract tracker geometry
    //
    edm::ESHandle<TrackerGeometry> theG;
    setup.get<TrackerDigiGeometryRecord>().get(theG);
    
    using namespace std;
    
    std::cout << "\nEvent ID = "<< event.id() << std::endl ;
    
    edm::Handle<reco::TrackCollection> trackCollection;
    //    event.getByLabel("trackp", trackCollection);
    event.getByType(trackCollection);
    
    //get simtrack info
   std::vector<EmbdSimTrack> theSimTracks;
   std::vector<EmbdSimVertex> theSimVertexes;

   Handle<EmbdSimTrackContainer> SimTk;
   Handle<EmbdSimVertexContainer> SimVtx;
   event.getByLabel("SimG4Object",SimTk);
   event.getByLabel("SimG4Object",SimVtx);
   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
   theSimVertexes.insert(theSimVertexes.end(),SimVtx->begin(),SimVtx->end());

    //NEW
    std::vector<PSimHit> matched;
    TrackerHitAssociator associate(event);
    
    const reco::TrackCollection tC = *(trackCollection.product());
    
    std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;
    
    int i=1;
    for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
      std::cout << "Track number "<< i << std::endl ;
      std::cout << "\tmomentum: " << track->momentum()<< std::endl;
      std::cout << "\tPT: " << track->pt()<< std::endl;
      std::cout << "\tvertex: " << track->vertex()<< std::endl;
      std::cout << "\timpact parameter: " << track->d0()<< std::endl;
      std::cout << "\tcharge: " << track->charge()<< std::endl;
      std::cout << "\tnormalizedChi2: " << track->normalizedChi2()<< std::endl;
      i++;
      cout<<"\tFrom EXTRA : "<<endl;
      cout<<"\t\touter PT "<< track->outerPt()<<endl;
      //
      // try and access Hits
      //
      int count =0;
      cout <<"\t\tNumber of RecHits "<<track->recHitsSize()<<endl;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	if ((*it)->isValid()){
	  cout <<"\t\t\tRecHit on det "<<(*it)->geographicalId().rawId()<<endl;
	  cout <<"\t\t\tRecHit in LP "<<(*it)->localPosition()<<endl;
	  cout <<"\t\t\tRecHit in GP "<<theG->idToDet((*it)->geographicalId())->surface().toGlobal((*it)->localPosition()) <<endl;
	  
	  
	  //try SimHit matching
	  count++;
	  matched.clear();	  
	  matched = associate.associateHit((**it));
	  if(!matched.empty()){
	    cout << "\t\t\tmatched  " << matched.size() << endl;
	    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	      cout << "\t\t\tSimhit  ID  " << (*m).trackId() 
		   << "\t\t\tSimhit  LP  " << (*m).localPosition() 
		   << "\t\t\tSimhit  GP  " << theG->idToDet((*it)->geographicalId())->surface().toGlobal((*m).localPosition()) << endl;   
	      cout << "Track parameters " << theSimTracks[(*m).trackId()].momentum() << endl;
	    }

	    //now figure out which is the majority of the ids
	    //myid[count] = matched[0].trackId();
	  }
	//now get the track parameters
	//simTk[myid]
	}else{
	  cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<endl;
	} 
      }
    }
  }
  
}



