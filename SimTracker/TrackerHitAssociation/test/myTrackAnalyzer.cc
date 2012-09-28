
#include "SimTracker/TrackerHitAssociation/test/myTrackAnalyzer.h"
#include "Math/GenVector/BitReproducible.h"

#include <memory>
#include <iostream>
#include <string>

using namespace edm;
class TrackerHitAssociator;

myTrackAnalyzer::myTrackAnalyzer(edm::ParameterSet const& conf) : 
  conf_(conf),
  doPixel_( conf.getParameter<bool>("associatePixel") ),
  doStrip_( conf.getParameter<bool>("associateStrip") ),
  trackCollectionTag_(conf.getParameter<edm::InputTag>("trackCollectionTag")) {
}

myTrackAnalyzer::~myTrackAnalyzer()
{
}



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
    event.getByLabel(trackCollectionTag_, trackCollection);
    
    //get simtrack info
   std::vector<SimTrack> theSimTracks;
   std::vector<SimVertex> theSimVertexes;

   edm::Handle<SimTrackContainer> SimTk;
   edm::Handle<SimVertexContainer> SimVtx;
   event.getByLabel("g4SimHits",SimTk);
   event.getByLabel("g4SimHits",SimVtx);
   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
   theSimVertexes.insert(theSimVertexes.end(),SimVtx->begin(),SimVtx->end());

    if(!doPixel_ && !doStrip_)  throw edm::Exception(errors::Configuration,"Strip and pixel association disabled");
    //NEW
    std::vector<PSimHit> matched;
    TrackerHitAssociator associate(event, conf_);
    std::vector<unsigned int> SimTrackIds;

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
      cout<<"\tFrom EXTRA : "<<endl;
      cout<<"\t\touter PT "<< track->outerPt()<<endl;
      //
      // try and access Hits
      //
      SimTrackIds.clear();
      cout <<"\t\tNumber of RecHits "<<track->recHitsSize()<<endl;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	if ((*it)->isValid()){
	  std::cout <<"\t\t\tRecHit on det "<<(*it)->geographicalId().rawId()<<std::endl;
	  std::cout <<"\t\t\tRecHit in LP "<<(*it)->localPosition()<<std::endl;
	  std::cout <<"\t\t\tRecHit in GP "
	<<theG->idToDet((*it)->geographicalId())->surface().toGlobal((*it)->localPosition()) <<std::endl;
	  //try SimHit matching
	  float mindist = 999999;
	  float dist;
	  PSimHit closest;
	  matched.clear();	  
	  matched = associate.associateHit((**it));
	  if(!matched.empty()){
	    cout << "\t\t\tmatched  " << matched.size() << endl;
	    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    // 	      cout << "\t\t\tSimhit  ID  " << (*m).trackId() 
	    // 		   << "\t\t\tSimhit  LP  " << (*m).localPosition() 
	    // 		   << "\t\t\tSimhit  GP  " << theG->idToDet((*it)->geographicalId())->surface().toGlobal((*m).localPosition()) << endl;   
	    // 	      	      cout << "Track parameters " << theSimTracks[(*m).trackId()].momentum() << endl;
	    // 		      //do the majority of the simtrack here properly.  
	    // 	    }
	    //	    cout << "\t\t\tSimhit  ID  " << matched[0].trackId() << endl;
	    // << "\t\t\tSimhit  LP  " << matched[0].localPosition() 
	    //  << "\t\t\tSimhit  GP  " << theG->idToDet((*it)->geographicalId())->surface().toGlobal(matched[0].localPosition()) << endl;   
	    //cout << "Track parameters " << theSimTracks[matched[0].trackId()].momentum() << endl;
	    //now figure out which is the majority of the ids
	      dist = (*it)->localPosition().x() - (*m).localPosition().x();
	      if(dist<mindist){
		mindist = dist;
		closest = (*m);
	      }
	    }
	    SimTrackIds.push_back(closest.trackId());
	  }
	}else{
	  cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<endl;
	} 
      }

      int nmax = 0;
      int idmax = -1;
      for(size_t j=0; j<SimTrackIds.size(); j++){
	int n =0;
	n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	//	cout << " Tracks # of rechits = " << track->recHitsSize() << " found match = " << SimTrackIds.size() << endl;
	//cout << " rechit = " << i << " sim ID = " << SimTrackIds[i] << " Occurrence = " << n << endl; 
	if(n>nmax){
	  nmax = n;
	  idmax = SimTrackIds[i];
	}
      }
      float totsim = nmax;
      float tothits = track->recHitsSize();//include pixel as well..
      float fraction = totsim/tothits ;

      std::cout << " Track id # " << i << "# of rechits = " << track->recHitsSize() << " matched simtrack id= " << idmax 
	   << " fraction = " << fraction << std::endl;
      std::cout << " sim track mom = " <<  theSimTracks[idmax].momentum() << std::endl;
      i++;
      
    }
  }




