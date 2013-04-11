#ifndef TRANSIENTTRACKINGROUPOFTRACKS_H  
#define TRANSIENTTRACKINGROUPOFTRACKS_H 

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <vector>

// class to find out if a TransientTrack is in a group of TransientTracks 

class TransientTrackInGroupOfTracks {

public:

  TransientTrackInGroupOfTracks() {}

  ~TransientTrackInGroupOfTracks() {}
  
  static bool isInGroup ( const reco::TransientTrack & aTrack , 
    const std::vector<reco::TransientTrack> &groupOfTracks ) {
    bool trackFound = false;
    for( std::vector<reco::TransientTrack>::const_iterator itT = groupOfTracks.begin() ; 
      itT != groupOfTracks.end() ; itT++ ) {
      //if ( aTrack.sameAddress(*itT) ) trackFound = true ;
      //std::cout<<"a,b: "<<(aTrack.impactPointState().signedInverseMomentum())<<","<<((*itT).impactPointState().signedInverseMomentum())<<std::endl;
      if (aTrack == *itT) { 
	trackFound = true;
	//        std::cout<<"found!\n";
      }
      //      else std::cout<<"not found!\n";
    }
    //    if(!trackFound) std::cout<<"NOT FOUND!\n";
    return trackFound ;
  }
  
};
#endif
