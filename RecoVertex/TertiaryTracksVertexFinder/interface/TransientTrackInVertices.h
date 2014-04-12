#ifndef TRANSIENTTRACKINVERTICES_H  
#define TRANSIENTTRACKINVERTICES_H 

#include "RecoVertex/TertiaryTracksVertexFinder/interface/TransientTrackInGroupOfTracks.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <vector>

// class to find out if a TransientTrack is in a (group of) vertices 

class TransientTrackInVertices {

public:

  TransientTrackInVertices ()  {}
  ~TransientTrackInVertices () {}
  
  static bool isInVertex (const reco::TransientTrack & aTrack , 
    const TransientVertex & aVertex ) {
    return TransientTrackInGroupOfTracks::isInGroup ( aTrack , aVertex.originalTracks() );
  }

  static bool isInVertex ( const reco::TransientTrack & aTrack , 
    const std::vector<TransientVertex> &vertices ) {
    bool isInVertices = false ;
    for(std::vector<TransientVertex>::const_iterator itV = vertices.begin(); 
      itV != vertices.end() ; itV++ ) {
      if ( isInVertex ( aTrack , *itV ) ) isInVertices = true ;
    }
    return isInVertices ;
  }
  
};

#endif

