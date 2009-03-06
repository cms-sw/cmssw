#include "RecoVertex/NuclearInteractionProducer/interface/NuclearLikelihood.h"

void NuclearLikelihood::calculate( const reco::Vertex& vtx ) {
      likelihood_ = 0.0;
      if( vtx.isValid() ) {
           if(vtx.tracksSize() > 1) {
               likelihood_ = 0.5;
                 if( secondaryTrackMaxHits( vtx ) > 3 ) likelihood_ =1.0;
           }
      }
}

int NuclearLikelihood::secondaryTrackMaxHits( const reco::Vertex& vtx ) {
      int maxHits = 0;
      if( vtx.tracksSize() < 2 )  return 0;
      for( reco::Vertex::trackRef_iterator it = vtx.tracks_begin()+1; it != vtx.tracks_end(); ++it){
            int nhits = (*it)->numberOfValidHits();
            if( nhits > maxHits ) maxHits = nhits;
      }
      return maxHits;
}
