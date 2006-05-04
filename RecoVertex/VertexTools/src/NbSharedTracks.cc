#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// this class' header
#include "RecoVertex/VertexTools/interface/NbSharedTracks.h"


int reco::NbSharedTracks::sharedTracks(const reco::Vertex &v1,
				       const reco::Vertex &v2) const {

  int nSharedTracks = 0;

  // for first vertex
  track_iterator v1TrackIter;
  track_iterator v1TrackBegin = v1.tracks_begin();
  track_iterator v1TrackEnd   = v1.tracks_end();
  

  // for second vertex
  track_iterator v2TrackIter;
  track_iterator v2TrackBegin = v2.tracks_begin();
  track_iterator v2TrackEnd   = v2.tracks_end();
  
  for (v1TrackIter = v1TrackBegin; v1TrackIter != v1TrackEnd; v1TrackIter++) {
    for (v2TrackIter = v2TrackBegin; v2TrackIter != v2TrackEnd; v2TrackIter++) {
      if ( (*v1TrackIter) == (*v2TrackIter) ) {
	nSharedTracks++;
      } // if 
    } // for v2TrackIter
  } //for v1TrackIter


  return nSharedTracks;
} // int sharedTracks
