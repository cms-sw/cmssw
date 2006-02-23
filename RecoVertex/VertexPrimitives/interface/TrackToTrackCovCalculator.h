#ifndef TrackToTrackCovCalculator_H
#define TrackToTrackCovCalculator_H

#include <map>
#include "RecoVertex/VertexPrimitives/interface/RefCountedVertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TrackMap.h"
#include "RecoVertex/VertexPrimitives/interface/TrackToTrackMap.h"

/** \class TrackToTrackCovCalculator
 *  Abstract class for algorithms computing the covariance 
 *  matrices of each pair of tracks belonging to a CachingVertex.
 *  This covariance arises when refitting a track with the constraint 
 *  of the vertex.
 */

class CachingVertex;

class TrackToTrackCovCalculator {

public:

  TrackToTrackCovCalculator() {}
  virtual ~TrackToTrackCovCalculator() {}

  virtual TrackToTrackMap operator() (const CachingVertex &) const = 0;
  
  virtual TrackToTrackCovCalculator * clone() const = 0;

};

#endif
