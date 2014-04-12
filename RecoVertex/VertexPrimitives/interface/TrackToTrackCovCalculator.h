#ifndef TrackToTrackCovCalculator_H
#define TrackToTrackCovCalculator_H

#include <map>
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

/** \class TrackToTrackCovCalculator
 *  Abstract class for algorithms computing the covariance 
 *  matrices of each pair of tracks belonging to a CachingVertex.
 *  This covariance arises when refitting a track with the constraint 
 *  of the vertex.
 */

template <unsigned int N>
class TrackToTrackCovCalculator {

public:

  TrackToTrackCovCalculator() {}
  virtual ~TrackToTrackCovCalculator() {}

  virtual typename CachingVertex<N>::TrackToTrackMap operator() (const CachingVertex<N> &) const = 0;
  
  virtual TrackToTrackCovCalculator * clone() const = 0;

};

#endif
