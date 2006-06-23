#ifndef ChiSquareForWeightComputation_H
#define ChiSquareForWeightComputation_H

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
class reco::TransientTrack;

class ChiSquareForWeightComputation {
  /**
   *  Simple class used by the AdaptiveVertexFitter and MultiVertexFitter to
   *  determine the assignment probabilities.  Note that the estimate is _not_
   *  exactly what one would call a "chi-square".  (The vertex error is not
   *  taken into account)
   */

public:
  float estimate ( const GlobalPoint & vertex, const reco::TransientTrack & ) const;
  float estimate ( const GlobalPoint & vertex,
                   const RefCountedLinearizedTrackState & track ) const;
};

#endif
