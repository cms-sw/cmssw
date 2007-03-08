#ifndef KalmanChiSquare_H
#define KalmanChiSquare_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"

class KalmanChiSquare {
  /**
   *  A chi-2 criterion that relies on the Kalman formalism,
   *  exploiting data stored in LinearizedTrackState.
   *  It needs only a VertexState, not a full 
   */

public:
  float estimate ( const GlobalPoint &, 
                   RefCountedLinearizedTrackState ) const;
};

#endif
