#ifndef KalmanChiSquare_H
#define KalmanChiSquare_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"

class KalmanChiSquare {
  /**
   *  A chi-2 criterion that relies on the Kalman formalism,
   *  exploiting data stored in LinearizedTrackState.
   *  It needs only a VertexState, not a full 
   */

public:
  typedef ReferenceCountingPointer<VertexTrack<5> > RefCountedVertexTrack;
  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

  float estimate ( const GlobalPoint &, 
                   RefCountedLinearizedTrackState ) const;
};

#endif
