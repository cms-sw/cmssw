#ifndef SingleTrackVertexConstraint_H
#define SingleTrackVertexConstraint_H

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"

#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"

/**
 * Class to re-estimate the parameters of the track at the vertex, 
 *  with the vertex constraint, using the Kalman filter algorithms.
 * This will only change the parameters of the track at the vertex, but NOT 
 * at other points along the track.
 */


class SingleTrackVertexConstraint {

public:

  /** 
   *  The method which does the constaint, from a TransientTrack
   */
  reco::TransientTrack constrain(const reco::TransientTrack & track, 
	const GlobalPoint& priorPos, const GlobalError& priorError) const;

  /** 
   *  The method which does the constaint, from a FreeTrajectoryState
   */
  reco::TransientTrack constrain(const FreeTrajectoryState & fts, 
	const GlobalPoint& priorPos, const GlobalError& priorError) const;


  SingleTrackVertexConstraint * clone() const {
    return new SingleTrackVertexConstraint(* this);
  }

private:

  KalmanVertexUpdator vertexUpdator;
  KalmanVertexTrackUpdator theVertexTrackUpdator;

  LinearizedTrackStateFactory theLTrackFactory;
  VertexTrackFactory theVTrackFactory;
  TransientTrackFromFTSFactory ttFactory;

};

#endif
