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

  typedef std::pair<reco::TransientTrack, float> TrackFloatPair;
  /** 
   *  The method which does the constaint, from a TransientTrack.
   *  The track must NOT have been used in the vertex fit.
   */
  TrackFloatPair constrain(const reco::TransientTrack & track, 
	const GlobalPoint& priorPos, const GlobalError& priorError) const;

  /** 
   *  The method which does the constaint, from a FreeTrajectoryState
   *  The track must NOT have been used in the vertex fit.
   */
  TrackFloatPair constrain(const FreeTrajectoryState & fts,
	const GlobalPoint& priorPos, const GlobalError& priorError) const;


  TrackFloatPair constrain(const reco::TransientTrack & track,
	const VertexState priorVertex) const;

  TrackFloatPair constrain(
	const reco::TransientTrack & track, const reco::BeamSpot & spot ) const;


  TrackFloatPair constrain(
	const FreeTrajectoryState & fts, const reco::BeamSpot & spot) const;



  SingleTrackVertexConstraint * clone() const {
    return new SingleTrackVertexConstraint(* this);
  }

private:

  KalmanVertexUpdator<5> vertexUpdator;
  KalmanVertexTrackUpdator<5> theVertexTrackUpdator;

  LinearizedTrackStateFactory theLTrackFactory;
  VertexTrackFactory<5> theVTrackFactory;
  TransientTrackFromFTSFactory ttFactory;

};

#endif
