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
 *  with the vertex constraint or a BeamSpot, using the Kalman filter algorithms.
 * This will only change the parameters of the track at the vertex, but NOT
 * at other points along the track.
 * The constrain methods will return a bool telling whether the calculation was successful.
 * Only in this case can the result mothod be called to get the result.
 */


class SingleTrackVertexConstraint {

public:

  typedef std::pair<reco::TransientTrack, float> TrackFloatPair;

  SingleTrackVertexConstraint() : validity_(false) {}
  /**
   *  Constaint of a TransientTrack with a position and error.
   *  The track must NOT have been used in the vertex fit.
   *  The methods returns the status of the constraint.
   */
  bool constrain(const reco::TransientTrack & track,
	const GlobalPoint& priorPos, const GlobalError& priorError);

  /**
   *  Constaint of a FreeTrajectoryState with a position and error.
   *  The track must NOT have been used in the vertex fit.
   *  The methods returns the status of the constraint.
   */
  bool constrain(const FreeTrajectoryState & fts,
	const GlobalPoint& priorPos, const GlobalError& priorError);

  /**
   *  The method which does the constaint.
   *  The track must NOT have been used in the vertex fit.
   *  The methods returns the status of the constraint.
   */
  bool constrain(const reco::TransientTrack & track,
	const VertexState priorVertex);

  /**
   *  Constaint of a TransientTrack with a BeamSpot.
   *  The methods returns the status of the constraint.
   */
  bool constrain(const reco::TransientTrack & track,
	const reco::BeamSpot & spot );


  /**
   *  Constaint of a FreeTrajectoryState with a BeamSpot.
   *  The methods returns the status of the constraint.
   */
  bool constrain(const FreeTrajectoryState & fts,
	const reco::BeamSpot & spot);

  /**
   * Will return whether the previous constaint was successful
   */
  bool isValid() {return validity_;}

  /**
   * The result of the previous constaint, if that was successful.
   * If it was not, an exception will be thrown.
   */
  TrackFloatPair result() const;

  SingleTrackVertexConstraint * clone() const {
    return new SingleTrackVertexConstraint(* this);
  }

private:

  KalmanVertexUpdator<5> vertexUpdator;
  KalmanVertexTrackUpdator<5> theVertexTrackUpdator;

  LinearizedTrackStateFactory theLTrackFactory;
  VertexTrackFactory<5> theVTrackFactory;
  TransientTrackFromFTSFactory ttFactory;
  mutable TrackFloatPair result_;
  mutable bool validity_;

};

#endif
