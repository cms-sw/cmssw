#ifndef SingleTrackVertexConstraint_H
#define SingleTrackVertexConstraint_H

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"

#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "boost/tuple/tuple.hpp"
/**
 * Class to re-estimate the parameters of the track at the vertex,
 *  with the vertex constraint or a BeamSpot, using the Kalman filter algorithms.
 * This will only change the parameters of the track at the vertex, but NOT
 * at other points along the track.
 */


class SingleTrackVertexConstraint {

public:

  typedef std::pair<reco::TransientTrack, float> TrackFloatPair;
  typedef boost::tuple<bool, reco::TransientTrack, float> BTFtuple;

  SingleTrackVertexConstraint(bool doTrackerBoundCheck = true) : 
	doTrackerBoundCheck_(doTrackerBoundCheck){}

  /**
   *  Constaint of a TransientTrack with a position and error.
   *  The track must NOT have been used in the vertex fit.
   */
  BTFtuple constrain(const reco::TransientTrack & track,
	const GlobalPoint& priorPos, const GlobalError& priorError) const;

  /**
   *  Constaint of a FreeTrajectoryState with a position and error.
   *  The track must NOT have been used in the vertex fit.
   */
  BTFtuple constrain(const FreeTrajectoryState & fts,
	const GlobalPoint& priorPos, const GlobalError& priorError) const;

  /**
   *  The method which does the constaint.
   *  The track must NOT have been used in the vertex fit.
   */
  BTFtuple constrain(const reco::TransientTrack & track,
	const VertexState priorVertex) const;

  /**
   *  Constaint of a TransientTrack with a BeamSpot.
   */
  BTFtuple constrain(const reco::TransientTrack & track,
	const reco::BeamSpot & spot ) const;


  /**
   *  Constaint of a FreeTrajectoryState with a BeamSpot.
   */
  BTFtuple constrain(const FreeTrajectoryState & fts,
	const reco::BeamSpot & spot) const;


private:

  KalmanVertexUpdator<5> vertexUpdator;
  KalmanVertexTrackUpdator<5> theVertexTrackUpdator;

  LinearizedTrackStateFactory theLTrackFactory;
  VertexTrackFactory<5> theVTrackFactory;
  TransientTrackFromFTSFactory ttFactory;
  bool doTrackerBoundCheck_;

};

#endif
