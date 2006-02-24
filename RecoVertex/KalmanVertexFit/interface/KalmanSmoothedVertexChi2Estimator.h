#ifndef KalmanSmoothedVertexChi2Estimator_H
#define KalmanSmoothedVertexChi2Estimator_H

#include "RecoVertex/VertexPrimitives/interface/VertexSmoothedChiSquaredEstimator.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

  /**
   * Class to calculate the smoothed chi**2 of the vertex using the Kalman 
   *  filter algorithms after the vertex has been fit and the tracks refit.
   */

class KalmanSmoothedVertexChi2Estimator:public VertexSmoothedChiSquaredEstimator {

public:

  virtual ~KalmanSmoothedVertexChi2Estimator() {}

  /**
   *  Methode which calculates the smoothed vertex chi**2.
   *  \param vertex is the final estimate of the vertex, with the refited tracks
   *  \return the smoothed vertex chi**2
   */
  float estimate(const CachingVertex & vertex) const;
   
  KalmanSmoothedVertexChi2Estimator * clone() const 
  {
   return new KalmanSmoothedVertexChi2Estimator(* this);
  }
   

  /**
   *  Methode which calculates the chi**2 between the prior and the fitted vertex.
   *  This method will not take into account multiple states, so in case one of
   *  the VertexStates is a multi-state vertex, only the mean will be used.
   *  \param priorVertex The prior vertex state
   *  \param fittedVertex The fitted vertex state
   */
  double priorVertexChi2(const VertexState priorVertex, 
	const VertexState fittedVertex) const;

  /**
   *  Methode which calculates the chi**2 between the prior and the fitted 
   *   track parameters.
   *  \param linTrack	The track as linearized
   *  \param refittedTrackState The refitted track
   */
  float trackParameterChi2(const RefCountedLinearizedTrackState linTrack, 
	const RefCountedRefittedTrackState refittedTrackState) const;
   
};

#endif
