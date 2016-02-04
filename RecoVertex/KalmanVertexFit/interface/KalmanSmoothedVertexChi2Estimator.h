#ifndef KalmanSmoothedVertexChi2Estimator_H
#define KalmanSmoothedVertexChi2Estimator_H

#include "RecoVertex/VertexPrimitives/interface/VertexSmoothedChiSquaredEstimator.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"

  /**
   * Class to calculate the smoothed chi**2 of the vertex using the Kalman 
   *  filter algorithms after the vertex has been fit and the tracks refit.
   */

template <unsigned int N>
class KalmanSmoothedVertexChi2Estimator:public VertexSmoothedChiSquaredEstimator<N> {

public:

  typedef typename std::pair<bool, double> BDpair;

  virtual ~KalmanSmoothedVertexChi2Estimator() {}

  /**
   *  Methode which calculates the smoothed vertex chi**2.
   *  \param vertex is the final estimate of the vertex, with the refited tracks
   *  \return the smoothed vertex chi**2
   */
  BDpair estimate(const CachingVertex<N> & vertex) const;
   
  KalmanSmoothedVertexChi2Estimator * clone() const 
  {
   return new KalmanSmoothedVertexChi2Estimator(* this);
  }
   
private:

  KVFHelper<N> helper;
};

#endif
