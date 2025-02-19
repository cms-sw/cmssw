#ifndef VertexSmoothedChiSquaredEstimator_H
#define VertexSmoothedChiSquaredEstimator_H

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

/** \class VertexSmoothedChiSquaredEstimator
 *  Pure abstract base class for algorithms computing 
 *  a better estimation of vertex chi-squared after vertex fitting. 
 *  For the KalmanVertexFit both fitted and smoothed vertices are 
 *  needed, hence the 2 vertices passed as argument in the method... 
 */

template <unsigned int N>
class VertexSmoothedChiSquaredEstimator {

public:

  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef typename std::pair<bool, double> BDpair;

  VertexSmoothedChiSquaredEstimator() {}
  virtual ~VertexSmoothedChiSquaredEstimator() {}

  virtual BDpair estimate(const CachingVertex<N> &) const = 0;
  
  virtual VertexSmoothedChiSquaredEstimator<N> * clone() const = 0; 
  

};

#endif
