#ifndef VertexSmoothedChiSquaredEstimator_H
#define VertexSmoothedChiSquaredEstimator_H

/** \class VertexSmoothedChiSquaredEstimator
 *  Pure abstract base class for algorithms computing 
 *  a better estimation of vertex chi-squared after vertex fitting. 
 *  For the KalmanVertexFit both fitted and smoothed vertices are 
 *  needed, hence the 2 vertices passed as argument in the method... 
 */

class CachingVertex;

class VertexSmoothedChiSquaredEstimator {

public:

  VertexSmoothedChiSquaredEstimator() {}
  virtual ~VertexSmoothedChiSquaredEstimator() {}

  virtual float estimate(const CachingVertex &) const = 0;
  
  virtual VertexSmoothedChiSquaredEstimator * clone() const = 0; 
  

};

#endif
