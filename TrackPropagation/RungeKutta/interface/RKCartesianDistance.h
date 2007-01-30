#ifndef RKCartesianDistance_H
#define RKCartesianDistance_H

#include "TrackPropagation/RungeKutta/interface/RKDistance.h"
#include "TrackPropagation/RungeKutta/interface/RKSmallVector.h"
#include "TrackPropagation/RungeKutta/interface/CartesianStateAdaptor.h"

#include <cmath>

/// Estimator of the distance between two state vectors, e.g. for convergence test

class RKCartesianDistance : public RKDistance<double,6> {
public:
 
  typedef double                                 Scalar;
  typedef RKSmallVector<double,6>                Vector;

  virtual ~RKCartesianDistance() {}

  virtual Scalar operator()( const Vector& rka, const Vector& rkb, const Scalar& s) const {
    CartesianStateAdaptor a(rka), b(rkb);

    return (a.position()-b.position()).mag() + 
      (a.momentum() - b.momentum()).mag() / b.momentum().mag();
  }
 
};

#endif
