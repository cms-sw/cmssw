#ifndef RKCartesianDistance_H
#define RKCartesianDistance_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "RKDistance.h"
#include "RKSmallVector.h"
#include "CartesianStateAdaptor.h"

#include <cmath>

/// Estimator of the distance between two state vectors, e.g. for convergence test

class dso_internal RKCartesianDistance final : public RKDistance<double,6> {
public:
 
  typedef double                                 Scalar;
  typedef RKSmallVector<double,6>                Vector;

  ~RKCartesianDistance() override {}

  Scalar operator()( const Vector& rka, const Vector& rkb, const Scalar& s) const override {
    CartesianStateAdaptor a(rka), b(rkb);

    return (a.position()-b.position()).mag() + 
      (a.momentum() - b.momentum()).mag() / b.momentum().mag();
  }
 
};

#endif
