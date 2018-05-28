#ifndef RKCartesianDerivative_H
#define RKCartesianDerivative_H

#include "TrackPropagation/RungeKutta/src/CartesianState.h"

class dso_internal RKCartesianDerivative {
public:
 
  virtual ~RKCartesianDerivative() {}

  virtual CartesianState operator()( double step, const CartesianState& start) const = 0;

};

#endif
