#ifndef RK4OneStep_H
#define RK4OneStep_H

#include "CartesianState.h"

class RKCartesianDerivative;

class RK4OneStep {
public:

  CartesianState
  operator()( const CartesianState& start, const RKCartesianDerivative& deriv,
	      double step) const;


  //  DeltaState errorEstimate();

};

#endif
