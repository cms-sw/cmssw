#ifndef RK4OneStep_H
#define RK4OneStep_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "CartesianState.h"

class RKCartesianDerivative;

class dso_internal RK4OneStep {
public:

  CartesianState
  operator()( const CartesianState& start, const RKCartesianDerivative& deriv,
	      double step) const;


  //  DeltaState errorEstimate();

};

#endif
