#ifndef RK4PreciseStep_H
#define RK4PreciseStep_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "CartesianState.h"
#include <utility>

class RKCartesianDerivative;

class dso_internal RK4PreciseStep {
public:

  CartesianState
  operator()( const CartesianState& start, const RKCartesianDerivative& deriv,
	      double step, double eps) const;

  double distance( const CartesianState& a, const CartesianState& b) const;

  std::pair<CartesianState, double>
  stepWithAccuracy( const CartesianState& start, const RKCartesianDerivative& deriv, double step) const;

private:

  bool verbose() const;

};

#endif
