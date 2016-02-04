#ifndef RKOneCashKarpStep_H
#define RKOneCashKarpStep_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKSmallVector.h"
#include "RKDerivative.h"
#include "RKDistance.h"

#include <utility>

template <typename T, int N>
class dso_internal RKOneCashKarpStep // : RKStepWithPrecision 
{
public:

  typedef T                                   Scalar;
  typedef RKSmallVector<T,N>                  Vector;

  std::pair< Vector, T> 
  operator()( Scalar startPar, const Vector& startState,
	      const RKDerivative<T,N>& deriv,
	      const RKDistance<T,N>& dist, Scalar step);
  

};

#include "TrackPropagation/RungeKutta/src/RKOneCashKarpStep.icc"

#endif
