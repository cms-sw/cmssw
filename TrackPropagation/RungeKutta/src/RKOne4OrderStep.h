#ifndef RKOne4OrderStep_H
#define RKOne4OrderStep_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKDistance.h"
#include "RK4OneStepTempl.h"

#include <utility>

template <typename T, int N>
class dso_internal RKOne4OrderStep {
public:

  typedef T                                   Scalar;
  typedef RKSmallVector<T,N>                  Vector;

  std::pair< Vector, T> 
  operator()( Scalar startPar, const Vector& startState,
	      const RKDerivative<T,N>& deriv,
	      const RKDistance<T,N>& dist, Scalar step) {
    const Scalar huge = 1.e5;  // ad hoc protection against infinities, must be done better!
    const Scalar hugediff = 100.;

    RK4OneStepTempl<T,N> solver;
    Vector one(       solver(startPar, startState, deriv, step));
    if (std::abs(one[0])>huge || std::abs(one(1))>huge) return std::pair<Vector, Scalar>(one,hugediff);

    Vector firstHalf( solver(startPar, startState, deriv, step/2));
    Vector secondHalf(solver(startPar+step/2, firstHalf, deriv, step/2));
    Scalar diff = dist(one, secondHalf, startPar+step);
    return std::pair<Vector, Scalar>(secondHalf,diff);
  }
};
#endif
