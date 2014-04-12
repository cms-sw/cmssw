#ifndef RK4OneStepTempl_H
#define RK4OneStepTempl_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKSmallVector.h"
#include "RKDerivative.h"

template <typename T, int N>
class dso_internal RK4OneStepTempl {
 public:

    typedef T                                   Scalar;
    typedef RKSmallVector<T,N>                  Vector;

  
    Vector operator()( Scalar startPar, const Vector& startState,
		       const RKDerivative<T,N>& deriv, Scalar step) const {

 	// cout << "RK4OneStepTempl: starting from " << startPar << startState << endl;

	Vector k1 = step * deriv( startPar, startState);
	Vector k2 = step * deriv( startPar+step/2, startState+k1/2);
	Vector k3 = step * deriv( startPar+step/2, startState+k2/2);
	Vector k4 = step * deriv( startPar+step, startState+k3);

	Vector result = startState + k1/6 + k2/3 + k3/3 + k4/6;

 	// cout << "RK4OneStepTempl: result for step " << step << " is " << result << endl;

	return result;
    }
};

#endif
