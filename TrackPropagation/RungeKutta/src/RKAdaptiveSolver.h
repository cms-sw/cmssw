#ifndef RKAdaptiveSolver_H
#define RKAdaptiveSolver_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKSolver.h"

//#include "Utilities/UI/interface/SimpleConfigurable.h"

template <typename T, 
	  template <typename,int> class StepWithPrec, 
	  int N>
class dso_internal RKAdaptiveSolver : public RKSolver<T,N> {
public:

    typedef RKSolver<T,N>                       Base;
    typedef typename Base::Scalar               Scalar;
    typedef typename Base::Vector               Vector;

    virtual Vector operator()( Scalar startPar, const Vector& startState,
			       Scalar step, const RKDerivative<T,N>& deriv,
			       const RKDistance<T,N>& dist,
			       Scalar eps);

    std::pair< Vector, T> 
    stepWithAccuracy( Scalar startPar, const Vector& startState,
		      const RKDerivative<T,N>& deriv,
		      const RKDistance<T,N>& dist, Scalar step);

protected:

    bool verbose() const {
      static bool verb = false; //SimpleConfigurable<bool>(false,"RKAdaptiveSolver:verbose").value();
      return verb;
    }

};

#include "TrackPropagation/RungeKutta/src/RKAdaptiveSolver.icc"

#endif
