#include "RK4PreciseStep.h"
#include "RK4OneStep.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
#include <iostream>

CartesianState
RK4PreciseStep::operator()( const CartesianState& start, const RKCartesianDerivative& deriv,
			    double step, double eps) const
{
    const double Safety = 0.9;
    double remainigStep = step;
    double stepSize = step;
    CartesianState currentStart = start;
    int nsteps = 0;
    std::pair<CartesianState, double> tryStep;

    do {
	tryStep = stepWithAccuracy( currentStart, deriv, stepSize);
	nsteps++;
	if (tryStep.second <eps) {
	    if (remainigStep - stepSize < eps/2) {
		if (verbose()) std::cout << "Accuracy reached, and full step taken in " 
				    << nsteps << " steps" << std::endl;
		return tryStep.first; // we are there
	    }
	    else {
		remainigStep -= stepSize;
                // increase step size
		double factor =  std::min( Safety * pow( fabs(eps/tryStep.second),0.2), 4.);
		stepSize = std::min( stepSize*factor, remainigStep);
		currentStart = tryStep.first;
		if (verbose()) std::cout << "Accuracy reached, but " << remainigStep 
		     << " remain after " << nsteps << " steps. Step size increased by " 
		     << factor << " to " << stepSize << std::endl;
	    }
	}
	else {
	    // decrease step size
	    double factor =  std::max( Safety * pow( fabs(eps/tryStep.second),0.25), 0.1);
	    stepSize *= factor;
	    if (verbose()) std::cout << "Accuracy not yet reached: delta = " << tryStep.second
		 << ", step reduced by " << factor << " to " << stepSize 
		 << ", (R,z)= " << currentStart.position().perp() 
		 << ", " << currentStart.position().z() << std::endl;
	}
    } while (remainigStep > eps/2);

    return tryStep.first;
}

std::pair<CartesianState, double>
RK4PreciseStep::stepWithAccuracy( const CartesianState& start, const RKCartesianDerivative& deriv,
				  double step) const
{
    RK4OneStep solver;
    CartesianState one(solver(start, deriv, step));
    CartesianState firstHalf(solver(start, deriv, step/2));
    CartesianState secondHalf(solver(firstHalf, deriv, step/2));
    double diff = distance(one, secondHalf);
    return std::pair<CartesianState, double>(secondHalf,diff);
}

double RK4PreciseStep::distance( const CartesianState& a, const CartesianState& b) const
{
    return (a.position() - b.position()).mag() + (a.momentum() - b.momentum()).mag() / b.momentum().mag();
}

bool RK4PreciseStep::verbose() const
{
  // static bool verb = SimpleConfigurable<bool>(false,"RK4PreciseStep:verbose").value();

  static bool verb = true;
  return verb;
}
