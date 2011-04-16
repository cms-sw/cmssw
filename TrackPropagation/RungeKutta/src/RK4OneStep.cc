#include "RK4OneStep.h"
#include "RKCartesianDerivative.h"

#include <iostream>

CartesianState
RK4OneStep::operator()( const CartesianState& start, const RKCartesianDerivative& deriv,
			double step) const
{
  double s0 = 0; // derivatives do't depend on absolute value of the integration variable
  CartesianState k1 = step * deriv( s0, start);
  CartesianState k2 = step * deriv( s0+step/2, start+k1/2);
  CartesianState k3 = step * deriv( s0+step/2, start+k2/2);
  CartesianState k4 = step * deriv( s0+step, start+k3);
/*
  std::cout << "k1 = " << k1.position() << k1.momentum() << std::endl;
  std::cout << "k2 = " << k2.position() << k2.momentum() << std::endl;
  std::cout << "k3 = " << k3.position() << k3.momentum() << std::endl;
  std::cout << "k4 = " << k4.position() << k4.momentum() << std::endl;
*/
  CartesianState result = start + k1/6 + k2/3 + k3/3 + k4/6;
  return result;
}
