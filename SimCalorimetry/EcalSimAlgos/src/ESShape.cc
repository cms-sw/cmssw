#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include <cmath>

ESShape::ESShape() {}

double ESShape::operator()(double time_) const {
  if (time_ > 0.00001) {
    double wc = 0.07291;
    double n = 1.798;  // n-1 (in fact)
    double v1 = pow(wc / n * time_, n);
    double v2 = exp(n - wc * time_);
    double v = v1 * v2;

    return v;
  } else {
    return 0.0;
  }
}

double ESShape::timeToRise() const { return 0.0; }

/*
double ESShape::derivative (double time_) const
{
  if (time_>0.00001) {
    double xf = A_*omegac_*time_;
    return (Qcf_/norm_)*pow(xf,M_-1.)*exp(-omegac_*time_);
  } 
  else {
    return 0.0;
  }
}
*/
