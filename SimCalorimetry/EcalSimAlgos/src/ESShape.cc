#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include <cmath>

double ESShape::operator () (double time_) const
{   
  if (time_>0.00001) {
    double xf = A*omegac*time_;
    return (Qcf/(2.*norm))*xf*xf*exp(-omegac*time_);
  } 
  else {
    return 0.0;
  }
}

double ESShape::derivative (double time_) const
{
  if (time_>0.00001) {
    double xf = A*omegac*time_;
    return (Qcf/(2.*norm))*xf*xf*exp(-omegac*time_);
  } 
  else {
    return 0.0;
  }
}


