#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include <cmath>

ESShape::ESShape(int Gain):
  theGain(Gain)
{
  setTpeak(20.0);

  if (theGain==0) {
    A = 6.;
    Qcf = 4./350.;
    omegac = 2./25.;
    norm = 0.11136;
  }
  else if (theGain==1) {
    // preliminary numbers, need to be approved by preshower group
    A = 5.99994;
    Qcf = 0.0114319;
    omegac = 0.0736172;
    norm = 0.111492;
  }
  else if (theGain==2) {
    // preliminary numbers, need to be approved by preshower group
    A = 5.99912; 
    Qcf = 0.0114275;
    omegac = 0.086403;
    norm = 0.152841;
  }
}

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


