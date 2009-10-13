#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include <cmath>

ESShape::ESShape(int Gain):
  theGain_(Gain)
{
  if (theGain_==0) {
    A_ = 6.;
    Qcf_ = 4./350.;
    omegac_ = 2./25.;
    M_ = 2.;
    norm_ = 0.11136*M_;
  }
  else if (theGain_==1) {
    A_ = 17.73; 
    Qcf_ = 6.044;
    omegac_ = 0.1;
    M_ = 3.324;
    norm_ = 1.374*2438.76;
  }
  else if (theGain_==2) {
    A_ = 18.12;
    Qcf_ = 7.58;
    omegac_ = 0.08757;
    M_ = 3.192;
    norm_ = 1.24*2184.13;
  }
}

double ESShape::operator () (double time_) const
{   
  if (time_>0.00001) {
    double xf = A_*omegac_*time_;
    return (Qcf_/norm_)*pow(xf,M_-1.)*exp(-omegac_*time_);
  } 
  else {
    return 0.0;
  }
}

double
ESShape::timeToRise() const
{
   return 0.0 ;
}

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

