
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"

#include <cmath>
#include <iostream>

HcalSiPMShape::HcalSiPMShape() : CaloVShape(), nBins_(35*2+1), 
				 nt_(nBins_, 0.) {
  computeShape();
}

HcalSiPMShape::HcalSiPMShape(const HcalSiPMShape & other) :
  CaloVShape(other), nBins_(other.nBins_), nt_(other.nt_) {
}

double HcalSiPMShape::operator () (double time) const {
  int jtime = static_cast<int>(time*2 + 0.5);
  if (jtime>=0 && jtime<nBins_) 
    return nt_[jtime];
  return 0.;
}

void HcalSiPMShape::computeShape() {

  double norm = 0.;
  for (int j = 0; j < nBins_; ++j) {
    nt_[j] = analyticPulseShape(j/2.);
    norm += (nt_[j]>0) ? nt_[j] : 0.;
  }

  // std::cout << "SiPM pulse shape: ";
  for (int j = 0; j < nBins_; ++j) {
    nt_[j] /= norm;
    // std::cout << nt_[j] << ' ';
  }
  // std::cout << std::endl;
}

double HcalSiPMShape::gexp(double t, double A, double c, double t0, double s) {
  static double const root2(sqrt(2));
  return -A*0.5*exp(c*t+0.5*c*c*s*s-c*s)*(erf(-0.5*root2/s*(t-t0+c*s*s))-1);
}

double HcalSiPMShape::gexpIndefIntegral(double t, double A, double c, 
					double t0, double s) {
  static double const root2(sqrt(2));

  return (exp(-c*t0)*(exp(c*t0)*erf((root2*t0-root2*t)/(2*s))-exp(c*t+(c*c*s*2)/2)*erf((t0-t-c*s*s)/(root2*s))+exp(c*t+(c*c*s*s)/2))*A)/(2*c);
}

double HcalSiPMShape::gexpIntegral0Inf(double A, double c, double t0, 
				       double s) {
  static double const root2(sqrt(2));
  return (exp(-c*t0)*(exp((c*c*s*s)/2)*erf((root2*t0-root2*c*s*s)/(2*s))-exp(c*t0)*erf(t0/(root2*s))-exp(c*t0)-exp((c*c*s*s)/2))*A)/(2*c);
}

double HcalSiPMShape::analyticPulseShape(double t) const {
  double A1(0.08757), c1(-0.5257), t01(2.4013), s1(0.6721);
  double A2(0.007598), c2(-0.1501), t02(6.9412), s2(0.8710);
  return gexp(t,A1,c1,t01,s1) + gexp(t,A2,c2,t02,s2);
}
