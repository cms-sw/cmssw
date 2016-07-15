
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"
#include "TMath.h"
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

inline double onePulse(double t, double A, double sigma, double theta, double m) {
  return (t<theta) ? 0 : A*TMath::LogNormal(t,sigma,theta,m);
}

double HcalSiPMShape::analyticPulseShape(double t) const {
  // taken from fit to laser measurement taken by Iouri M. in Spring 2016.
  double A1(5.204/6.94419), sigma1_shape(0.5387), theta1_loc(-0.3976), m1_scale(4.428);
  double A2(1.855/6.94419), sigma2_shape(0.8132), theta2_loc(7.025),   m2_scale(12.29);
  return
    onePulse(t,A1,sigma1_shape,theta1_loc,m1_scale) +
    onePulse(t,A2,sigma2_shape,theta2_loc,m2_scale);
}
