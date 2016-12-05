
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "TMath.h"
#include <iostream>

HcalSiPMShape::HcalSiPMShape(unsigned int signalShape) : CaloVShape(), nBins_(250*2), 
				 nt_(nBins_, 0.) {
  computeShape(signalShape);
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

void HcalSiPMShape::computeShape(unsigned int signalShape) {

  double norm = 0.;
  for (int j = 0; j < nBins_; ++j) {
    nt_[j] = analyticPulseShape(j/2.,signalShape);
    norm += (nt_[j]>0) ? nt_[j] : 0.;
  }

  // std::cout << "SiPM pulse shape: ";
  for (int j = 0; j < nBins_; ++j) {
    nt_[j] /= norm;
    // std::cout << nt_[j] << ' ';
  }
  // std::cout << std::endl;
}

inline double gexp(double t, double A, double c, double t0, double s) {
  static double const root2(sqrt(2));
  return -A*0.5*exp(c*t+0.5*c*c*s*s-c*s)*(erf(-0.5*root2/s*(t-t0+c*s*s))-1);
}

inline double onePulse(double t, double A, double sigma, double theta, double m) {
  return (t<theta) ? 0 : A*TMath::LogNormal(t,sigma,theta,m);
}

double HcalSiPMShape::analyticPulseShape(double t, unsigned int signalShape) const {
  if(signalShape==HcalShapes::ZECOTEK || signalShape==HcalShapes::HAMAMATSU){
    // HO SiPM pulse shape fit from Jake Anderson ca. 2013
    double A1(0.08757), c1(-0.5257), t01(2.4013), s1(0.6721);
    double A2(0.007598), c2(-0.1501), t02(6.9412), s2(0.8710);
    return gexp(t,A1,c1,t01,s1) + gexp(t,A2,c2,t02,s2);
  }
  else if(signalShape==HcalShapes::HE2017){
    // taken from fit to laser measurement taken by Iouri M. in Spring 2016.
    double A1(5.204/6.94419), sigma1_shape(0.5387), theta1_loc(-0.3976), m1_scale(4.428);
    double A2(1.855/6.94419), sigma2_shape(0.8132), theta2_loc(7.025),   m2_scale(12.29);
    return
      onePulse(t,A1,sigma1_shape,theta1_loc,m1_scale) +
      onePulse(t,A2,sigma2_shape,theta2_loc,m2_scale);
  }
  else return 0.;
}
