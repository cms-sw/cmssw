/*
 *  $Date: 2009/05/25 07:16:08 $
 *  $Revision: 1.3 $
 *  \author G. Bevilacqua, N. Amapane - INFN Torino
 */

#include "Utils.h"

#include "SimMuon/DTDigitizer/src/DTDriftTimeParametrization.h"
#include "DTTime2DriftParametrization.h"

#include <CLHEP/Random/RandGaussQ.h>
#include <CLHEP/Random/RandFlat.h>
#include <cmath>


void* driftTime(double x, double theta, double bWire, double bNorm,
		short interpolate){
  static DTDriftTimeParametrization::drift_time DT;

  // Convert coordinates...
  x*=10.; // cm->mm
  double by_par = bNorm;
  double bz_par = -bWire;
  
  DTDriftTimeParametrization param;
  
  //  unsigned short flag = 
  param.MB_DT_drift_time(x,theta,by_par,bz_par,0,&DT, interpolate);
  return &DT;

}

void* trackDistance(double t, double theta, double bWire, double bNorm,
		    short interpolate){
  static DTTime2DriftParametrization::drift_distance DX;

  // Convert coordinates...
  double by_par = bNorm;
  double bz_par = -bWire;

  DTTime2DriftParametrization param;
  
  // unsigned short flag = 
  param.MB_DT_drift_distance(t,theta,by_par,bz_par,&DX, interpolate);
  return &DX;

}

double smearedTime(double x, double theta, double Bwire, double Bnorm,
		   short interpolate){
  DTDriftTimeParametrization::drift_time * DT;
  DT = (DTDriftTimeParametrization::drift_time *) 
    driftTime(x, theta, Bwire, Bnorm, interpolate);
  return asymGausSample(DT->t_drift, DT->t_width_m, DT->t_width_p);
}

double asymGausSample(double mean, double sigma1, double sigma2) {

  double f = sigma1/(sigma1+sigma2);
  double t;

  if (CLHEP::RandFlat::shoot() <= f) {
    t = CLHEP::RandGaussQ::shoot(mean,sigma1);
    return mean - fabs(t - mean);
  } else {
    t = CLHEP::RandGaussQ::shoot(mean,sigma2);
    return mean + fabs(t - mean);    
  }
}


#include <string>
#include <sstream>
#include <iomanip>

#include <iostream>

TString dToTString(double f, int precision){
 std::stringstream StrStm;
 std::string Str;

 StrStm << std::setprecision(precision); // (Actually the default is 6);
 StrStm << f;
 Str = StrStm.str();
 TString tStr(Str.c_str());
 return tStr; 
}
