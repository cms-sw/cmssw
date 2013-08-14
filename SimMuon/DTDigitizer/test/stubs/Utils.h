#ifndef Utils_H
#define Utils_H

/*
 * Parametrization function and related utilities, for testing purposes. 
 *
 * Coordinates are 
 * x in cm; theta in degrees, B in T, t in ns
 * Bwire = By in ORCA r.f.
 * Bnorm = Bz in ORCA r.f.
 *
 *  $Date: 2006/03/09 16:56:34 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - INFN Torino
 */


// Drift space-time parametrization after J. Puerta, P. Garcia-Abia
void* driftTime(double x, double theta, double Bwire, double Bnorm, short interpolate);

// Inverse time-space parametrization after J. Puerta, P. Garcia-Abia
void* trackDistance(double x, double theta, double Bwire, double Bnorm, short interpolate);

// Drift time parametrization after T. Rovelli, A. Gresele
float oldParametrization(float x, float theta, float Bwire, float Bnorm);

double smearedTime(double x, double theta, double Bwire, double Bnorm, short interpolate);

// Smearing of an asymmetric distribution which results from combining two 
// gaussians with the same peak and different sigmas.
double asymGausSample(double mean, double sigma1 ,double sigma2);


#include "TString.h"
// Convert a double to a TString with a given precision.
TString dToTString(double f, int precision=6);

#endif

