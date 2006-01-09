#include "SimMuon/CSCDigitizer/src/CSCGattiFunction.h"
#include "Geometry/CSCSimAlgo/interface/CSCChamberSpecs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>
#include <iostream>
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

// Author: Rick Wilkinson

void CSCGattiFunction::initChamberSpecs(const CSCChamberSpecs & chamberSpecs) {
  if(&chamberSpecs != previousSpecs) {
    LogDebug("CSCGattiFunction") << "CSCGattiFunction::initChamberSpecs setting new values.";
    h = chamberSpecs.anodeCathodeSpacing();
    double s = chamberSpecs.wireSpacing();
    double ra = chamberSpecs.wireRadius();
    static const double parm[5] = {.1989337e-02, -.6901542e-04,  .8665786, 
				   154.6177, -.6801630e-03 };
    k3 = (parm[0]*s/h + parm[1]) 
           * (parm[2]*s/ra + parm[3] + parm[4]*s*s/ra/ra);
    sqrtk3 = sqrt(k3);
    norm = 0.5 / std::atan( sqrtk3 );
    k2 = M_PI_2 * (1. - sqrtk3/2.);
    k1 = 0.25 * k2 * sqrtk3 / std::atan(sqrtk3);
    previousSpecs = &chamberSpecs;
  }

  LogDebug("CSCGattiFunction")  << "Gatti function constants k1=" << 
      k1 << ", k2=" << k2 << ", k3=" << k3 << 
      ", h=" << h << ", norm=" << norm;
}


double CSCGattiFunction::binValue( double x, double stripWidth) const {
  double tanh1 = tanh(k2 * (x+stripWidth*0.5)/h );
  double tanh2 = tanh(k2 * (x-stripWidth*0.5)/h );
  return norm * ( std::atan(sqrtk3*tanh1) - std::atan(sqrtk3*tanh2) );
}

