#include "SimMuon/CSCDigitizer/src/CSCCrossGap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>

#include <iostream>
CSCCrossGap:: CSCCrossGap(double mass, float mom, LocalVector gap)
: theBeta2(0.),
  theGamma(1.),
  loggam(0.),
  theGap(gap),
  clusters(),
  electronsInClusters(),
  steps(),
  elosses()
{
  logGamma( mass, mom);
  LogTrace("CSCCrossGap")
     << "CSCCrossGap: simhit \n"
     << "mass = " << mass << "GeV/c2, momentum = " << mom << 
       " GeV/c, gap length = " << length() << " cm \n";
}

double CSCCrossGap::logGamma( double mass, float mom ) 
{
  theGamma = sqrt((mom/mass)*(mom/mass) + 1. );
  theBeta2 = 1. - 1./(theGamma*theGamma);
  double betgam = sqrt(theGamma*theGamma -1.);
  LogTrace("CSCCrossGap") << "gamma = " << theGamma << ", beta2 = " << theBeta2 <<
    ", beta*gamma = " << betgam;

  // The lowest value in table (=theGammaBins[0]) is ln(1.1)=0.0953102
  // (Compensate later if lower)
  loggam = log( std::max(1.1, theGamma ) ); // F-P literal IS double by default!
  LogTrace("CSCCrossGap") << "logGamma = " << loggam;

  return loggam;
}

