#include "SimMuon/CSCDigitizer/src/CSCCrossGap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/HepPDT/interface/HepPDT.h"
#include "SimGeneral/HepPDT/interface/HepParticleData.h"
#include <cmath>

#include <iostream>
CSCCrossGap:: CSCCrossGap(int iam, float mom,  LocalVector gap)
: theBeta2(0.),
  theGamma(1.),
  loggam(0.),
  theGap(gap),
  theParticleData(0),
  clusters(),
  electronsInClusters(),
  steps(),
  elosses()
{
  iam = setParticle( iam ); // treat some types as others
  theParticleData = HepPDT::getParticleData(iam);
  double mass = theParticleData->mass();
  
  logGamma( mass, mom);
  LogDebug("CSCCrossGap")
     << "CSCCrossGap: simhit due to " << theParticleData->name() << "\n"
     << "mass = " << mass << "GeV/c2, momentum = " << mom << 
       " GeV/c, gap length = " << length() << " cm \n";
}

double CSCCrossGap::logGamma( double mass, float mom ) 
{
  theGamma = sqrt((mom/mass)*(mom/mass) + 1. );
  theBeta2 = 1. - 1./(theGamma*theGamma);
  double betgam = sqrt(theGamma*theGamma -1.);
  LogDebug("CSCCrossGap") << "gamma = " << theGamma << ", beta2 = " << theBeta2 <<
    ", beta*gamma = " << betgam;

  // The lowest value in table (=theGammaBins[0]) is ln(1.1)=0.0953102
  // (Compensate later if lower)
  loggam = log( std::max(1.1, theGamma ) ); // F-P literal IS double by default!
  LogDebug("CSCCrossGap") << "logGamma = " << loggam;

  return loggam;
}


int CSCCrossGap::setParticle(int iam)
{
  LogDebug("CSCCrossGap") << "input type = " << iam;
  
  switch ( iam ) {
    case 0:                 // @@ treat unknown type as a muon
    case 13:
      iam = 13;
        break;
    case 22:                // treat photon as electron
    case 11:
      iam = 11;
        break;
    default:
      break;
  }
  return iam;
}
