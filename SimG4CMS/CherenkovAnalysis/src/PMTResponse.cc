#include "TMath.h"

#include "SimG4CMS/CherenkovAnalysis/interface/PMTResponse.h"

//________________________________________________________________________________________
const double PMTResponse::getEfficiency( const double& waveLengthNm ) {

  // Overall range
  if ( waveLengthNm<300. || waveLengthNm>850 ) return 0.;

  // Parameterisation
  if ( waveLengthNm<500. )
    return TMath::Exp(+waveLengthNm/144.3 - 5.0752);
  else
    return TMath::Exp(-waveLengthNm/290.7 + 0.1105);
  
}
