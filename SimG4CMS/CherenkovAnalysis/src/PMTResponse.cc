#include "SimG4CMS/CherenkovAnalysis/interface/PMTResponse.h"

#include <cmath>

//________________________________________________________________________________________
double PMTResponse::getEfficiency(const double &waveLengthNm) {
  // Overall range
  if (waveLengthNm < 300. || waveLengthNm > 850)
    return 0.;

  // Parameterisation
  if (waveLengthNm < 500.)
    return std::exp(+waveLengthNm / 144.3 - 5.0752);
  else
    return std::exp(-waveLengthNm / 290.7 + 0.1105);
}
