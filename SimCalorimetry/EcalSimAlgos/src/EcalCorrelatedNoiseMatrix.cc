#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


EcalCorrelatedNoiseMatrix::EcalCorrelatedNoiseMatrix(const EcalCorrMatrix & matrix)
: theMatrix(matrix),
  theSize(theMatrix.kRows)
{
  if ( (int)theSize != CaloSamples::MAXSAMPLES  ) {
    throw cms::Exception("SizeDoesNotMatch", "ECAL noise correlation matrix size wrong");
    return;
  }
    
  edm::LogInfo("EcalNoise") << "Using the noise correlation matrix: " << theMatrix;
}


