#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


EcalCorrelatedNoiseMatrix::EcalCorrelatedNoiseMatrix(int nFrames)
: theMatrix(nFrames, 1),
  theSize(nFrames)
{

  if ( theSize != CaloSamples::MAXSAMPLES || CaloSamples::MAXSAMPLES != 10 ) {
    throw cms::Exception("SizeDoesNotMatch", "ECAL noise correlation matrix size wrong");
    return;
  }

  theMatrix[0][0] = 1.;
  theMatrix[0][1] = 0.67;
  theMatrix[0][2] = 0.53;
  theMatrix[0][3] = 0.44;
  theMatrix[0][4] = 0.39;
  theMatrix[0][5] = 0.36;
  theMatrix[0][6] = 0.38;
  theMatrix[0][7] = 0.35;
  theMatrix[0][8] = 0.36;
  theMatrix[0][9] = 0.32;
  
  theMatrix[1][0] = 0.67;
  theMatrix[1][1] = 1.;
  theMatrix[1][2] = 0.67;
  theMatrix[1][3] = 0.53;
  theMatrix[1][4] = 0.44;
  theMatrix[1][5] = 0.39;
  theMatrix[1][6] = 0.36;
  theMatrix[1][7] = 0.38;
  theMatrix[1][8] = 0.35;
  theMatrix[1][9] = 0.36;
  
  theMatrix[2][0] = 0.53;
  theMatrix[2][1] = 0.67;
  theMatrix[2][2] = 1.;
  theMatrix[2][3] = 0.67;
  theMatrix[2][4] = 0.53;
  theMatrix[2][5] = 0.44;
  theMatrix[2][6] = 0.39;
  theMatrix[2][7] = 0.36;
  theMatrix[2][8] = 0.38;
  theMatrix[2][9] = 0.35;
  
  theMatrix[3][0] = 0.44;
  theMatrix[3][1] = 0.53;
  theMatrix[3][2] = 0.67;
  theMatrix[3][3] = 1.;
  theMatrix[3][4] = 0.67;
  theMatrix[3][5] = 0.53;
  theMatrix[3][6] = 0.44;
  theMatrix[3][7] = 0.39;
  theMatrix[3][8] = 0.36;
  theMatrix[3][9] = 0.38;
  
  theMatrix[4][0] = 0.39;
  theMatrix[4][1] = 0.44;
  theMatrix[4][2] = 0.53;
  theMatrix[4][3] = 0.67;
  theMatrix[4][4] = 1.;
  theMatrix[4][5] = 0.67;
  theMatrix[4][6] = 0.53;
  theMatrix[4][7] = 0.44;
  theMatrix[4][8] = 0.39;
  theMatrix[4][9] = 0.36;
  
  theMatrix[5][0] = 0.36;
  theMatrix[5][1] = 0.39;
  theMatrix[5][2] = 0.44;
  theMatrix[5][3] = 0.53;
  theMatrix[5][4] = 0.67;
  theMatrix[5][5] = 1.;
  theMatrix[5][6] = 0.67;
  theMatrix[5][7] = 0.53;
  theMatrix[5][8] = 0.44;
  theMatrix[5][9] = 0.39;
  
  theMatrix[6][0] = 0.38;
  theMatrix[6][1] = 0.36;
  theMatrix[6][2] = 0.39;
  theMatrix[6][3] = 0.44;
  theMatrix[6][4] = 0.53;
  theMatrix[6][5] = 0.67;
  theMatrix[6][6] = 1.;
  theMatrix[6][7] = 0.67;
  theMatrix[6][8] = 0.53;
  theMatrix[6][9] = 0.44;
  
  theMatrix[7][0] = 0.35;
  theMatrix[7][1] = 0.38;
  theMatrix[7][2] = 0.36;
  theMatrix[7][3] = 0.39;
  theMatrix[7][4] = 0.44;
  theMatrix[7][5] = 0.53;
  theMatrix[7][6] = 0.67;
  theMatrix[7][7] = 1.;
  theMatrix[7][8] = 0.67;
  theMatrix[7][9] = 0.53;
  
  theMatrix[8][0] = 0.36;
  theMatrix[8][1] = 0.35;
  theMatrix[8][2] = 0.38;
  theMatrix[8][3] = 0.36;
  theMatrix[8][4] = 0.39;
  theMatrix[8][5] = 0.44;
  theMatrix[8][6] = 0.53;
  theMatrix[8][7] = 0.67;
  theMatrix[8][8] = 1.;
  theMatrix[8][9] = 0.67;
  
  theMatrix[9][0] = 0.32;
  theMatrix[9][1] = 0.36;
  theMatrix[9][2] = 0.35;
  theMatrix[9][3] = 0.38;
  theMatrix[9][4] = 0.36;
  theMatrix[9][5] = 0.39;
  theMatrix[9][6] = 0.44;
  theMatrix[9][7] = 0.53;
  theMatrix[9][8] = 0.67;
  theMatrix[9][9] = 1.;
    
  edm::LogInfo("EcalNoise") << "Using the noise correlation matrix: " << theMatrix;

}


EcalCorrelatedNoiseMatrix::EcalCorrelatedNoiseMatrix(const HepSymMatrix & matrix)
: theMatrix(matrix),
  theSize(theMatrix.num_row())
{
  if ( theSize != CaloSamples::MAXSAMPLES  ) {
    throw cms::Exception("SizeDoesNotMatch", "ECAL noise correlation matrix size wrong");
    return;
  }
    
  edm::LogInfo("EcalNoise") << "Using the noise correlation matrix: " << theMatrix;
}


