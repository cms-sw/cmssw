#include "SimCalorimetry/CaloSimAlgos/interface/CaloCorrelatedNoisifier.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Matrix/Vector.h"
#include <stdexcept>

CaloCorrelatedNoisifier::CaloCorrelatedNoisifier(int nFrames)
: theMatrix(nFrames, 1),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(nFrames)
{
  theNorma.reserve(theSize);
  theNorma.clear();
  normaDone = false;
}


CaloCorrelatedNoisifier::CaloCorrelatedNoisifier(const HepSymMatrix & matrix)
: theMatrix(matrix),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(theMatrix.num_row())
{
  theNorma.reserve(theSize);
  theNorma.clear();
  normaDone = false;
}

void CaloCorrelatedNoisifier::setDiagonal(double value) 
{
  for(int i = 0; i < theSize; ++i) 
  {
    theMatrix[i][i] = value;
  }

  computeNormalization(theNorma); 
} 

void CaloCorrelatedNoisifier::setOffDiagonal(int distance, double value)
{
  for(int column = 0; column < theSize; ++column)
  {
    // first do the upper neighbor
    int row = column - distance;
    if(row < 0) row += theSize;
    theMatrix[row][column] = value;

    // and the lower neighbor
    // probably unnecessary in a symmetric matrix
    row = column + distance;
    if(row >= theSize) row -= theSize;
    theMatrix[row][column] = value;
  }

  computeNormalization(theNorma); 
}

#include <iostream>
void CaloCorrelatedNoisifier::noisify(CaloSamples & frame) 
{

  //check if the normalizations are avilable, otherwise compute them
  if ( ! normaDone) computeNormalization(theNorma); 

  // make a vector of random values
  assert(frame.size() == theSize);
  HepVector uncorrelated(theSize, theRandomGaussian);
  
  // rotate them to make a correlated noise vector
  HepVector correlated = theMatrix * uncorrelated; 

  // stuff 'em in the frame
  for(int i = 0; i < theSize; ++i)
  {
    frame[i] += (correlated[i]/theNorma[i]);
  }
}

void CaloCorrelatedNoisifier::computeNormalization(std::vector<double> & norma) 
{
  normaDone = false;

  for (int row = 0; row < theSize; ++row) {

    double part1 = 0.;

    for (int j = 0; j < theSize; ++j ) {
      part1 += theMatrix[row][j]*theMatrix[row][j];
    }
    
    if ( part1 <= 0. ) throw(std::runtime_error("CaloCorrelatedNoisifier:  normalization equal to zero."));

    theNorma[row] = sqrt(part1);

  }

  normaDone = true;

}
