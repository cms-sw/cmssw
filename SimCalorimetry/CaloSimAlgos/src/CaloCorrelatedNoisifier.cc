#include "SimCalorimetry/CaloSimAlgos/interface/CaloCorrelatedNoisifier.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Matrix/Vector.h"

CaloCorrelatedNoisifier::CaloCorrelatedNoisifier(int nFrames)
: theMatrix(nFrames, 1),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(nFrames)
{
}

void CaloCorrelatedNoisifier::setDiagonal(double value) 
{
  for(int i = 0; i < theSize; ++i) 
  {
    theMatrix[i][i] = value;
  }
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
}

#include <iostream>
void CaloCorrelatedNoisifier::noisify(CaloSamples & frame) const
{
  // make a vector of random values
  assert(frame.size() == theSize);
  HepVector uncorrelated(theSize, theRandomGaussian);
  
  // rotate them to make a correlated noise vector
  HepVector correlated = theMatrix * uncorrelated; 

  // stuff 'em in the frame
  for(int i = 0; i < theSize; ++i)
  {
    frame[i] += correlated[i];
  }
}

