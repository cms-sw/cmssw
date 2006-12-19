#include "SimCalorimetry/CaloSimAlgos/interface/CaloCorrelatedNoisifier.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Matrix/Vector.h"
#include <stdexcept>

CaloCorrelatedNoisifier::CaloCorrelatedNoisifier(int nFrames)
: theMatrix(nFrames, 1.0),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(nFrames),
  theNorma(0.,theSize)
{
  computeNormalization(); 

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

}


CaloCorrelatedNoisifier::CaloCorrelatedNoisifier(const HepSymMatrix & matrix)
: theMatrix(matrix.num_row(),matrix),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(theMatrix.rank()),
  theNorma(0.,theSize)
{
  computeNormalization(); 

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

}


void CaloCorrelatedNoisifier::setDiagonal(double value) 
{
  for(int i = 0; i < theSize; ++i) 
  {
    theMatrix(i,i) = value;
  }

  computeNormalization(); 

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

} 

void CaloCorrelatedNoisifier::setOffDiagonal(int distance, double value)
{
  for(int column = 0; column < theSize; ++column)
  {
    int row = column - distance;
    if(row < 0) continue;
    theMatrix(row,column) = value;
    theMatrix(column,row) = value;

  }

  computeNormalization(); 

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

}

#include <iostream>
void CaloCorrelatedNoisifier::noisify(CaloSamples & frame) 
{

 // make a vector of random values

  assert(frame.size() == theSize);
  std::valarray<double> uncorrelated(0.,theSize);
  for (int i=0; i<theSize; i++)
    uncorrelated[i]=theRandomGaussian.shoot();

  if ( isDiagonal_ ) {     
    for(int i = 0; i < theSize; ++i)
      { frame[i] += uncorrelated[i]; }
  }
  
  else {
    
    // rotate them to make a correlated noise vector
    std::valarray<double> correlated = theMatrix * uncorrelated; 
    
    // stuff 'em in the frame
    for(int i = 0; i < theSize; ++i)
      {
        frame[i] += (correlated[i]*theNorma[i]);
      }
  }
}

void CaloCorrelatedNoisifier::computeNormalization() 
{
  theNorma = 0;
  caloMath::SparseMatrix<double>::const_iterator p = theMatrix.values().begin();
  caloMath::SparseMatrix<double>::const_iterator e = theMatrix.values().end();
  for (;p!=e;p++)
    theNorma[(*p).i] += (*p).v*(*p).v;
  for (int i=0; i<theSize; i++)
    if (theNorma[i] <= 0. ) throw(std::runtime_error("CaloCorrelatedNoisifier:  normalization equal to zero."));
  theNorma = 1./std::sqrt(theNorma);

}

void CaloCorrelatedNoisifier::checkOffDiagonal(bool & isDiagonal_){

  isDiagonal_ = true;

  for ( int i = 0 ; i < theSize ; i++ ) {
    for ( int j = 0 ; j < theSize ; j++ ) {

      if ( i != j && theMatrix(i,j) != 0. ) { isDiagonal_ = false ; return ; }
      
    }
  }
  

}
