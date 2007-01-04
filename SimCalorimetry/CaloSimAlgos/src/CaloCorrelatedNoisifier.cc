#include "SimCalorimetry/CaloSimAlgos/interface/CaloCorrelatedNoisifier.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Matrix/Vector.h"
#include <stdexcept>

CaloCorrelatedNoisifier::CaloCorrelatedNoisifier(int nFrames)
: theCovarianceMatrix(nFrames, 1.0),
  theMatrix(nFrames, 1.0),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(nFrames)
{

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();

}


CaloCorrelatedNoisifier::CaloCorrelatedNoisifier(const HepSymMatrix & matrix)
: theCovarianceMatrix(matrix.num_row(),matrix),
  theMatrix(matrix.num_row(),1.0),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(theCovarianceMatrix.rank())
{

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();

}


void CaloCorrelatedNoisifier::setDiagonal(double value) 
{
  for(int i = 0; i < theSize; ++i) 
  {
    theCovarianceMatrix(i,i) = value;
  }

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();

} 

void CaloCorrelatedNoisifier::setOffDiagonal(int distance, double value)
{
  for(int column = 0; column < theSize; ++column)
  {
    int row = column - distance;
    if(row < 0) continue;
    theCovarianceMatrix(row,column) = value;
    theCovarianceMatrix(column,row) = value;

  }

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();

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
        frame[i] += correlated[i];
      }
  }
}

void CaloCorrelatedNoisifier::computeDecomposition()
{

  for ( int i = 0 ; i < theSize ; i++ ) {
    for ( int j = 0 ; j < theSize ; j++ ) {
      theMatrix(i,j) = 0.;
    }
  }

  double sqrtSigma00 = theCovarianceMatrix(0,0);
  if ( sqrtSigma00 <= 0. ) throw(std::runtime_error("CaloCorrelatedNoisifier: non positive variance."));
  sqrtSigma00 = std::sqrt(sqrtSigma00);

  for ( int i = 0 ; i < theSize ; i++ )
    {
      double hi0 = theCovarianceMatrix(i,0)/sqrtSigma00;
      theMatrix(i,0) = hi0;
    }

  for ( int i = 1 ; i < theSize ; i++ ) 
    {

      for ( int j = 1 ; j < i ; j++ )
        {
          double hij = theCovarianceMatrix(i,j);
          for ( int k = 0 ; k <= j-1 ; k++ ) hij -= theMatrix(i,k)*theMatrix(j,k);
          hij /= theMatrix(j,j);
          theMatrix(i,j) = hij;
        }
      
      double hii = theCovarianceMatrix(i,i);
      for ( int j = 0 ; j <= i-1 ; j++ ) {
        double hij = theMatrix(i,j);
        hii -= hij*hij;
      }
      hii = sqrt(hii);
      theMatrix(i,i) = hii;

    }

}

void CaloCorrelatedNoisifier::checkOffDiagonal(bool & isDiagonal_){

  isDiagonal_ = true;

  for ( int i = 0 ; i < theSize ; i++ ) {
    for ( int j = 0 ; j < theSize ; j++ ) {

      if ( i != j && theCovarianceMatrix(i,j) != 0. ) { isDiagonal_ = false ; return ; }
      
    }
  }
  

}
