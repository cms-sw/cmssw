#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "CLHEP/Matrix/Vector.h"
#include <stdexcept>

CorrelatedNoisifier::CorrelatedNoisifier(int nFrames)
: theMatrix(nFrames, 1.0),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(nFrames),
  theNorma(0.,theSize)
{
  computeNormalization(); 

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

}


CorrelatedNoisifier::CorrelatedNoisifier(const HepSymMatrix & matrix)
: theMatrix(matrix.num_row(),matrix),
  theRandomGaussian(*(HepRandom::getTheEngine())),
  theSize(theMatrix.rank()),
  theNorma(0.,theSize)
{
  computeNormalization(); 

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

}


void CorrelatedNoisifier::setDiagonal(double value) 
{
  for(int i = 0; i < theSize; ++i) 
  {
    theMatrix(i,i) = value;
  }

  computeNormalization(); 

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

} 

void CorrelatedNoisifier::setOffDiagonal(int distance, double value)
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


void CorrelatedNoisifier::computeNormalization() 
{
  theNorma = 0;
  noiseMath::SparseMatrix<double>::const_iterator p = theMatrix.values().begin();
  noiseMath::SparseMatrix<double>::const_iterator e = theMatrix.values().end();
  for (;p!=e;p++)
    theNorma[(*p).i] += (*p).v*(*p).v;
  for (int i=0; i<theSize; i++)
  {
    if (theNorma[i] <= 0. ) throw(std::runtime_error("CorrelatedNoisifier:  normalization equal to zero."));
    theNorma[i] /= theMatrix(i,i);
  }
  theNorma = 1./std::sqrt(theNorma);

}

void CorrelatedNoisifier::checkOffDiagonal(bool & isDiagonal_){

  isDiagonal_ = true;

  for ( int i = 0 ; i < theSize ; i++ ) {
    for ( int j = 0 ; j < theSize ; j++ ) {

      if ( i != j && theMatrix(i,j) != 0. ) { isDiagonal_ = false ; return ; }
      
    }
  }
  

}
