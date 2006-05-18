#ifndef CaloSimAlgos_CaloCorrelatedNoisifier_h
#define CaloSimAlgos_CaloCorrelatedNoisifier_h

/**
   \class CaloCorrelatedNoisifier

   \brief adds noise to the given frame

*/
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Matrix/SymMatrix.h"
class CaloSamples;

class CaloCorrelatedNoisifier
{
public:
  explicit CaloCorrelatedNoisifier(int nFrames);

  virtual ~CaloCorrelatedNoisifier() {}

  /// sets all elements along the diagonal of
  /// the correlation matrix to be value
  void setDiagonal(double value);
  
  void setOffDiagonal(int neighbor, double value);

  virtual void noisify(CaloSamples & frame) const;

private:
  HepSymMatrix theMatrix;
  mutable RandGauss theRandomGaussian;
  int theSize; 
};

#endif
