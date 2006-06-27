#ifndef EcalSimAlgos_EcalCorrelatedNoiseMatrix_h
#define EcalSimAlgos_EcalCorrelatedNoiseMatrix_h

/*
 *
 * $Id:$
 *
 */

#include "CLHEP/Matrix/SymMatrix.h"

class EcalCorrelatedNoiseMatrix
{

 public:

  explicit EcalCorrelatedNoiseMatrix(int nFrames);
 
  explicit EcalCorrelatedNoiseMatrix(const HepSymMatrix & matrix);

  ~EcalCorrelatedNoiseMatrix() {};

  void setMatrix(const HepSymMatrix & matrix) { theMatrix = matrix; }

  void getMatrix(HepSymMatrix & matrix) { matrix = theMatrix; }

private:

  HepSymMatrix theMatrix;
  int theSize; 
};

#endif
