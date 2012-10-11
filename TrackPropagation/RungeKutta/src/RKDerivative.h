#ifndef RKDerivative_H
#define RKDerivative_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKSmallVector.h"

/// Base class for derivative calculation. 

template <typename T, int N>
class dso_internal RKDerivative {
public:
 
  typedef T                                   Scalar;
  typedef RKSmallVector<T,N>                  Vector;

  virtual ~RKDerivative() {}

  virtual Vector operator()( Scalar startPar, const Vector& startState) const = 0;

};

#endif
