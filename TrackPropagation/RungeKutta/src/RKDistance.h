#ifndef RKDistance_H
#define RKDistance_H

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "RKSmallVector.h"

template <typename T, int N>
class dso_internal RKDistance {
public:
 
  typedef T                                   Scalar;
  typedef RKSmallVector<T,N>                  Vector;

  virtual ~RKDistance() {}

  virtual Scalar operator()( const Vector& a, const Vector& b, const Scalar& s) const = 0;

};

#endif
