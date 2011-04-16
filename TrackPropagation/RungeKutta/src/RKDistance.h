#ifndef RKDistance_H
#define RKDistance_H

#include "RKSmallVector.h"

template <typename T, int N>
class RKDistance {
public:
 
  typedef T                                   Scalar;
  typedef RKSmallVector<T,N>                  Vector;

  virtual ~RKDistance() {}

  virtual Scalar operator()( const Vector& a, const Vector& b, const Scalar& s) const = 0;

};

#endif
