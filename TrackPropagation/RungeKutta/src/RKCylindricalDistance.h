#ifndef RKCylindricalDistance_H
#define RKCylindricalDistance_H

#include "RKDistance.h"
#include "RKSmallVector.h"
#include "CylindricalState.h"

template <typename T, int N>
class dso_internal RKCylindricalDistance final : public RKDistance<T,N> {
public:
 
  typedef T                                   Scalar;
  typedef RKSmallVector<T,N>                  Vector;

  ~RKCylindricalDistance() override {}

  Scalar operator()( const Vector& a, const Vector& b, const Scalar& rho) const override {
      CylindricalState astate(rho,a,1.);
      CylindricalState bstate(rho,b,1.);
      return (astate.position()-bstate.position()).mag() +
	  (astate.momentum()-bstate.momentum()).mag() / bstate.momentum().mag();
  }
 
};

#endif
