#ifndef RKCurvilinearDistance_H
#define RKCurvilinearDistance_H

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "FWCore/Utilities/interface/Visibility.h"
#include "RKDistance.h"
#include "RKSmallVector.h"

template <typename T, int N>
class dso_internal RKCurvilinearDistance : public RKDistance<T,N> {
public:
 
  typedef T                                   Scalar;
  typedef RKSmallVector<T,N>                  Vector;

  ~RKCurvilinearDistance() override {}

  Scalar operator()( const Vector& a, const Vector& b, const Scalar& s) const override {
      Basic3DVector<Scalar> amom = momentum(a);
      Basic3DVector<Scalar> bmom = momentum(b);

    return sqrt( sqr(a(0)-b(0)) + sqr(a(1)-b(1))) + (amom - bmom).mag() / bmom.mag();
  }

  Basic3DVector<Scalar> momentum( const Vector& v) const {
      Scalar k = sqrt(1 + sqr(v(2)) + sqr(v(3)));
      Scalar p = std::abs(1 / v(4));
      Scalar pz = p/k;
      return Basic3DVector<Scalar>( v(2)*pz, v(3)*pz, pz);
  }

  T sqr(const T& t) const {return t*t;}
 
};

#endif
