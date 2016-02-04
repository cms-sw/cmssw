#ifndef VectorDoublet_H
#define VectorDoublet_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/PreciseFloatType.h"

template <class V1, class V2>
class dso_internal VectorDoublet {
public:

  typedef typename V1::ScalarType Scalar1;
  typedef typename V2::ScalarType Scalar2;

  VectorDoublet() {}
  VectorDoublet( const V1& a, const V2& b) : a_(a), b_(b) {}

  const V1& first() const {return a_;}
  const V2& second() const {return b_;}

  VectorDoublet& operator+= ( const VectorDoublet& v) {
    a_ += v.first();
    b_ += v.second();
    return *this;
  } 
  VectorDoublet& operator-= ( const VectorDoublet& v) {
    a_ -= v.first();
    b_ -= v.second();
    return *this;
  } 

  VectorDoublet operator-() const { return VectorDoublet( -a_, -b_);}

  template <class T>
  VectorDoublet& operator*= ( const T& t) {
    a_ *= t;
    b_ *= t;
    return *this;
  } 
  template <class T>
  VectorDoublet& operator/= ( const T& t) {
    a_ /= t;
    b_ /= t;
    return *this;
  } 

  typename PreciseFloatType<Scalar1,Scalar2>::Type dot( const VectorDoublet& v) const { 
    return first()*v.first() + second()*v.second();
  }

private:

  V1 a_;
  V2 b_;

};

/// vector sum and subtraction
template <class V1, class V2>
inline VectorDoublet<V1,V2>
operator+( const VectorDoublet<V1,V2>& a, const VectorDoublet<V1,V2>& b) {
  return VectorDoublet<V1,V2>(a.first()+b.first(), a.second()+b.second());
}

template <class V1, class V2>
inline VectorDoublet<V1,V2>
operator-( const VectorDoublet<V1,V2>& a, const VectorDoublet<V1,V2>& b) {
  return VectorDoublet<V1,V2>(a.first()-b.first(), a.second()-b.second());
}

/** Multiplication by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class V1, class V2, class Scalar>
inline VectorDoublet<V1,V2> operator*( const VectorDoublet<V1,V2>& v, const Scalar& s) {
  return VectorDoublet<V1,V2>( v.first()*s, v.second()*s);
}
template <class V1, class V2, class Scalar>
inline VectorDoublet<V1,V2> operator*( const Scalar& s, const VectorDoublet<V1,V2>& v) {
  return VectorDoublet<V1,V2>( v.first()*s, v.second()*s);
}

template <class V1, class V2, class Scalar>
inline VectorDoublet<V1,V2> operator/( const VectorDoublet<V1,V2>& v, const Scalar& s) {
  return VectorDoublet<V1,V2>( v.first()/s, v.second()/s);
}

#endif
