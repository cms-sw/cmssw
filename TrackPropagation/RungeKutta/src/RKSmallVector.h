#ifndef RKSmallVector_H
#define RKSmallVector_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/PreciseFloatType.h"

#include <iostream>

template<typename T, int N> 
class dso_internal RKSmallVector {
public:
  typedef T                                   Scalar;

  RKSmallVector() {}

  /// Construct from array
  RKSmallVector( const Scalar* d) {
    for (int i=0; i<N; ++i) data_[i] = d[i];
  }

  /// construct from pair of iterators; when dereferenced, they should 
  /// return a type convertible to Scalar. The iterator range shold be exactly N
  template <class Iter> RKSmallVector(Iter begin, Iter end) {
    for (int i=0; i<N; i++) data_[i] = *begin++;
  }

  int size() const {return N;}
  int dim() const {return N;}

  /// access to values
  const T& operator()(int i) const {return data_[i];}
  T& operator()(int i) {return data_[i];}
  const T& operator[](int i) const {return data_[i];}
  T& operator[](int i) {return data_[i];}

  /// Operations with vector of same size

  template <class U> 
  RKSmallVector& operator+= ( const RKSmallVector<U,N>& v) {
    for (int i=0; i<N; ++i) data_[i] += v[i];
    return *this;
  } 

  template <class U> 
  RKSmallVector& operator-= ( const RKSmallVector<U,N>& v) {
    for (int i=0; i<N; ++i) data_[i] -= v[i];
    return *this;
  } 

  /// Increment by another vector multiplied by a scalar
  template <class U> 
  RKSmallVector& increment( const RKSmallVector<U,N>& v, const T& t) {
    for (int i=0; i<N; ++i) data_[i] += t*v[i];
    return *this;
  } 


  /// Unary minus, returns a vector with negated components 
  RKSmallVector operator-() const { 
    RKSmallVector r;
    for (int i=0; i<N; ++i) r[i] = -data_[i];
    return r;
  }

  /// Scaling by a scalar value (multiplication)
  RKSmallVector& operator*= ( const T& t) {
    for (int i=0; i<N; ++i) data_[i] *= t;
    return *this;
  }
  /// Scaling by a scalar value (division)
  RKSmallVector& operator/= ( const T& t) {
    for (int i=0; i<N; ++i) data_[i] /= t;
    return *this;
  }

  /// Scalar product, or "dot" product, with a vector of same type.
  T dot( const RKSmallVector& v) const {
    RKSmallVector r;
    for (int i=0; i<N; ++i) r[i] = data_[i]*data_[i];
    return r;
  }

private:

  Scalar data_[N];

};

/// simple text output to standard streams
template <class T, int N>
inline std::ostream & operator<<( std::ostream& s, const RKSmallVector<T,N>& v) {
  s << std::endl;
  for (int i=0; i<N; ++i) s << "v[" << i << "] = " << v[i] << std::endl;
  return s;
} 

/// vector sum and subtraction of vectors of possibly different precision
template <class T, class U, int N>
inline RKSmallVector<typename PreciseFloatType<T,U>::Type, N>
operator+( const RKSmallVector<T, N>& a, const RKSmallVector<U, N>& b) {
  typedef RKSmallVector<typename PreciseFloatType<T,U>::Type, N> RT;
  RT r;
  for (int i=0; i<N; ++i) r[i] = a[i]+b[i];
  return r;
}
template <class T, class U, int N>
inline RKSmallVector<typename PreciseFloatType<T,U>::Type, N>
operator-( const RKSmallVector<T, N>& a, const RKSmallVector<U, N>& b) {
  typedef RKSmallVector<typename PreciseFloatType<T,U>::Type, N> RT;
  RT r;
  for (int i=0; i<N; ++i) r[i] = a[i]-b[i];
  return r;
}

/** Multiplication by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class Scal, int N>
inline RKSmallVector<T,N> operator*( const RKSmallVector<T,N>& v, const Scal& s) {
  T t = static_cast<T>(s);
  RKSmallVector<T,N> r;
  for (int i=0; i<N; ++i) r[i] = t*v[i];
  return r;
}
template <class T, class Scal, int N>
inline RKSmallVector<T,N> operator*( const Scal& s, const RKSmallVector<T,N>& v) {
  T t = static_cast<T>(s);
  RKSmallVector<T,N> r;
  for (int i=0; i<N; ++i) r[i] = t*v[i];
  return r;
}

/** Division by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class Scal, int N>
inline RKSmallVector<T,N> operator/( const RKSmallVector<T,N>& v, const Scal& s) {
  T t = static_cast<T>(s);
  RKSmallVector<T,N> r;
  for (int i=0; i<N; ++i) r[i] = v[i]/t;
  return r;
}


#endif
