#ifndef LinearEquation3_H
#define LinearEquation3_H

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include <algorithm>

#ifdef DEBUG_SOLUTION
#include <iostream>
#endif

template <class T>
class LinearEquation3 {
public:

  template <class U> 
  class Array3 {
  public:
    Array3() {}
    Array3( U a0, U a1, U a2) { a_[0] = a0; a_[1] = a1; a_[2] = a2;}
    Array3( const Basic3DVector<U>& v) {
      a_[0] = v.x(); a_[1] = v.y(); a_[2] = v.z();
    }

    Array3& operator=( const Array3& other) {
      a_[0] = other[0]; a_[1] = other[1]; a_[2] = other[2];
      return *this;
    }
      
    Array3& operator=(const Basic3DVector<U>& v) {
      a_[0] = v.x(); a_[1] = v.y(); a_[2] = v.z();
      return *this;
    }

    U& operator[]( int i) { return a_[i];}
    const U& operator[]( int i) const { return a_[i];}
    void operator-=( const Array3& other) {
      a_[0] -= other[0];
      a_[1] -= other[1];
      a_[2] -= other[2];
    }

    Array3 operator*( U t) const {
      return Array3( a_[0]*t, a_[1]*t, a_[2]*t);
    }

    void subtractScaled( const Array3& a, U c) {
      a_[0] -= a[0]*c;  a_[1] -= a[1]*c;  a_[2] -= a[2]*c; 
    }

  private:
    U a_[3];
  };

  Basic3DVector<T> solution( const Basic3DVector<T>& row0,
			     const Basic3DVector<T>& row1,
			     const Basic3DVector<T>& row2,
			     const Basic3DVector<T>& rhsvec) const {

    // copy the input to internal "matrix"
    Array3<T> row[3];
    row[0] = row0;
    row[1] = row1;
    row[2] = row2;
    Array3<T> rhs(rhsvec);

    // no implicit pivoting - rows expected to be normalized already

    // find pivot 0, i.e. row with largest first element
    int i0 = std::abs(row[0][0]) > std::abs(row[1][0]) ? 0 : 1;
    if (std::abs(row[i0][0]) < std::abs(row[2][0])) i0 = 2;

    int i1 = (i0+1)%3;
    int i2 = (i0+2)%3;

    // zero the first column of rows i1 and i2
    T c1 = row[i1][0] / row[i0][0];
    // row[i1] -= c1*row[i0];
    row[i1].subtractScaled( row[i0], c1);
    rhs[i1] -= c1*rhs[i0];
    T c2 = row[i2][0] / row[i0][0];
    // row[i2] -= c2*row[i0];
    row[i2].subtractScaled( row[i0], c2);
    rhs[i2] -= c2*rhs[i0];

    // find pivot 1, i.e. which row (i1 or i2) has the largest second element
    if (std::abs(row[i1][1]) < std::abs(row[i2][1])) std::swap( i1, i2);

    // zero the second column of row i2
    T c3 = row[i2][1] / row[i1][1];
    row[i2][1] -= c3 * row[i1][1];
    row[i2][2] -= c3 * row[i1][2];
    rhs[i2] -= c3*rhs[i1];

    // compute the solution
    T x2 = rhs[i2] / row[i2][2];
    T x1 = (rhs[i1] - x2*row[i1][2]) / row[i1][1];
    T x0 = (rhs[i0] - x1*row[i0][1] - x2*row[i0][2]) / row[i0][0];

    return Basic3DVector<T>(x0, x1, x2);
  }

#ifdef DEBUG_SOLUTION
private:
  typedef Array3<T> AT;
  void dump(const AT row[]) const {
    std::cout << " (" << row[0][0] << ',' << row[0][1] << ',' << row[0][2] << ") " << std::endl;
    std::cout << " (" << row[1][0] << ',' << row[1][1] << ',' << row[1][2] << ") " << std::endl;
    std::cout << " (" << row[2][0] << ',' << row[2][1] << ',' << row[2][2] << ") " << std::endl;
    std::cout << std::endl;
  }
#endif

};

#endif
