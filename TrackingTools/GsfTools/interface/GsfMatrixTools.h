#ifndef GsfMatrixTools_h_
#define GsfMatrixTools_h_

#define SMATRIX_USE_CONSTEXPR

#include "Math/SVector.h"
#include "Math/SMatrix.h"

namespace GsfMatrixTools {
  
//   template <unsigned int N>
//   double similarity (ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepSym<double, N> >,
// 		     ROOT::Math::SVector<double, N> vector) {
//     return vector*matrix*vector;
//   }

  template <typename T, unsigned int N>
  T trace (ROOT::Math::SMatrix<T, N, N> const & matrix) {
    T result = 0;
    for ( unsigned int i=0; i<N; ++i )  result += matrix(i,i);
    return result;
  }


  /* compute the trace of a product of two sym matrices
   *   a(i,j)*b(j,i) = a(i,j)*b(i,j) sum over i and j
   */
  template<typename T, unsigned int N>
  T trace(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & a,
	  ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & b) {
    auto i1 = a.begin();
    auto e1 = a.end();
    auto i2 = b.begin();
  
    T res =0;
    // sum of the lower triangle;
    for (;i1!=e1; i1++, i2++)
      res += (*i1)*(*i2);
    res *= T(2.);
    // remove the duplicated diagonal...
    for (unsigned int i=0;i<N;++i)
      res -= a(i,i)*b(i,i);
    return res;
  }
  
}
#endif

