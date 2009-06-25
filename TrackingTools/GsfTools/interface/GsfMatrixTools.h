#ifndef GsfMatrixTools_h_
#define GsfMatrixTools_h_

#include "Math/SVector.h"
#include "Math/SMatrix.h"

namespace GsfMatrixTools {
  
//   template <unsigned int N>
//   double similarity (ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepSym<double, N> >,
// 		     ROOT::Math::SVector<double, N> vector) {
//     return vector*matrix*vector;
//   }

  template <unsigned int N>
  double trace (const ROOT::Math::SMatrix<double, N, N>& matrix) {
    double result(0.);
    for ( unsigned int i=0; i<N; i++ )  result += matrix(i,i);
    return result;
  }


  /* compute the trace of a product of two sym matrices
   *   a(i,j)*b(j,i) = a(i,j)*b(i,j) sum over i and j
   */
  template<typename T, unsigned int N>
  double trace(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & a,
	       ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & b) {
    typedef typename ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> >::const_iterator CI;
    CI i1 = a.begin();
    CI e1 = a.end();
    CI i2 = b.begin();
    //  CI e2 = b.end();
  
    T res =0;
    // sum of the lower triangle;
    for (;i1!=e1; i1++, i2++)
      res += (*i1)*(*i2);
    res *=2.;
    // remove the duplicated diagonal...
    for (unsigned int i=0;i<N;i++)
      res -= a(i,i)*b(i,i);
  return res;
  }
  
}
#endif

