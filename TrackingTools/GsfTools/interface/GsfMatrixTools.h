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
}
#endif

