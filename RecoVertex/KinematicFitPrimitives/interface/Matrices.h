#ifndef _Kinefit_SMATRIX_H_
#define _Kinefit_SMATRIX_H_

#include <Math/SVector.h>
#include <Math/SMatrix.h>

typedef ROOT::Math::SVector<double,7> AlgebraicVector7;
typedef ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepSym<double,7> > AlgebraicSymMatrix77;
typedef ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepStd<double,7,7> > AlgebraicMatrix77;
typedef ROOT::Math::SMatrix<double,6,7,ROOT::Math::MatRepStd<double,6,7> > AlgebraicMatrix67;
typedef ROOT::Math::SMatrix<double,7,6,ROOT::Math::MatRepStd<double,7,6> > AlgebraicMatrix76;

#endif
