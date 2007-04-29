#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

// #include <iostream>

// using namespace std;

TrajectoryStateOnSurface 
TrajectoryStateCombiner::combine(const TSOS& Tsos1, const TSOS& Tsos2) const {

  int ierr;
  double pzSign = Tsos1.localParameters().pzSign();
  AlgebraicVector5 x1(Tsos1.localParameters().vector());
  AlgebraicVector5 x2(Tsos2.localParameters().vector());
  const AlgebraicSymMatrix55 &C1 = (Tsos1.localError().matrix());
  const AlgebraicSymMatrix55 &C2 = (Tsos2.localError().matrix());

  AlgebraicSymMatrix55 Csum = C1 + C2;
  AlgebraicMatrix55 K = C1*(Csum.Inverse(ierr));

  if(ierr != 0) {
//     if ( infoV )
//       cout<<"KFTrajectorySmoother: inversion of Csum failed!"
// 	  <<Tsos1.localError().matrix()<<endl;
    return TSOS();
  }

  AlgebraicVector5 xcomb = x1 + K*(x2 - x1);
  //AlgebraicSymMatrix55 Ccomb; Ccomb.assign(K*C2);
  AlgebraicSymMatrix55 Ccomb = ((const AlgebraicMatrix55 &)(K*C2)).LowerBlock();

  TSOS combTsos( LocalTrajectoryParameters(xcomb, pzSign),
		 LocalTrajectoryError(Ccomb), Tsos1.surface(),
		 &(Tsos1.globalParameters().magneticField()));
  return combTsos;  
}
