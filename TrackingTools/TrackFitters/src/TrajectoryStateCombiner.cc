#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

// #include <iostream>

// using namespace std;

TrajectoryStateOnSurface 
TrajectoryStateCombiner::combine(const TSOS& Tsos1, const TSOS& Tsos2) const {

  int ierr;
  double pzSign = Tsos1.localParameters().pzSign();
  AlgebraicVector x1(Tsos1.localParameters().vector());
  AlgebraicVector x2(Tsos2.localParameters().vector());
  AlgebraicSymMatrix C1(Tsos1.localError().matrix());
  AlgebraicSymMatrix C2(Tsos2.localError().matrix());

  AlgebraicSymMatrix Csum = C1 + C2;
  AlgebraicMatrix K = C1*(Csum.inverse(ierr));

  if(ierr != 0) {
//     if ( infoV )
//       cout<<"KFTrajectorySmoother: inversion of Csum failed!"
// 	  <<Tsos1.localError().matrix()<<endl;
    return TSOS();
  }

  AlgebraicVector xcomb = x1 + K*(x2 - x1);
  AlgebraicSymMatrix Ccomb; Ccomb.assign(K*C2);

  TSOS combTsos( LocalTrajectoryParameters(xcomb, pzSign),
		 LocalTrajectoryError(Ccomb), Tsos1.surface(),
		 &(Tsos1.globalParameters().magneticField()));
  return combTsos;  
}
