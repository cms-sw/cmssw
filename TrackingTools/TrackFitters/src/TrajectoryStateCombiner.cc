#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

TrajectoryStateOnSurface 
TrajectoryStateCombiner::combine(const TSOS& Tsos1, const TSOS& Tsos2) const {

  double pzSign = Tsos1.localParameters().pzSign();
  AlgebraicVector5 x1(Tsos1.localParameters().vector());
  AlgebraicVector5 x2(Tsos2.localParameters().vector());
  const AlgebraicSymMatrix55 &C1 = (Tsos1.localError().matrix());
  const AlgebraicSymMatrix55 &C2 = (Tsos2.localError().matrix());

  AlgebraicSymMatrix55 Csum = C1 + C2;
  bool ok = invertPosDefMatrix(Csum);
  AlgebraicMatrix55 K = C1*Csum;

  if(!ok) {
    edm::LogError("MatrixInversionFailure")
      <<"the inversion of the combined error matrix failed. Impossible to get a combined state."
      <<"\nmatrix 1:"<<C1
      <<"\nmatrix 2:"<<C2;
    return TSOS();
  }

  AlgebraicVector5 xcomb = x1 + K*(x2 - x1);
  //AlgebraicSymMatrix55 Ccomb; Ccomb.assign(K*C2);
  AlgebraicSymMatrix55 Ccomb = (AlgebraicMatrix55(K*C2)).LowerBlock();

  TSOS combTsos( LocalTrajectoryParameters(xcomb, pzSign),
		 LocalTrajectoryError(Ccomb), Tsos1.surface(),
		 &(Tsos1.globalParameters().magneticField()));
  return combTsos;  
}
