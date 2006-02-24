#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"

TrajectoryStateOnSurface 
TrajectoryStateWithArbitraryError::operator()(const TSOS& aTsos) const {
  
  AlgebraicSymMatrix C(5,1);
  C *= 100.;

  return TSOS( aTsos.localParameters(), LocalTrajectoryError(C), 
	       aTsos.surface(),
	       &(aTsos.globalParameters().magneticField()));
}
