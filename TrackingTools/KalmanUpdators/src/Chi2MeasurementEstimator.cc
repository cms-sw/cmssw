#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "Geometry/Surface/interface/BoundPlane.h"

std::pair<bool,double> 
Chi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TransientTrackingRecHit& aRecHit) const {
  
  MeasurementExtractor me(tsos);
  AlgebraicVector r(aRecHit.parameters(tsos) - me.measuredParameters(aRecHit));
  AlgebraicSymMatrix R(aRecHit.parametersError(tsos) + me.measuredError(aRecHit));
  int ierr; R.invert(ierr); // if (ierr != 0) throw exception;
  double est = R.similarity(r);
  return returnIt(est);
}
