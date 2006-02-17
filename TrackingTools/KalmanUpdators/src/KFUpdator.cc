#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/Surface/interface/BoundPlane.h"

TrajectoryStateOnSurface KFUpdator::update(const TrajectoryStateOnSurface& tsos,
				           const TransientTrackingRecHit& aRecHit) const {

  double pzSign = tsos.localParameters().pzSign();

  MeasurementExtractor me(tsos);

  AlgebraicVector x(tsos.localParameters().vector());
  AlgebraicSymMatrix C(tsos.localError().matrix());
  // Measurement matrix
  AlgebraicMatrix H(aRecHit.projectionMatrix());

  // Residuals of aPredictedState w.r.t. aRecHit, 
  AlgebraicVector r(aRecHit.parameters(tsos) - me.measuredParameters(aRecHit));

  // and covariance matrix of residuals
  AlgebraicSymMatrix V(aRecHit.parametersError(tsos));
  AlgebraicSymMatrix R(V + me.measuredError(aRecHit));
  int ierr; R.invert(ierr); // if (ierr != 0) throw exception;

  // Compute Kalman gain matrix
  AlgebraicMatrix K(C * H.T() * R);

  // Compute local filtered state vector
  AlgebraicVector fsv(x + K * r);

  // Compute covariance matrix of local filtered state vector
  AlgebraicSymMatrix I(5, 1);
  AlgebraicMatrix M(I - K * H);
  AlgebraicSymMatrix fse(C.similarity(M) + V.similarity(K));

  return TrajectoryStateOnSurface( LocalTrajectoryParameters(fsv, pzSign),
				   LocalTrajectoryError(fse), tsos.surface(),&(tsos.globalParameters().magneticField()));
}
