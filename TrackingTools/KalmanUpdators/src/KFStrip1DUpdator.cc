#include "TrackingTools/KalmanUpdators/interface/KFStrip1DUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Strip1DMeasurementTransformator.h"

TrajectoryStateOnSurface 
KFStrip1DUpdator::update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const {

  double pzSign = aTsos.localParameters().pzSign();

  Strip1DMeasurementTransformator myTrafo(aTsos, aHit);

  double m = myTrafo.hitParameters();
  AlgebraicVector x(myTrafo.trajectoryParameters());
  double px = myTrafo.projectedTrajectoryParameters();
  
  AlgebraicMatrix H(myTrafo.projectionMatrix());
  double V = myTrafo.hitError();
  AlgebraicSymMatrix C(myTrafo.trajectoryError());
  double pC = myTrafo.projectedTrajectoryError();

  double R = 1./(V + pC);
  
  // Compute Kalman gain matrix
  AlgebraicMatrix K(R * (C * H.T()));

  // Compute local filtered state vector
  AlgebraicVector fsv(x + (m - px) * K);

  // Compute covariance matrix of local filtered state vector
  AlgebraicSymMatrix I(5, 1);
  AlgebraicMatrix M(I - K * H);
  AlgebraicSymMatrix fse(C.similarity(M) + V * vT_times_v(K));
//   AlgebraicMatrix M((I - K * H)*C);
//   AlgebraicSymMatrix fse(5,0); fse.assign(M);

  return TSOS( LTP(fsv, pzSign), LTE(fse), aTsos.surface(), &(aTsos.globalParameters().magneticField()));  
}





