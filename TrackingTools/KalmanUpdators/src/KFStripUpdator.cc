#include "TrackingTools/KalmanUpdators/interface/KFStripUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/StripMeasurementTransformator.h"

TrajectoryStateOnSurface 
KFStripUpdator::update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const {

  double pzSign = aTsos.localParameters().pzSign();

  StripMeasurementTransformator myTrafo(aTsos, aHit);

  AlgebraicMatrix H(myTrafo.projectionMatrix());
  AlgebraicVector m(myTrafo.hitParameters());
  AlgebraicVector x(myTrafo.trajectoryParameters());
  AlgebraicVector px(myTrafo.projectedTrajectoryParameters());
  //  AlgebraicVector px = H*x;
  
  AlgebraicSymMatrix V(myTrafo.hitError());
  AlgebraicSymMatrix C(myTrafo.trajectoryError());
  AlgebraicSymMatrix pC(myTrafo.projectedTrajectoryError());
  //  AlgebraicSymMatrix pC = C.similarity(H);

  AlgebraicSymMatrix R(V + pC);
  int ierr; R.invert(ierr); // if (ierr != 0) throw exception;
  
  // Compute Kalman gain matrix
  //  AlgebraicMatrix Hm2l(myTrafo.measurement2LocalProj());
  AlgebraicMatrix K(C * H.T() * R);

  // Compute local filtered state vector
  AlgebraicVector fsv(x + K * (m - px));

  // Compute covariance matrix of local filtered state vector
  AlgebraicSymMatrix I(5, 1);
  AlgebraicMatrix M(I - K * H);
  AlgebraicSymMatrix fse(C.similarity(M) + V.similarity(K));

  return TSOS( LTP(fsv, pzSign), LTE(fse), aTsos.surface(),&(aTsos.globalParameters().magneticField()));  
}




