#include "TrackingTools/KalmanUpdators/interface/KFStripUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/StripMeasurementTransformator.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"


TrajectoryStateOnSurface 
KFStripUpdator::update(const TSOS& aTsos, const TransientTrackingRecHit& aHit) const {

  double pzSign = aTsos.localParameters().pzSign();

  StripMeasurementTransformator myTrafo(aTsos, aHit);

  AlgebraicMatrix25 H(myTrafo.projectionMatrix());
  AlgebraicVector2 m(myTrafo.hitParameters());
  AlgebraicVector5 x(myTrafo.trajectoryParameters());
  AlgebraicVector2 px(myTrafo.projectedTrajectoryParameters());
  //  AlgebraicVector px = H*x;
  
  AlgebraicSymMatrix22 V(myTrafo.hitError());
  const AlgebraicSymMatrix55 &C = myTrafo.trajectoryError();
  AlgebraicSymMatrix22 pC(myTrafo.projectedTrajectoryError());
  //  AlgebraicSymMatrix pC = C.similarity(H);

  AlgebraicSymMatrix22 R(V + pC);
  //int ierr; R.invert(ierr); // if (ierr != 0) throw exception;
  invertPosDefMatrix(R);
  
  // Compute Kalman gain matrix
  //  AlgebraicMatrix Hm2l(myTrafo.measurement2LocalProj());
  AlgebraicMatrix52 K(C * ROOT::Math::Transpose(H) * R);

  // Compute local filtered state vector
  AlgebraicVector5 fsv(x + K * (m - px));

  // Compute covariance matrix of local filtered state vector
  AlgebraicMatrix55 I = AlgebraicMatrixID();
  AlgebraicMatrix55 M = (I - K * H);
  AlgebraicSymMatrix55 fse = ROOT::Math::Similarity(M,C) + ROOT::Math::Similarity(K,V);

  return TSOS( LTP(fsv, pzSign), LTE(fse), aTsos.surface(),&(aTsos.globalParameters().magneticField()));  
}




