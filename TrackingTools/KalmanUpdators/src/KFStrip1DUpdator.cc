#include "TrackingTools/KalmanUpdators/interface/KFStrip1DUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Strip1DMeasurementTransformator.h"

TrajectoryStateOnSurface 
KFStrip1DUpdator::update(const TSOS& aTsos, const TrackingRecHit& aHit) const {

  double pzSign = aTsos.localParameters().pzSign();

  Strip1DMeasurementTransformator myTrafo(aTsos, aHit);

  double m = myTrafo.hitParameters();
  AlgebraicVector5 x(myTrafo.trajectoryParameters());
  double px = myTrafo.projectedTrajectoryParameters();
  
  AlgebraicMatrix15 H(myTrafo.projectionMatrix());
  double V = myTrafo.hitError();
  AlgebraicSymMatrix55 C(myTrafo.trajectoryError());
  double pC = myTrafo.projectedTrajectoryError();

  double R = 1./(V + pC);
  
  // Compute Kalman gain matrix
  AlgebraicMatrix51 K(R * (C * ROOT::Math::Transpose(H)));

  // Compute local filtered state vector
  AlgebraicVector5 fsv = x + K.Col(0) * (m - px);

  // Compute covariance matrix of local filtered state vector
  AlgebraicSymMatrix55 I = AlgebraicMatrixID();
  AlgebraicMatrix55 M = I - K * H;
  AlgebraicSymMatrix55 fse = ROOT::Math::Similarity(M, C) +  ROOT::Math::Similarity(K, AlgebraicSymMatrix11(V) );
//   AlgebraicMatrix M((I - K * H)*C);            // already commented when CLHEP was in use
//   AlgebraicSymMatrix fse(5,0); fse.assign(M);  // already commented when CLHEP was in use

  return TSOS( LTP(fsv, pzSign), LTE(fse), aTsos.surface(), &(aTsos.globalParameters().magneticField()), aTsos.surfaceSide() );  
}





