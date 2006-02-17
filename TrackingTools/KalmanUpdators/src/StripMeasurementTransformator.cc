#include "TrackingTools/KalmanUpdators/interface/StripMeasurementTransformator.h"

StripMeasurementTransformator::StripMeasurementTransformator(const TSOS& tsos,
							 const TransientTrackingRecHit& hit) : 
  theRecHit(hit),
  theState(tsos),
  theTopology(0) {

  init();
}

void StripMeasurementTransformator::init() {

  theTopology = 
    dynamic_cast<const StripTopology*>(&(hit().detUnit()->topology()));
}

AlgebraicVector StripMeasurementTransformator::hitParameters() const {
  
  AlgebraicVector av(2,0);
  MeasurementPoint mp = 
    topology()->measurementPosition(hit().localPosition());
  av(1) = mp.x();
  av(2) = mp.y();

  return av;
}

AlgebraicVector StripMeasurementTransformator::trajectoryParameters() const {
    
  return state().localParameters().vector();
}

AlgebraicVector 
StripMeasurementTransformator::projectedTrajectoryParameters() const {

  AlgebraicVector av(2,0);
  MeasurementPoint mp = 
    topology()->measurementPosition(state().localPosition());
  av(1) = mp.x();
  av(2) = mp.y();

  return av;
}

AlgebraicSymMatrix StripMeasurementTransformator::hitError() const {

  AlgebraicSymMatrix am(2,0);
   MeasurementError me = 
    topology()->measurementError(hit().localPosition(),
				 hit().localPositionError());
  am(1,1) = me.uu();
  am(2,1) = me.uv();
  am(2,2) = me.vv();
  
  return am;
}

AlgebraicSymMatrix StripMeasurementTransformator::trajectoryError() const {

  return state().localError().matrix();
}

AlgebraicSymMatrix 
StripMeasurementTransformator::projectedTrajectoryError() const {

  AlgebraicSymMatrix am(2,0);
  MeasurementError me = 
    topology()->measurementError(state().localPosition(),
				 state().localError().positionError());
  am(1,1) = me.uu();
  am(2,1) = me.uv();
  am(2,2) = me.vv();

  return am;
}

AlgebraicMatrix StripMeasurementTransformator::projectionMatrix() const {

  // H(measurement <- local)
  // m_meas = H*x_local + c
  AlgebraicMatrix H(2,5,0);
  
  float phi = 
    topology()->stripAngle(topology()->strip(state().localPosition()));
  float pitch = topology()->localPitch(state().localPosition());
  float length = topology()->localStripLength(state().localPosition());
  H(1,4) = cos(phi)/pitch; H(1,5) = sin(phi)/pitch;
  H(2,4) = -sin(phi)/length; H(2,5) = cos(phi)/length;

  return H;
}

