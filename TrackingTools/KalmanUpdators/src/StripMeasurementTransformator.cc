#include "TrackingTools/KalmanUpdators/interface/StripMeasurementTransformator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

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

AlgebraicVector2 StripMeasurementTransformator::hitParameters() const {
  
  AlgebraicVector2 av;
  MeasurementPoint mp = 
    topology()->measurementPosition(hit().localPosition());
  av[0] = mp.x();
  av[1] = mp.y();

  return av;
}

AlgebraicVector5 StripMeasurementTransformator::trajectoryParameters() const {
    
  return state().localParameters().vector();
}

AlgebraicVector2 
StripMeasurementTransformator::projectedTrajectoryParameters() const {

  AlgebraicVector2 av;
  MeasurementPoint mp = 
    topology()->measurementPosition(state().localPosition());
  av[0] = mp.x();
  av[1] = mp.y();

  return av;
}

AlgebraicSymMatrix22 StripMeasurementTransformator::hitError() const {

  AlgebraicSymMatrix22 am;
   MeasurementError me = 
    topology()->measurementError(hit().localPosition(),
				 hit().localPositionError());
  am(0,0) = me.uu();
  am(1,0) = me.uv();
  am(1,1) = me.vv();
  
  return am;
}

const AlgebraicSymMatrix55 & StripMeasurementTransformator::trajectoryError() const {

  return state().localError().matrix();
}

AlgebraicSymMatrix22 
StripMeasurementTransformator::projectedTrajectoryError() const {

  AlgebraicSymMatrix22 am;
  MeasurementError me = 
    topology()->measurementError(state().localPosition(),
				 state().localError().positionError());
  am(0,0) = me.uu();
  am(1,0) = me.uv();
  am(1,1) = me.vv();

  return am;
}

AlgebraicMatrix25 StripMeasurementTransformator::projectionMatrix() const {

  // H(measurement <- local)
  // m_meas = H*x_local + c
  AlgebraicMatrix25 H;
  
  float phi = 
    topology()->stripAngle(topology()->strip(state().localPosition()));
  float pitch = topology()->localPitch(state().localPosition());
  float length = topology()->localStripLength(state().localPosition());
  H(0,3) = cos(phi)/pitch; H(0,4) = sin(phi)/pitch;
  H(1,3) = -sin(phi)/length; H(1,4) = cos(phi)/length;

  return H;
}

