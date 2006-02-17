#include "TrackingTools/KalmanUpdators/interface/Strip1DMeasurementTransformator.h"
//#include "CommonDet/BasicDet/interface/Det.h"
//#include "CommonDet/BasicDet/interface/DetUnit.h"
//#include "CommonDet/DetGeometry/interface/BoundPlane.h"

Strip1DMeasurementTransformator::Strip1DMeasurementTransformator(const TSOS& tsos,
							 const TransientTrackingRecHit& hit) : 
  theRecHit(hit),
  theState(tsos),
  theTopology(0) {

  init();
}

void Strip1DMeasurementTransformator::init() {

  theTopology = 
    dynamic_cast<const StripTopology*>(&(hit().detUnit()->topology()));
}

double Strip1DMeasurementTransformator::hitParameters() const {
  
  return topology()->measurementPosition(hit().localPosition()).x();
}

AlgebraicVector Strip1DMeasurementTransformator::trajectoryParameters() const {
    
  return state().localParameters().vector();
}

double Strip1DMeasurementTransformator::projectedTrajectoryParameters() const {

  return topology()->measurementPosition(state().localPosition()).x();
}

double Strip1DMeasurementTransformator::hitError() const {

  return     
    topology()->measurementError(hit().localPosition(),
				 hit().localPositionError()).uu();
}

AlgebraicSymMatrix Strip1DMeasurementTransformator::trajectoryError() const {

  return state().localError().matrix();
}

double Strip1DMeasurementTransformator::projectedTrajectoryError() const {

  return 
    topology()->measurementError(state().localPosition(),
				 state().localError().positionError()).uu();
}

AlgebraicMatrix Strip1DMeasurementTransformator::projectionMatrix() const {

  //  H(measurement <- local)
  //  m_meas = H*x_local + c
  AlgebraicMatrix H(1,5,0);

  double phi = 
    topology()->stripAngle(topology()->strip(state().localPosition()));
  double pitch = topology()->localPitch(state().localPosition());
  H(1,4) = cos(phi)/pitch; H(1,5) = sin(phi)/pitch;
  
  return H;
}






