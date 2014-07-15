#include "TrackingTools/KalmanUpdators/interface/Strip1DMeasurementTransformator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
// #include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"

Strip1DMeasurementTransformator::Strip1DMeasurementTransformator(const TSOS& tsos,
							 const   TrackingRecHit& hit) : 
  theRecHit(hit),
  theState(tsos),
  theTopology(0) {

  init();
}

void Strip1DMeasurementTransformator::init() {

  theTopology =  dynamic_cast<const StripTopology*>(&(hit().detUnit()->topology()));
  theIdealTopology =  dynamic_cast<const StripTopology*>(&(hit().detUnit()->type().topology()));
}

double Strip1DMeasurementTransformator::hitParameters() const {
  
  return topology()->measurementPosition(hit().localPosition()).x();
}

AlgebraicVector5 Strip1DMeasurementTransformator::trajectoryParameters() const {
    
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

const AlgebraicSymMatrix55 & Strip1DMeasurementTransformator::trajectoryError() const {

  return state().localError().matrix();
}

double Strip1DMeasurementTransformator::projectedTrajectoryError() const {

  return 
    topology()->measurementError(state().localPosition(),
				 state().localError().positionError()).uu();
}

AlgebraicMatrix15 Strip1DMeasurementTransformator::projectionMatrix() const {

  //  H(measurement <- local)
  //  m_meas = H*x_local + c
  AlgebraicMatrix15 H;
  if(const RadialStripTopology* tmp = dynamic_cast<const RadialStripTopology*>(idealTopology())) {
    double yHitToInter = tmp->yDistanceToIntersection( hit().localPosition().y() );
    double t  = tmp->yAxisOrientation() * hit().localPosition().x() / yHitToInter;
    double c2 = 1./(1. + t*t);  // cos(angle)**2
    //double cs = t*c2;           // sin(angle)*cos(angle); tan carries sign of sin!
    double s2 = 1. - c2;        // sin(angle)**2
    double A  = tmp->angularWidth();
    // D is distance from intersection of edges to hit on strip
    double D2 = hit().localPosition().x()*hit().localPosition().x() + yHitToInter*yHitToInter;
    double D = std::sqrt(D2);
    
    double cp = std::sqrt(c2);
    double sp;
    if(t > 0) {
      sp = std::sqrt(s2);
    } else {
      sp = -std::sqrt(s2);
    }
    H(0,3) = cp/(D*A); H(0,4) = -sp/(D*A);
  } else {
    double phi = 
      topology()->stripAngle(topology()->strip(state().localPosition()));
    double pitch = topology()->localPitch(state().localPosition());
    H(0,3) = cos(phi)/pitch; H(0,4) = sin(phi)/pitch;
  }
  return H;
}






