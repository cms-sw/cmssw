#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Surface/interface/Plane.h"
#include "Geometry/Surface/interface/Cylinder.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

std::pair<TrajectoryStateOnSurface,double> 
StraightLinePropagator::propagateWithPath(const FreeTrajectoryState& fts, 
					  const Plane& plane) const
{
  // propagate parameters
  LocalPoint x;
  LocalVector p;
  double s = 0;
  bool parametersOK = propagateParametersOnPlane(fts, plane, x, p, s);
  if(!parametersOK) return std::make_pair(TrajectoryStateOnSurface(), 0.);

  // compute propagated state
  if (fts.hasError()) {
    return std::make_pair( propagatedState(fts, plane, jacobian(s), x,  p), s);
  } else {
    // return state without errors
    return std::make_pair(TSOS(LocalTrajectoryParameters(x, p, fts.charge()), 
			  plane, theField), s);
  }
}

std::pair<TrajectoryStateOnSurface,double> 
StraightLinePropagator::propagateWithPath(const FreeTrajectoryState& fts, 
					  const Cylinder& cylinder) const
{
  // propagate parameters
  GlobalPoint x;
  GlobalVector p;
  double s = 0;
  bool parametersOK = propagateParametersOnCylinder(fts, cylinder,  x, p, s);
  if(!parametersOK) return std::make_pair(TrajectoryStateOnSurface(), 0.);

  // compute propagated state
  if (fts.hasError()) {
    return std::make_pair( propagatedState(fts, cylinder, jacobian(s), x,  p), s);
  } else {
    // return state without errors
    return std::make_pair(TSOS(GlobalTrajectoryParameters(x, p, fts.charge(),theField),
			  cylinder), s);
  }
}

TrajectoryStateOnSurface 
StraightLinePropagator::propagatedState(const FTS& fts,
					const Surface& surface,
					const AlgebraicMatrix& jacobian,
					const LocalPoint& x, 
					const LocalVector& p) const {
  if(fts.hasError()) {
    // propagate error
    TSOS tmp( fts, surface);
    AlgebraicSymMatrix eLocal(tmp.localError().matrix());
    AlgebraicSymMatrix lte = eLocal.similarity(jacobian);
    LocalTrajectoryError eloc(lte);
    LocalTrajectoryParameters ltp(x, p, fts.charge());
    return TSOS(ltp, eloc, surface, theField);
  } else {
    // return state without errors
    return TSOS(LocalTrajectoryParameters(x, p, fts.charge()), surface, theField);
  }
}

TrajectoryStateOnSurface 
StraightLinePropagator::propagatedState(const FTS& fts,
					const Surface& surface,
					const AlgebraicMatrix& jacobian,
					const GlobalPoint& x, 
					const GlobalVector& p) const {

  if(fts.hasError()) {
    // propagate error
    TSOS tmp(fts, surface);
    AlgebraicSymMatrix eLocal(tmp.localError().matrix());
    AlgebraicSymMatrix lte = eLocal.similarity(jacobian);
    LocalTrajectoryError eloc(lte);

    TSOS tmp2(tmp.localParameters(), eloc, surface, theField);
    GlobalTrajectoryParameters gtp(x, p, fts.charge(), theField);
    return TSOS(gtp, tmp2.cartesianError(), surface);
  } else {
    // return state without errors
    return TSOS(GlobalTrajectoryParameters(x, p, fts.charge(), theField), surface);
  }
}

AlgebraicMatrix StraightLinePropagator::jacobian(double& s) const {
  //Jacobian for 5*5 local error matrix
  AlgebraicMatrix F(5,5,1);//Jacobian
  
  double dir = (propagationDirection() == alongMomentum) ? 1. : -1.;
  if (s*dir < 0.) return F;

  F(4,2) = s; 
  F(5,3) = s; 

  return F;
}

bool StraightLinePropagator::propagateParametersOnCylinder(const FTS& fts, 
						const Cylinder& cylinder, 
							   GlobalPoint& x, 
							   GlobalVector& p, 
							   double& s) const {
  GlobalPoint sp = cylinder.toGlobal(LocalPoint(0., 0.));
  if (sp.x()!=0. || sp.y()!=0.) {
    throw PropagationException("Cannot propagate to an arbitrary cylinder");
  }

  x = fts.position();
  p = fts.momentum();
  s = cylinder.radius() - x.perp();

  double dir = (propagationDirection() == alongMomentum) ? 1. : -1.;
  if(s*dir < 0.) return false;

  AlgebraicVector x_k1(3);//extrapolate position
  x_k1(1) = x.x() + (p.x()/p.perp())*s; 
  x_k1(2) = x.y() + (p.y()/p.perp())*s; 
  x_k1(3) = x.z() + (p.z()/p.perp())*s;
  
  x = GlobalPoint(x_k1(1), x_k1(2), x_k1(3));    

  return true;
}
  
bool StraightLinePropagator::propagateParametersOnPlane(const FTS& fts, 
							const Plane& plane, 
							LocalPoint& x, 
							LocalVector& p, 
							double& s) const {
  
  //Do extrapolation in local frame of plane
  //  LocalPoint sp = plane.toLocal(plane.position());
  x = plane.toLocal(fts.position());
  p = plane.toLocal(fts.momentum());
  s = -x.z(); // sp.z() - x.z(); local z of plane always 0
  
  //double dir = (propagationDirection() == alongMomentum) ? 1. : -1.;
  //if(s*dir < 0.) return false;

  AlgebraicVector x_k1(3);//extrapolate position
  x_k1(1) = x.x() + (p.x()/p.z())*s; 
  x_k1(2) = x.y() + (p.y()/p.z())*s; 
  x_k1(3) = x.z() + s;
  
  x = LocalPoint(x_k1(1), x_k1(2), x_k1(3));    

  return true;
}





