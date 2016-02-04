#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/StraightLineBarrelCylinderCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>

using namespace SurfaceSideDefinition;

std::pair<TrajectoryStateOnSurface,double>
AnalyticalPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
					const Plane& plane) const
{
  // check curvature
  double rho = fts.transverseCurvature();
  
  // propagate parameters
  GlobalPoint x;
  GlobalVector p;
  double s;
  
  // check if already on plane
  const float maxDistToPlane(0.1e-4);
  const float numericalPrecision(5.e-7);
  float maxDz = numericalPrecision*plane.position().mag();
  if ( fabs(plane.localZ(fts.position()))>(maxDistToPlane>maxDz?maxDistToPlane:maxDz) ) {
    // propagate
    bool parametersOK = this->propagateParametersOnPlane(fts, plane, x, p, s);
    // check status and deltaPhi limit
    float dphi2 = s*rho;
    dphi2 = dphi2*dphi2*fts.momentum().perp2()/fts.momentum().mag2();
    if ( !parametersOK || dphi2>theMaxDPhi2 )  return TsosWP(TrajectoryStateOnSurface(),0.);
  }
  else {
    LogDebug("AnalyticalPropagator")<<"not going anywhere. Already on surface.\n"
				    <<"plane.localZ(fts.position()): "<<plane.localZ(fts.position())<<"\n"
				    <<"maxDistToPlane: "<<maxDistToPlane<<"\n"
				    <<"maxDz: "<<maxDz<<"\n"
				    <<"plane.position().mag(): "<<plane.position().mag();
    x = fts.position();
    p = fts.momentum();
    s = 0.;
  }
  //
  // Compute propagated state and check change in curvature
  //
  GlobalTrajectoryParameters gtp(x,p,fts.charge(),theField);
  if ( fabs(rho)>1.e-10 && fabs((gtp.transverseCurvature()-rho)/rho)>theMaxDBzRatio ) 
    return TsosWP(TrajectoryStateOnSurface(),0.);
  //
  // construct TrajectoryStateOnSurface
  //
  return propagatedStateWithPath(fts,plane,gtp,s);
}


std::pair<TrajectoryStateOnSurface,double>
AnalyticalPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
					const Cylinder& cylinder) const
{
  // check curvature
  double rho = fts.transverseCurvature();

  // propagate parameters
  GlobalPoint x;
  GlobalVector p;
  double s = 0;

  bool parametersOK = this->propagateParametersOnCylinder(fts, cylinder, x, p, s);
  // check status and deltaPhi limit
  float dphi2 = s*rho;
  dphi2 = dphi2*dphi2*fts.momentum().perp2()/fts.momentum().mag2();
  if ( !parametersOK || dphi2>theMaxDPhi2 )  return TsosWP(TrajectoryStateOnSurface(),0.);
  //
  // Compute propagated state and check change in curvature
  //
  GlobalTrajectoryParameters gtp(x,p,fts.charge(),theField);
  if ( fabs(rho)>1.e-10 && fabs((gtp.transverseCurvature()-rho)/rho)>theMaxDBzRatio ) 
    return TsosWP(TrajectoryStateOnSurface(),0.);
  //
  // create result TSOS on TangentPlane (local parameters & errors are better defined)
  //
  ReferenceCountingPointer<TangentPlane> plane(cylinder.tangentPlane(x));  // need to be here until tsos is created!
  return propagatedStateWithPath(fts,*plane,gtp,s);
}

std::pair<TrajectoryStateOnSurface,double>
AnalyticalPropagator::propagatedStateWithPath (const FreeTrajectoryState& fts, 
					       const Surface& surface, 
					       const GlobalTrajectoryParameters& gtp, 
					       const double& s) const
{
  //
  // for forward propagation: state is before surface,
  // for backward propagation: state is after surface
  //
  SurfaceSide side = PropagationDirectionFromPath()(s,propagationDirection())==alongMomentum 
    ? beforeSurface : afterSurface;
  // 
  //
  // error propagation (if needed) and conversion to a TrajectoryStateOnSurface
  //
  if (fts.hasError()) {
    //
    // compute jacobian
    //
    AnalyticalCurvilinearJacobian analyticalJacobian(fts.parameters(), gtp.position(), gtp.momentum(), s);
    const AlgebraicMatrix55 &jacobian = analyticalJacobian.jacobian();
    CurvilinearTrajectoryError cte(ROOT::Math::Similarity(jacobian, fts.curvilinearError().matrix()));
    return TsosWP(TrajectoryStateOnSurface(gtp,cte,surface,side),s);
  }
  else {
    //
    // return state without errors
    //
    return TsosWP(TrajectoryStateOnSurface(gtp,surface,side),s);
  }
}

bool AnalyticalPropagator::propagateParametersOnCylinder(
  const FreeTrajectoryState& fts, const Cylinder& cylinder, 
  GlobalPoint& x, GlobalVector& p, double& s) const
{

  GlobalPoint const & sp = cylinder.position();
  if (sp.x()!=0. || sp.y()!=0.) {
    throw PropagationException("Cannot propagate to an arbitrary cylinder");
  }
  // preset output
  x = fts.position();
  p = fts.momentum();
  s = 0;
  // (transverse) curvature
  double rho = fts.transverseCurvature();
  //
  // Straight line approximation? |rho|<1.e-10 equivalent to ~ 1um 
  // difference in transversal position at 10m.
  //
  if( fabs(rho)<1.e-10 )
    return propagateWithLineCrossing(fts.position(),p,cylinder,x,s);
  //
  // Helix case
  //
  // check for possible intersection
  const double tolerance = 1.e-4; // 1 micron distance
  double rdiff = x.perp() - cylinder.radius();
  if ( fabs(rdiff) < tolerance )  return true;
  //
  // Instantiate HelixBarrelCylinderCrossing and get solutions
  //
  HelixBarrelCylinderCrossing cylinderCrossing(fts.position(),fts.momentum(),rho,
					       propagationDirection(),cylinder);
  if ( !cylinderCrossing.hasSolution() )  return false;
  // path length
  s = cylinderCrossing.pathLength();
  // point
  x = cylinderCrossing.position();
  // direction (renormalised)
  p = cylinderCrossing.direction().unit()*fts.momentum().mag();
  return true;
}
  
bool 
AnalyticalPropagator::propagateParametersOnPlane(const FreeTrajectoryState& fts, 
						 const Plane& plane, 
						 GlobalPoint& x, 
						 GlobalVector& p, 
						 double& s) const
{
  // initialisation of position, momentum and path length
  x = fts.position();
  p = fts.momentum();
  s = 0;
  // (transverse) curvature
  double rho = fts.transverseCurvature();
  //
  // Straight line approximation? |rho|<1.e-10 equivalent to ~ 1um 
  // difference in transversal position at 10m.
  //
  if( fabs(rho)<1.e-10 )
    return propagateWithLineCrossing(fts.position(),p,plane,x,s);
  //
  // Helix case 
  //
  GlobalVector u = plane.normalVector();
  const double small = 1.e-6; // for orientation of planes
  //
  // Frame-independant point and vector are created explicitely to 
  // avoid confusing gcc (refuses to compile with temporary objects
  // in the constructor).
  //
  HelixPlaneCrossing::PositionType helixPos(x);
  HelixPlaneCrossing::DirectionType helixDir(p);
  if (fabs(u.z()) < small) {
    // barrel plane:
    // instantiate HelixBarrelPlaneCrossing, get vector of solutions and check for existance
    HelixBarrelPlaneCrossingByCircle planeCrossing(helixPos,helixDir,rho,propagationDirection());
    return propagateWithHelixCrossing(planeCrossing,plane,fts.momentum().mag(),x,p,s);
  }
  if (fabs(u.x()) < small && fabs(u.y()) < small) {
    // forward plane:
    // instantiate HelixForwardPlaneCrossing, get vector of solutions and check for existance
    HelixForwardPlaneCrossing planeCrossing(helixPos,helixDir,rho,propagationDirection());
    return propagateWithHelixCrossing(planeCrossing,plane,fts.momentum().mag(),x,p,s);
  }
  else {
    // arbitrary plane:
    // instantiate HelixArbitraryPlaneCrossing, get vector of solutions and check for existance
    HelixArbitraryPlaneCrossing planeCrossing(helixPos,helixDir,rho,propagationDirection());
    return propagateWithHelixCrossing(planeCrossing,plane,fts.momentum().mag(),x,p,s);
  }
}

bool
AnalyticalPropagator::propagateWithLineCrossing (const GlobalPoint& x0, 
						 const GlobalVector& p0,
						 const Plane& plane,
						 GlobalPoint& x, double& s) const {
  //
  // Instantiate auxiliary object for finding intersection.
  // Frame-independant point and vector are created explicitely to 
  // avoid confusing gcc (refuses to compile with temporary objects
  // in the constructor).
  //
  StraightLinePlaneCrossing::PositionType pos(x0);
  StraightLinePlaneCrossing::DirectionType dir(p0);
  StraightLinePlaneCrossing planeCrossing(pos,dir,propagationDirection());
  //
  // get solution
  //
  std::pair<bool,double> propResult = planeCrossing.pathLength(plane);
  if ( !propResult.first )  return false;
  s = propResult.second;
  // point (reconverted to GlobalPoint)
  x = GlobalPoint(planeCrossing.position(s));
  //
  return true;
}

bool
AnalyticalPropagator::propagateWithLineCrossing (const GlobalPoint& x0, 
						 const GlobalVector& p0,
						 const Cylinder& cylinder,
						 GlobalPoint& x, double& s) const {
  //
  // Instantiate auxiliary object for finding intersection.
  // Frame-independant point and vector are created explicitely to 
  // avoid confusing gcc (refuses to compile with temporary objects
  // in the constructor).
  //
  StraightLineBarrelCylinderCrossing cylCrossing(x0,p0,propagationDirection());
  //
  // get solution
  //
  std::pair<bool,double> propResult = cylCrossing.pathLength(cylinder);
  if ( !propResult.first )  return false;
  s = propResult.second;
  // point (reconverted to GlobalPoint)
  x = cylCrossing.position(s);
  //
  return true;
}

bool
AnalyticalPropagator::propagateWithHelixCrossing (HelixPlaneCrossing& planeCrossing,
						  const Plane& plane,
						  const float pmag,
						  GlobalPoint& x,
						  GlobalVector& p,
						  double& s) const {
  // get solution
  std::pair<bool,double> propResult = planeCrossing.pathLength(plane);
  if ( !propResult.first )  return false;
  s = propResult.second;
  // point (reconverted to GlobalPoint)
  HelixPlaneCrossing::PositionType xGen = planeCrossing.position(s);
  x = GlobalPoint(xGen);
  // direction (reconverted to GlobalVector, renormalised)
  HelixPlaneCrossing::DirectionType pGen = planeCrossing.direction(s);
  pGen *= pmag/pGen.mag();
  p = GlobalVector(pGen);
  //
  return true;
}
