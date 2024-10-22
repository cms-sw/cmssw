#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/StraightLineBarrelCylinderCrossing.h"
#include "TrackingTools/GeomPropagators/interface/OptimalHelixPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Likely.h"

#include <cmath>

using namespace SurfaceSideDefinition;

std::pair<TrajectoryStateOnSurface, double> AnalyticalPropagator::propagateWithPath(const FreeTrajectoryState& fts,
                                                                                    const Plane& plane) const {
  // check curvature
  float rho = fts.transverseCurvature();

  // propagate parameters
  GlobalPoint x;
  GlobalVector p;
  double s;

  // check if already on plane
  if LIKELY (plane.localZclamped(fts.position()) != 0) {
    // propagate
    bool parametersOK = this->propagateParametersOnPlane(fts, plane, x, p, s);
    // check status and deltaPhi limit
    float dphi2 = float(s) * rho;
    dphi2 = dphi2 * dphi2 * fts.momentum().perp2();
    if UNLIKELY (!parametersOK || dphi2 > theMaxDPhi2 * fts.momentum().mag2())
      return TsosWP(TrajectoryStateOnSurface(), 0.);
  } else {
    LogDebug("AnalyticalPropagator") << "not going anywhere. Already on surface.\n"
                                     << "plane.localZ(fts.position()): " << plane.localZ(fts.position()) << "\n"
                                     << "plane.position().mag(): " << plane.position().mag() << "\n"
                                     << "plane.posPrec: " << plane.posPrec();
    x = fts.position();
    p = fts.momentum();
    s = 0;
  }
  //
  // Compute propagated state and check change in curvature
  //
  GlobalTrajectoryParameters gtp(x, p, fts.charge(), theField);
  if UNLIKELY (std::abs(gtp.transverseCurvature() - rho) > theMaxDBzRatio * std::abs(rho))
    return TsosWP(TrajectoryStateOnSurface(), 0.);
  //
  // construct TrajectoryStateOnSurface
  //
  return propagatedStateWithPath(fts, plane, gtp, s);
}

std::pair<TrajectoryStateOnSurface, double> AnalyticalPropagator::propagateWithPath(const FreeTrajectoryState& fts,
                                                                                    const Cylinder& cylinder) const {
  // check curvature
  auto rho = fts.transverseCurvature();

  // propagate parameters
  GlobalPoint x;
  GlobalVector p;
  double s = 0;

  bool parametersOK = this->propagateParametersOnCylinder(fts, cylinder, x, p, s);
  // check status and deltaPhi limit
  float dphi2 = s * rho;
  dphi2 = dphi2 * dphi2 * fts.momentum().perp2();
  if UNLIKELY (!parametersOK || dphi2 > theMaxDPhi2 * fts.momentum().mag2())
    return TsosWP(TrajectoryStateOnSurface(), 0.);
  //
  // Compute propagated state and check change in curvature
  //
  GlobalTrajectoryParameters gtp(x, p, fts.charge(), theField);
  if UNLIKELY (std::abs(gtp.transverseCurvature() - rho) > theMaxDBzRatio * std::abs(rho))
    return TsosWP(TrajectoryStateOnSurface(), 0.);
  //
  // create result TSOS on TangentPlane (local parameters & errors are better defined)
  //

  //try {
  ConstReferenceCountingPointer<TangentPlane> plane(
      cylinder.tangentPlane(x));  // need to be here until tsos is created!
  return propagatedStateWithPath(fts, *plane, gtp, s);
  /*
  } catch(...) {
    std::cout << "wrong tangent to cylinder " << x 
              << " pos, rad " << cylinder.position() << " " << cylinder.radius()
              << std::endl;
    return TsosWP(TrajectoryStateOnSurface(),0.);
  }
  */
}

std::pair<TrajectoryStateOnSurface, double> AnalyticalPropagator::propagatedStateWithPath(
    const FreeTrajectoryState& fts,
    const Surface& surface,
    const GlobalTrajectoryParameters& gtp,
    const double& s) const {
  //
  // for forward propagation: state is before surface,
  // for backward propagation: state is after surface
  //
  SurfaceSide side =
      PropagationDirectionFromPath()(s, propagationDirection()) == alongMomentum ? beforeSurface : afterSurface;
  //
  //
  // error propagation (if needed) and conversion to a TrajectoryStateOnSurface
  //
  if (fts.hasError()) {
    //
    // compute jacobian
    //
    AnalyticalCurvilinearJacobian analyticalJacobian(fts.parameters(), gtp.position(), gtp.momentum(), s);
    const AlgebraicMatrix55& jacobian = analyticalJacobian.jacobian();
    // CurvilinearTrajectoryError cte(ROOT::Math::Similarity(jacobian, fts.curvilinearError().matrix()));
    return TsosWP(
        TrajectoryStateOnSurface(gtp, ROOT::Math::Similarity(jacobian, fts.curvilinearError().matrix()), surface, side),
        s);
  } else {
    //
    // return state without errors
    //
    return TsosWP(TrajectoryStateOnSurface(gtp, surface, side), s);
  }
}

bool AnalyticalPropagator::propagateParametersOnCylinder(
    const FreeTrajectoryState& fts, const Cylinder& cylinder, GlobalPoint& x, GlobalVector& p, double& s) const {
  GlobalPoint const& sp = cylinder.position();
  if UNLIKELY (sp.x() != 0. || sp.y() != 0.) {
    throw PropagationException("Cannot propagate to an arbitrary cylinder");
  }
  // preset output
  x = fts.position();
  p = fts.momentum();
  s = 0;
  // (transverse) curvature
  auto rho = fts.transverseCurvature();
  //
  // Straight line approximation? |rho|<1.e-10 equivalent to ~ 1um
  // difference in transversal position at 10m.
  //
  if UNLIKELY (std::abs(rho) < 1.e-10f)
    return propagateWithLineCrossing(fts.position(), p, cylinder, x, s);
  //
  // Helix case
  //
  // check for possible intersection
  constexpr float tolerance = 1.e-4;  // 1 micron distance
  auto rdiff = x.perp() - cylinder.radius();
  if (std::abs(rdiff) < tolerance)
    return true;
  //
  // Instantiate HelixBarrelCylinderCrossing and get solutions
  //
  HelixBarrelCylinderCrossing cylinderCrossing(fts.position(), fts.momentum(), rho, propagationDirection(), cylinder);
  if UNLIKELY (!cylinderCrossing.hasSolution())
    return false;
  // path length
  s = cylinderCrossing.pathLength();
  // point
  x = cylinderCrossing.position();
  // direction (renormalised)
  p = cylinderCrossing.direction().unit() * fts.momentum().mag();
  return true;
}

bool AnalyticalPropagator::propagateParametersOnPlane(
    const FreeTrajectoryState& fts, const Plane& plane, GlobalPoint& x, GlobalVector& p, double& s) const {
  // initialisation of position, momentum and path length
  x = fts.position();
  p = fts.momentum();
  s = 0;
  // (transverse) curvature
  auto rho = fts.transverseCurvature();
  //
  // Straight line approximation? |rho|<1.e-10 equivalent to ~ 1um
  // difference in transversal position at 10m.
  //
  if UNLIKELY (std::abs(rho) < 1.e-10f)
    return propagateWithLineCrossing(fts.position(), p, plane, x, s);
  //
  // Helix case
  //

  //
  // Frame-independant point and vector are created explicitely to
  // avoid confusing gcc (refuses to compile with temporary objects
  // in the constructor).
  //
  HelixPlaneCrossing::PositionType helixPos(x);
  HelixPlaneCrossing::DirectionType helixDir(p);
  if LIKELY (isOldPropagationType) {
    OptimalHelixPlaneCrossing planeCrossing(plane, helixPos, helixDir, rho, propagationDirection());
    return propagateWithHelixCrossing(*planeCrossing, plane, fts.momentum().mag(), x, p, s);
  }

  //--- Alternative implementation to be used for the propagation of the parameters  of looping
  //    particles that cross twice the (infinite) surface of the plane. It is not trivial to determine
  //    which of the two intersections has to be returned.

  //---- FIXME: WHAT FOLLOWS HAS TO BE REWRITTEN IN A CLEANER (AND CPU-OPTIMIZED) WAY ---------
  LogDebug("AnalyticalPropagator") << "In AnaliticalProp, calling HAPC "
                                   << "\n"
                                   << "plane is centered in xyz: " << plane.position().x() << " , "
                                   << plane.position().y() << " , " << plane.position().z() << "\n";

  GlobalPoint gp1 = fts.position();
  GlobalVector gm1 = fts.momentum();
  double s1 = 0;
  double rho1 = fts.transverseCurvature();
  HelixPlaneCrossing::PositionType helixPos1(gp1);
  HelixPlaneCrossing::DirectionType helixDir1(gm1);
  LogDebug("AnalyticalPropagator") << "gp1 before calling planeCrossing1: " << gp1 << "\n";
  OptimalHelixPlaneCrossing planeCrossing1(plane, helixPos1, helixDir1, rho1, propagationDirection());

  HelixPlaneCrossing::PositionType xGen;
  HelixPlaneCrossing::DirectionType pGen;

  double tolerance(0.0050);
  if (propagationDirection() == oppositeToMomentum)
    tolerance *= -1;

  bool check1 = propagateWithHelixCrossing(*planeCrossing1, plane, fts.momentum().mag(), gp1, gm1, s1);
  double dphi1 = fabs(fts.momentum().phi() - gm1.phi());
  LogDebug("AnalyticalPropagator") << "check1, s1, dphi, gp1: " << check1 << " , " << s1 << " , " << dphi1 << " , "
                                   << gp1 << "\n";

  //move forward a bit to avoid that the propagator doesn't propagate because the state is already on surface.
  //we want to go to the other point of intersection between the helix and the plane
  xGen = (*planeCrossing1).position(s1 + tolerance);
  pGen = (*planeCrossing1).direction(s1 + tolerance);

  /*
    if(!check1 || s1>170 ){
    //PropagationDirection newDir = (propagationDirection() == alongMomentum) ? oppositeToMomentum : alongMomentum;
    PropagationDirection newDir = anyDirection;
    HelixArbitraryPlaneCrossing  planeCrossing1B(helixPos1,helixDir1,rho1,newDir);
    check1 = propagateWithHelixCrossing(planeCrossing1B,plane,fts.momentum().mag(),gp1,gm1,s1);
    LogDebug("AnalyticalPropagator") << "after second attempt, check1, s1,gp1: "
    << check1 << " , "
    << s1 << " , " << gp1 << "\n";
    
    xGen = planeCrossing1B.position(s1+tolerance);
    pGen = planeCrossing1B.direction(s1+tolerance);
    }
      */

  if (!check1) {
    LogDebug("AnalyticalPropagator") << "failed also second attempt. No idea what to do, then bailout"
                                     << "\n";
  }

  pGen *= gm1.mag() / pGen.mag();
  GlobalPoint gp2(xGen);
  GlobalVector gm2(pGen);
  double s2 = 0;
  double rho2 = rho1;
  HelixPlaneCrossing::PositionType helixPos2(gp2);
  HelixPlaneCrossing::DirectionType helixDir2(gm2);
  OptimalHelixPlaneCrossing planeCrossing2(plane, helixPos2, helixDir2, rho2, propagationDirection());

  bool check2 = propagateWithHelixCrossing(*planeCrossing2, plane, gm2.mag(), gp2, gm2, s2);

  if (!check2) {
    x = gp1;
    p = gm1;
    s = s1;
    return check1;
  }

  if (!check1) {
    edm::LogError("AnalyticalPropagator") << "LOGIC ERROR: I should not have entered here!"
                                          << "\n";
    return false;
  }

  LogDebug("AnalyticalPropagator") << "check2, s2, gp2: " << check2 << " , " << s2 << " , " << gp2 << "\n";

  double dist1 = (plane.position() - gp1).perp();
  double dist2 = (plane.position() - gp2).perp();

  LogDebug("AnalyticalPropagator") << "propDir, dist1, dist2: " << propagationDirection() << " , " << dist1 << " , "
                                   << dist2 << "\n";

  //If there are two solutions, the one which is the closest to the module's center is chosen
  if (dist1 < 2 * dist2) {
    x = gp1;
    p = gm1;
    s = s1;
    return check1;
  } else if (dist2 < 2 * dist1) {
    x = gp2;
    p = gm2;
    s = s1 + s2 + tolerance;
    return check2;
  } else {
    if (fabs(s1) < fabs(s2)) {
      x = gp1;
      p = gm1;
      s = s1;
      return check1;
    } else {
      x = gp2;
      p = gm2;
      s = s1 + s2 + tolerance;
      return check2;
    }
  }

  //-------- END of ugly piece of code  ---------------
}

bool AnalyticalPropagator::propagateWithLineCrossing(
    const GlobalPoint& x0, const GlobalVector& p0, const Plane& plane, GlobalPoint& x, double& s) const {
  //
  // Instantiate auxiliary object for finding intersection.
  // Frame-independant point and vector are created explicitely to
  // avoid confusing gcc (refuses to compile with temporary objects
  // in the constructor).
  //
  StraightLinePlaneCrossing::PositionType pos(x0);
  StraightLinePlaneCrossing::DirectionType dir(p0);
  StraightLinePlaneCrossing planeCrossing(pos, dir, propagationDirection());
  //
  // get solution
  //
  std::pair<bool, double> propResult = planeCrossing.pathLength(plane);
  if (!propResult.first)
    return false;
  s = propResult.second;
  // point (reconverted to GlobalPoint)
  x = GlobalPoint(planeCrossing.position(s));
  //
  return true;
}

bool AnalyticalPropagator::propagateWithLineCrossing(
    const GlobalPoint& x0, const GlobalVector& p0, const Cylinder& cylinder, GlobalPoint& x, double& s) const {
  //
  // Instantiate auxiliary object for finding intersection.
  // Frame-independant point and vector are created explicitely to
  // avoid confusing gcc (refuses to compile with temporary objects
  // in the constructor).
  //
  StraightLineBarrelCylinderCrossing cylCrossing(x0, p0, propagationDirection());
  //
  // get solution
  //
  std::pair<bool, double> propResult = cylCrossing.pathLength(cylinder);
  if (!propResult.first)
    return false;
  s = propResult.second;
  // point (reconverted to GlobalPoint)
  x = cylCrossing.position(s);
  //
  return true;
}

bool AnalyticalPropagator::propagateWithHelixCrossing(HelixPlaneCrossing& planeCrossing,
                                                      const Plane& plane,
                                                      const float pmag,
                                                      GlobalPoint& x,
                                                      GlobalVector& p,
                                                      double& s) const {
  // get solution
  std::pair<bool, double> propResult = planeCrossing.pathLength(plane);
  if UNLIKELY (!propResult.first)
    return false;

  s = propResult.second;
  x = GlobalPoint(planeCrossing.position(s));
  // direction (reconverted to GlobalVector, renormalised)
  GlobalVector pGen = GlobalVector(planeCrossing.direction(s));
  pGen *= pmag / pGen.mag();
  p = pGen;
  //
  return true;
}
