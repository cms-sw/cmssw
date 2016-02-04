#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/GeomPropagators/interface/IterativeHelixExtrapolatorToLine.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

// #include "CommonDet/DetUtilities/interface/DetailedDetTimer.h"

AnalyticalImpactPointExtrapolator::AnalyticalImpactPointExtrapolator (const MagneticField* field) :
  thePropagator(new AnalyticalPropagator(theField, anyDirection)),
  theField(field)
{}

AnalyticalImpactPointExtrapolator::AnalyticalImpactPointExtrapolator (const Propagator& propagator,
								      const MagneticField* field) :
  thePropagator(propagator.clone()),
  theField(field)
{
  thePropagator->setPropagationDirection(anyDirection);
}

TrajectoryStateOnSurface 
AnalyticalImpactPointExtrapolator::extrapolate (const FreeTrajectoryState& fts, 
						const GlobalPoint& vtx) const
{
//   static TimingReport::Item& timer = detailedDetTimer("AnalyticalImpactPointExtrapolator");
//   TimeMe t(timer,false);

  return extrapolateSingleState(fts, vtx);
}

TrajectoryStateOnSurface 
AnalyticalImpactPointExtrapolator::extrapolate (const TrajectoryStateOnSurface tsos, 
						const GlobalPoint& vtx) const
{
  if ( tsos.isValid() )  return extrapolateFullState(tsos,vtx);
  else  return tsos;
}

TrajectoryStateOnSurface 
AnalyticalImpactPointExtrapolator::extrapolateFullState (const TrajectoryStateOnSurface tsos, 
							 const GlobalPoint& vertex) const
{
  //
  // first determine IP plane using propagation with (single) FTS
  // could be optimised (will propagate errors even if duplicated below)
  //
  TrajectoryStateOnSurface singleState = 
    extrapolateSingleState(*tsos.freeTrajectoryState(),vertex);
  if ( !singleState.isValid() || tsos.components().size()==1 )  return singleState;
  //
  // propagate multiTsos to plane found above
  //
  return thePropagator->propagate(tsos,singleState.surface());
}

TrajectoryStateOnSurface 
AnalyticalImpactPointExtrapolator::extrapolateSingleState (const FreeTrajectoryState& fts, 
							   const GlobalPoint& vertex) const
{
  //
  // initialisation of position, momentum and transverse curvature
  //
  GlobalPoint x(fts.position());
  GlobalVector p(fts.momentum());
  double rho = fts.transverseCurvature();
  //
  // Straight line approximation? |rho|<1.e-10 equivalent to ~ 1um 
  // difference in transversal position at 10m.
  //
  double s(0);
  if( fabs(rho)<1.e-10 ) {
    GlobalVector dx(p.dot(vertex-x)/p.mag2()*p);
    x += dx;
    float sign = p.dot(dx);
    s = sign>0 ? dx.mag() : -dx.mag();
  }
  //
  // Helix case 
  //
  else {
    HelixLineExtrapolation::PositionType helixPos(x);
    HelixLineExtrapolation::DirectionType helixDir(p);
    IterativeHelixExtrapolatorToLine extrapolator(helixPos,helixDir,rho,anyDirection);
    if ( !propagateWithHelix(extrapolator,vertex,x,p,s) )  return TrajectoryStateOnSurface();
  }
  //
  // Define target surface: origin on line, x_local from line 
  //   to helix at closest approach, z_local along the helix
  //   and y_local to complete right-handed system
  //
  GlobalVector zLocal(p.unit());
  GlobalVector yLocal(zLocal.cross(x-vertex).unit());
  GlobalVector xLocal(yLocal.cross(zLocal));
  Surface::RotationType rot(xLocal,yLocal,zLocal);
  PlaneBuilder::ReturnType surface = PlaneBuilder().plane(vertex,rot);
  //
  // Compute propagated state
  //
  GlobalTrajectoryParameters gtp(x,p,fts.charge(), theField);
  if (fts.hasError()) {
    //
    // compute jacobian
    //
    AnalyticalCurvilinearJacobian analyticalJacobian(fts.parameters(), gtp.position(), gtp.momentum(), s);
    CurvilinearTrajectoryError cte( ROOT::Math::Similarity( analyticalJacobian.jacobian(), 
                                                            fts.curvilinearError().matrix()) );
    return TrajectoryStateOnSurface(gtp,cte,*surface);
  }
  else {
    //
    // return state without errors
    //
    return TrajectoryStateOnSurface(gtp,*surface);
  }
}

bool
AnalyticalImpactPointExtrapolator::propagateWithHelix (const IterativeHelixExtrapolatorToLine& extrapolator,
						       const GlobalPoint& vertex,
						       GlobalPoint& x, GlobalVector& p, double& s) const {
  //
  // save absolute value of momentum
  //
  double pmag(p.mag());
  //
  // get path length to solution
  //
  std::pair<bool,double> propResult = extrapolator.pathLength(vertex);
  if ( !propResult.first )  return false;
  s = propResult.second;
  // 
  // get point and (normalised) direction from path length
  //
  HelixLineExtrapolation::PositionType xGen = extrapolator.position(s);
  HelixLineExtrapolation::DirectionType pGen = extrapolator.direction(s);
  //
  // Fix normalisation and convert back to GlobalPoint / GlobalVector
  //
  x = GlobalPoint(xGen);
  pGen *= pmag/pGen.mag();
  p = GlobalVector(pGen);
  //
  return true;
}
