#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"

#include "TrackingTools/GeomPropagators/interface/IterativeHelixExtrapolatorToLine.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"

#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/Line.h"

#include <cmath>


AnalyticalTrajectoryExtrapolatorToLine::AnalyticalTrajectoryExtrapolatorToLine (const MagneticField* field) :
  thePropagator(new AnalyticalPropagator(field, anyDirection)) {}

AnalyticalTrajectoryExtrapolatorToLine::AnalyticalTrajectoryExtrapolatorToLine 
(const Propagator& propagator) : thePropagator(propagator.clone()) 
{
  thePropagator->setPropagationDirection(anyDirection);
}

TrajectoryStateOnSurface 
AnalyticalTrajectoryExtrapolatorToLine::extrapolate (const FreeTrajectoryState& fts, 
						     const Line& line) const
{
  return extrapolateSingleState(fts,line);
}

TrajectoryStateOnSurface 
AnalyticalTrajectoryExtrapolatorToLine::extrapolate (const TrajectoryStateOnSurface tsos, 
						     const Line& line) const
{
  if ( tsos.isValid() )  return extrapolateFullState(tsos,line);
  else  return tsos;
}

TrajectoryStateOnSurface 
AnalyticalTrajectoryExtrapolatorToLine::extrapolateFullState (const TrajectoryStateOnSurface tsos, 
							      const Line& line) const
{
  //
  // first determine IP plane using propagation with (single) FTS
  // could be optimised (will propagate errors even if duplicated below)
  //
  TrajectoryStateOnSurface singleState = 
    extrapolateSingleState(*tsos.freeTrajectoryState(),line);
  if ( !singleState.isValid() || tsos.components().size()==1 )  return singleState;
  //
  // propagate multiTsos to plane found above
  //
  return thePropagator->propagate(tsos,singleState.surface());
}

TrajectoryStateOnSurface
AnalyticalTrajectoryExtrapolatorToLine::extrapolateSingleState (const FreeTrajectoryState& fts, 
								const Line& line) const
{
//   static TimingReport::Item& timer = detailedDetTimer("AnalyticalTrajectoryExtrapolatorToLine");
//   TimeMe t(timer,false);
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
    Line tangent(x,p);
    GlobalPoint xold(x);
    x = tangent.closerPointToLine(line);
    GlobalVector dx(x-xold);
    float sign = p.dot(x-xold);
    s = sign>0 ? dx.mag() : -dx.mag();
  }
  //
  // Helix case 
  //
  else {
    HelixLineExtrapolation::PositionType helixPos(x);
    HelixLineExtrapolation::DirectionType helixDir(p);
    IterativeHelixExtrapolatorToLine extrapolator(helixPos,helixDir,rho,anyDirection);
    if ( !propagateWithHelix(extrapolator,line,x,p,s) )  return TrajectoryStateOnSurface();
  }
  //
  // Define target surface: origin on line, x_local from line 
  //   to helix at closest approach, z_local along the helix
  //   and y_local to complete right-handed system
  //
  GlobalPoint origin(line.closerPointToLine(Line(x,p)));
  GlobalVector zLocal(p.unit());
  GlobalVector yLocal(zLocal.cross(x-origin).unit());
  GlobalVector xLocal(yLocal.cross(zLocal));
  Surface::RotationType rot(xLocal,yLocal,zLocal);
  PlaneBuilder::ReturnType surface = PlaneBuilder().plane(origin,rot);
  //
  // Compute propagated state
  //
  GlobalTrajectoryParameters gtp(x,p,fts.charge(), thePropagator->magneticField());
  if (fts.hasError()) {
    //
    // compute jacobian
    //
    AnalyticalCurvilinearJacobian analyticalJacobian(fts.parameters(), gtp.position(), gtp.momentum(), s);
    const AlgebraicMatrix55 &jacobian = analyticalJacobian.jacobian();
    CurvilinearTrajectoryError cte( ROOT::Math::Similarity (jacobian, fts.curvilinearError().matrix()) );
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
AnalyticalTrajectoryExtrapolatorToLine::propagateWithHelix (const IterativeHelixExtrapolatorToLine& extrapolator,
							    const Line& line,
							    GlobalPoint& x, GlobalVector& p, double& s) const {
  //
  // save absolute value of momentum
  //
  double pmag(p.mag());
  //
  // get path length to solution
  //
  std::pair<bool,double> propResult = extrapolator.pathLength(line);
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
