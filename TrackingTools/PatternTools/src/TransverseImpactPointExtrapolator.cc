#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "boost/intrusive_ptr.hpp" 
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/GeometryVector/interface/Point2DBase.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

TransverseImpactPointExtrapolator::TransverseImpactPointExtrapolator () :
  thePropagator(0) {}


TransverseImpactPointExtrapolator::TransverseImpactPointExtrapolator (const MagneticField* field) :
  thePropagator(new AnalyticalPropagator(field, anyDirection)) {}

TransverseImpactPointExtrapolator::TransverseImpactPointExtrapolator (const Propagator& u) :
  thePropagator(u.clone()) 
{
  thePropagator->setPropagationDirection(anyDirection);
}

TrajectoryStateOnSurface 
TransverseImpactPointExtrapolator::extrapolate (const FreeTrajectoryState& fts, 
						const GlobalPoint& vtx) const
{
  return doExtrapolation(fts, vtx, *thePropagator);
}

TrajectoryStateOnSurface 
TransverseImpactPointExtrapolator::extrapolate (const TrajectoryStateOnSurface tsos, 
						const GlobalPoint& vtx) const
{
  if ( !tsos.isValid() )  return tsos;
  return doExtrapolation(tsos, vtx, *thePropagator);
}

TrajectoryStateOnSurface 
TransverseImpactPointExtrapolator::extrapolate (const FreeTrajectoryState& fts, 
						const GlobalPoint& vtx, 
						const Propagator& u) const
{
  //
  // use clone of propagator for bidirectional propagation
  //
  DeepCopyPointerByClone<Propagator> p(u.clone());
  p->setPropagationDirection(anyDirection);

  return doExtrapolation(fts,vtx,*p);
} 

TrajectoryStateOnSurface 
TransverseImpactPointExtrapolator::extrapolate (const TrajectoryStateOnSurface tsos, 
						const GlobalPoint& vtx, 
						const Propagator& u) const
{
  if ( !tsos.isValid() )  return tsos;
  //
  // use clone of propagator for bidirectional propagation
  //
  DeepCopyPointerByClone<Propagator> p(u.clone());
  p->setPropagationDirection(anyDirection);

  return doExtrapolation(tsos,vtx,*p);
} 

TrajectoryStateOnSurface 
TransverseImpactPointExtrapolator::doExtrapolation (const TrajectoryStateOnSurface tsos, 
						    const GlobalPoint& vtx, 
						    const Propagator& p) const
{
  //
  // Compute tip surface
  //
  ReferenceCountingPointer<BoundPlane> surface = 
    tipSurface(tsos.globalPosition(),tsos.globalMomentum(),
	       1./tsos.transverseCurvature(),vtx);
  //
  // propagate
  //
  return p.propagate(tsos,*surface);
}

TrajectoryStateOnSurface 
TransverseImpactPointExtrapolator::doExtrapolation (const FreeTrajectoryState& fts, 
						    const GlobalPoint& vtx, 
						    const Propagator& p) const
{
  //
  // Compute tip surface
  //
  ReferenceCountingPointer<BoundPlane> surface = 
    tipSurface(fts.position(),fts.momentum(),
	       1./fts.transverseCurvature(),vtx);
  //
  // propagate
  //
  return p.propagate(fts,*surface);
}

ReferenceCountingPointer<BoundPlane>
TransverseImpactPointExtrapolator::tipSurface (const GlobalPoint& position,
					       const GlobalVector& momentum,
					       const double& signedTransverseRadius,
					       const GlobalPoint& vertex) const
{
  typedef Point2DBase<double,GlobalTag> PositionType2D;
  typedef Vector2DBase<double,GlobalTag> DirectionType2D;
  
  PositionType2D x0(position.x(),position.y());
  DirectionType2D t0(-momentum.y(),momentum.x());
  t0 = t0/t0.mag();

  PositionType2D xc(x0+signedTransverseRadius*t0);

  DirectionType2D vtxDirection(xc.x()-vertex.x(),xc.y()-vertex.y());
  double vtxDistance = vtxDirection.mag();

  Surface::PositionType origin(vertex);
  GlobalVector xLocal(vtxDirection.x()/vtxDistance,
		      vtxDirection.y()/vtxDistance,
		      0.);
  if ( vtxDistance<fabs(signedTransverseRadius) )  xLocal = -xLocal;
  GlobalVector yLocal(0.,0.,1.);
  GlobalVector zLocal(xLocal.cross(yLocal));
  if ( zLocal.dot(momentum)<0. ) {
    yLocal = -yLocal;
    zLocal = -zLocal;
  }
  Surface::RotationType rotation(xLocal,yLocal,zLocal);
  return PlaneBuilder().plane(origin,rotation);
}
