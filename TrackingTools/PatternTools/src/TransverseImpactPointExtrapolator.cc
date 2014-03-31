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

#include "FWCore/MessageLogger/interface/MessageLogger.h"


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
						const Propagator& p) const
{
  //
  // set propagator for bidirectional propagation
  //
   std::unique_ptr<Propagator> p_cloned = SetPropagationDirection(p, anyDirection);
   return doExtrapolation(fts, vtx, *(p_cloned.get()));
}

TrajectoryStateOnSurface
TransverseImpactPointExtrapolator::extrapolate (const TrajectoryStateOnSurface tsos,
						const GlobalPoint& vtx,
						const Propagator& p) const
{
  if ( !tsos.isValid() )  return tsos;
  //
  // set propagator for bidirectional propagation
  //
  std::unique_ptr<Propagator> p_cloned = SetPropagationDirection(p, anyDirection);
  return doExtrapolation(tsos, vtx, *(p_cloned.get()));
}

TrajectoryStateOnSurface
TransverseImpactPointExtrapolator::doExtrapolation (const TrajectoryStateOnSurface tsos,
						    const GlobalPoint& vtx,
						    const Propagator& p) const
{
  //
  // Compute tip surface
  //
  if (fabs(tsos.freeState()->transverseCurvature())<1.E-6){
    LogDebug("TransverseImpactPointExtrapolator")<< "negligeable curvature: using a trick to extrapolate:\n"<<tsos;

    //0T field probably
    //x is perpendicular to the momentum
    GlobalVector xLocal = GlobalVector(-tsos.globalMomentum().y(),tsos.globalMomentum().x(),0).unit();
    //y along global Z
    GlobalVector yLocal(0.,0.,1.);
    //z accordingly
    GlobalVector zLocal(xLocal.cross(yLocal));

    Surface::PositionType origin(vtx);
    Surface::RotationType rotation(xLocal,yLocal,zLocal);
    ReferenceCountingPointer<Plane> surface =  PlaneBuilder().plane(origin,rotation);

    return p.propagate(*tsos.freeState(),*surface);
  }else{
  ReferenceCountingPointer<Plane> surface =
    tipSurface(tsos.globalPosition(),tsos.globalMomentum(),
	       1./tsos.transverseCurvature(),vtx);
  //
  // propagate
  //
  return p.propagate(tsos,*surface);
  }
}

TrajectoryStateOnSurface
TransverseImpactPointExtrapolator::doExtrapolation (const FreeTrajectoryState& fts,
						    const GlobalPoint& vtx,
						    const Propagator& p) const
{
  //
  // Compute tip surface
  //
  if (fabs(fts.transverseCurvature())<1.E-6){
    LogDebug("TransverseImpactPointExtrapolator")<< "negligeable curvature: using a trick to extrapolate:\n"<<fts;

    //0T field probably
    //x is perpendicular to the momentum
    GlobalVector xLocal = GlobalVector(-fts.momentum().y(),fts.momentum().x(),0).unit();
    //y along global Z
    GlobalVector yLocal(0.,0.,1.);
    //z accordingly
    GlobalVector zLocal(xLocal.cross(yLocal));

    Surface::PositionType origin(vtx);
    Surface::RotationType rotation(xLocal,yLocal,zLocal);
    ReferenceCountingPointer<Plane> surface =  PlaneBuilder().plane(origin,rotation);

    return p.propagate(fts,*surface);
  }else{
  ReferenceCountingPointer<Plane> surface =
    tipSurface(fts.position(),fts.momentum(),
	       1./fts.transverseCurvature(),vtx);
  //
  // propagate
  //
  return p.propagate(fts,*surface);
  }
}

ReferenceCountingPointer<Plane>
TransverseImpactPointExtrapolator::tipSurface (const GlobalPoint& position,
					       const GlobalVector& momentum,
					       const double& signedTransverseRadius,
					       const GlobalPoint& vertex) const
{

  LogDebug("TransverseImpactPointExtrapolator")<< position<<"\n"
						    <<momentum<<"\n"
						    <<"signedTransverseRadius : "<<signedTransverseRadius<<"\n"
						    <<vertex;

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
  if ( vtxDistance<fabs(signedTransverseRadius) ) {
    LogDebug("TransverseImpactPointExtrapolator")<<"Inverting the x axis.";
    xLocal = -xLocal;
  }
  GlobalVector yLocal(0.,0.,1.);
  GlobalVector zLocal(xLocal.cross(yLocal));
  if ( zLocal.dot(momentum)<0. ) {
    LogDebug("TransverseImpactPointExtrapolator")<<"Inverting the y,z frame.";
    yLocal = -yLocal;
    zLocal = -zLocal;
  }
  Surface::RotationType rotation(xLocal,yLocal,zLocal);

  LogDebug("TransverseImpactPointExtrapolator")<<"plane center: "<<origin<<"\n"
					       <<"plane rotation axis:\n"
					       <<xLocal<<"\n"
					       <<yLocal<<"\n"
					       <<zLocal<<"\n"
					       <<"x0: "<<x0<<"\n"
					       <<"t0: "<<t0<<"\n"
					       <<"xc: "<<xc<<"\n"
					       <<"vtxDirection: "<<vtxDirection;

  return PlaneBuilder().plane(origin,rotation);
}
