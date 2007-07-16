#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"

InvalidTransientRecHit::InvalidTransientRecHit( const GeomDet* geom, const DetLayer * layer, Type type ) :
  GenericTransientTrackingRecHit( geom, InvalidTrackingRecHit( geom == 0 ? DetId(0) : geom->geographicalId(), type)), 
  layer_(layer)
{
}

/*
AlgebraicVector InvalidTransientRecHit::parameters(const TrajectoryStateOnSurface& ts) const
{
  throw cms::Exception("Invalid TransientTrackingRecHit used");
  return AlgebraicVector();
}

AlgebraicSymMatrix InvalidTransientRecHit::parametersError(const TrajectoryStateOnSurface& ts) const
{
  throw cms::Exception("Invalid TransientTrackingRecHit used");
  return AlgebraicSymMatrix();
}
*/
