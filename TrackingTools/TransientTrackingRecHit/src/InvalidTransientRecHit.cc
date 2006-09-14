#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"

InvalidTransientRecHit::InvalidTransientRecHit( const GeomDet* geom) :
  GenericTransientTrackingRecHit( geom, InvalidTrackingRecHit( geom == 0 ? DetId(0) : geom->geographicalId()))
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
