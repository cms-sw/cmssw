#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

const GeomDetUnit * GenericTransientTrackingRecHit::detUnit() const
{
  return dynamic_cast<const GeomDetUnit*>(det());
}
