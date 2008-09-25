#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerGSRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "FastSimDataFormats/External/interface/FastTrackerCluster.h"



TransientTrackingRecHit::RecHitPointer
TSiTrackerGSRecHit::clone (const TrajectoryStateOnSurface& ts) const
{
  // Just clones the old hit
  /* 
  /// Copies everything from the old hit
  
  if(!specificHit()->cluster().isNull()){
    const FastTrackerCluster clust = *specificHit()->cluster();  
    return TSiTrackerGSRecHit::build( clust.localPosition(), clust.localPositionError(), det(), 
                                      specificHit()->cluster(),clust.simhitId(), clust.simtrackId(),
                                      clust.eeId(), specificHit()->simMultX(),
                                      specificHit()->simMultY(), weight(), getAnnealingFactor());
    
  }
  /// FIXME: should report the problem somehow
  else */ return clone();
}

const GeomDetUnit* TSiTrackerGSRecHit::detUnit() const
{
  return static_cast<const GeomDetUnit*>(det());
}
