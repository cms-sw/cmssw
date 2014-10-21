#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//
// include all the concrete ones
//
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
//
// For FAMOS
//
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"                         
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"                         

#include "Utilities/General/interface/ClassName.h"
#include <typeinfo>

TkTransientTrackingRecHitBuilder::TkTransientTrackingRecHitBuilder( const TrackingGeometry* trackingGeometry, 
								    const PixelClusterParameterEstimator * pCPE,
								    const StripClusterParameterEstimator * sCPE,
								    const SiStripRecHitMatcher * matcher,
								    bool computeCoarseLocalPositionFromDisk):  
  tGeometry_(trackingGeometry),
  pixelCPE(pCPE),
  stripCPE(sCPE),
  theMatcher(matcher),
  theComputeCoarseLocalPosition(computeCoarseLocalPositionFromDisk){}

TransientTrackingRecHit::RecHitPointer 
TkTransientTrackingRecHitBuilder::build (const TrackingRecHit * p) const 
{
//  assert("TkTransientTrackingRecHitBuilder::build"==nullptr);

  return (*p).cloneSH();
}

