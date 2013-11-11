#include "RecoTracker/TkTrackingRegions/interface/OuterHitCompatibility.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

bool OuterHitCompatibility::operator() ( const TransientTrackingRecHit * hit) const
{
  GlobalPoint hitPos = hit->globalPosition();
  float hitR = hitPos.perp();
  float hitPhi = hitPos.phi();

  if ( !checkPhi(hitPhi, hitR) ) return 0;

  float hitZ = hitPos.z();
  if ( !(*theRZCompatibility)(hitR,hitZ) ) return 0;

  return 1;
}


bool OuterHitCompatibility::operator() ( const TrackingRecHit* hit,  const edm::EventSetup& iSetup) const
{
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  DetId tmp=hit->geographicalId();
  GlobalPoint hitPos = tracker->idToDet(tmp)->surface().toGlobal(hit->localPosition());
  float hitR = hitPos.perp();
  float hitPhi = hitPos.phi();

  if ( !checkPhi(hitPhi, hitR) ) return 0;
 
  float hitZ = hitPos.z();
  if ( !(*theRZCompatibility)(hitR,hitZ) ) return 0;

  return 1;
}

bool OuterHitCompatibility::checkPhi(const float & phi, const float & r) const 
{
  OuterHitPhiPrediction::Range hitPhiRange = thePhiPrediction(r);
  PhiLess less;
  bool phiOK = less(hitPhiRange.min(),phi) && less(phi,hitPhiRange.max());
  return phiOK;
}

