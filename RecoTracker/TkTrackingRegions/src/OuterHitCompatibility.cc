#include "OuterHitCompatibility.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

bool OuterHitCompatibility::operator() ( const TrackingRecHit & hit) const
{
  GlobalPoint hitPos = hit.globalPosition();
  float hitR = hitPos.perp();
  float hitPhi = hitPos.phi();

  if ( !checkPhi(hitPhi, hitR) ) return false;

  float hitZ = hitPos.z();
  if ( !(*theRZCompatibility)(hitR,hitZ) ) return false;

  return true;
}

