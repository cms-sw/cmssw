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
  auto hitPos = hit.globalPosition();
  auto hitR = hitPos.perp();

  auto hitZ = hitPos.z();
  if ( !(*theRZCompatibility)(hitR,hitZ) ) return false;

  auto hitPhi = hitPos.barePhi();
  if ( !checkPhi(hitPhi, hitR) ) return false;


  return true;
}

