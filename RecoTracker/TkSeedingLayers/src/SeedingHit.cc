#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace ctfseeding;

SeedingHit::SeedingHit(const TrackingRecHit * hit ,  const edm::EventSetup& iSetup)
 : theRecHit(hit)
{
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  GlobalPoint gp = tracker->idToDet(
      hit->geographicalId())->surface().toGlobal(hit->localPosition());
  thePhi = gp.phi();
  theR = gp.perp();
  theZ = gp.z();
  unsigned int subid=hit->geographicalId().subdetId();
  theRZ = (   subid == PixelSubdetector::PixelBarrel
           || subid == StripSubdetector::TIB
           || subid == StripSubdetector::TOB) ? theZ : theR;
}
