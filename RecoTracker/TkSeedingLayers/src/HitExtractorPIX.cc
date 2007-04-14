#include "HitExtractorPIX.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

using namespace ctfseeding;
using namespace std;

HitExtractorPIX::HitExtractorPIX(
    SeedingLayer::Side & side, int idLayer, const std::string & hitProducer,
    double hitErrorRPhi, double hitErrorRZ)
  : theSide(side), theIdLayer(idLayer), theHitProducer(hitProducer),
    theUseHitErrors(true), theHitErrorRPhi(hitErrorRPhi), theHitErrorRZ(hitErrorRZ)
{ }
HitExtractorPIX::HitExtractorPIX(
    SeedingLayer::Side & side, int idLayer, const std::string & hitProducer)
  : theSide(side), theIdLayer(idLayer), theHitProducer(hitProducer),
    theUseHitErrors(false), theHitErrorRPhi(0), theHitErrorRZ(0)
{ }


vector<SeedingHit> HitExtractorPIX::hits(const edm::Event& ev, const edm::EventSetup& es) const
{
  TrackerLayerIdAccessor accessor;
  std::vector<SeedingHit> result;
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByLabel( theHitProducer, pixelHits);
  const SiPixelRecHitCollection::range range = (theSide==SeedingLayer::Barrel) ?
        pixelHits->get(accessor.pixelBarrelLayer(theIdLayer))
      : pixelHits->get(accessor.pixelForwardDisk(theSide,theIdLayer));
  for(SiPixelRecHitCollection::const_iterator it = range.first; it != range.second; it++){
      result.push_back( SeedingHit(&(*it), es) );
  }
  return result;
}
