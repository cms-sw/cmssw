#include "HitExtractorPIX.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/Common/interface/DetSetAlgorithm.h"


using namespace ctfseeding;
using namespace std;

HitExtractorPIX::HitExtractorPIX(
    SeedingLayer::Side & side, int idLayer, const std::string & hitProducer)
  : theSide(side), theIdLayer(idLayer), theHitProducer(hitProducer)
{ }



vector<SeedingHit> HitExtractorPIX::hits(const SeedingLayer & sl,const edm::Event& ev, const edm::EventSetup& es) const
{
  TrackerLayerIdAccessor accessor;
  std::vector<SeedingHit> result;
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByLabel( theHitProducer, pixelHits);

  const SiPixelRecHitCollection::Range range = (theSide==SeedingLayer::Barrel) ?
    edmNew::detsetRangeFromPair(*pixelHits,accessor.pixelBarrelLayer(theIdLayer))
    :  edmNew::detsetRangeFromPair(*pixelHits,accessor.pixelForwardDisk(theSide,theIdLayer));
  for(SiPixelRecHitCollection::const_iterator id = range.first; id != range.second; id++)
    for(SiPixelRecHitCollection::DetSet::const_iterator it = (*id).begin(); it != (*id).end(); it++){
      result.push_back( SeedingHit(&(*it), sl, es) );
    }
  return result;
}
