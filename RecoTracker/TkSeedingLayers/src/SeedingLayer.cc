#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

using namespace ctfseeding;
using namespace std;

SeedingLayer::SeedingLayer(const DetLayer* layer, const string & name, Side & side, int idLayer)
  : theLayer(layer), theName(name), theHitProducer("siPixelRecHits"), 
    theSide(side), theIdLayer(idLayer) 
{ }


SeedingLayer::~SeedingLayer()
{ }

std::vector<SeedingHit> SeedingLayer::hits(const edm::Event& ev, const edm::EventSetup& es) const
{
  TrackerLayerIdAccessor accessor;
  std::vector<SeedingHit> result;
  if (theLayer->subDetector() == GeomDetEnumerators::PixelBarrel ||
      theLayer->subDetector() == GeomDetEnumerators::PixelEndcap) {
    edm::Handle<SiPixelRecHitCollection> pixelHits;
    ev.getByLabel( theHitProducer, pixelHits);
    const SiPixelRecHitCollection::range range = (theSide==Barrel) ?
        pixelHits->get(accessor.pixelBarrelLayer(theIdLayer))
      : pixelHits->get(accessor.pixelForwardDisk(theSide,theIdLayer));
    for(SiPixelRecHitCollection::const_iterator it = range.first; it != range.second; it++){
      result.push_back( SeedingHit(&(*it), es) );
    }
  } 
  else if (theLayer->subDetector() == GeomDetEnumerators::TIB) {
    edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
    ev.getByLabel( theHitProducer, matchedHits);
    const SiStripMatchedRecHit2DCollection::range range =
         matchedHits->get(accessor.stripTIBLayer(theIdLayer) );
    for(SiStripMatchedRecHit2DCollection::const_iterator it=range.first; it!=range.second; it++){
      result.push_back( SeedingHit(&(*it), es) );
    }
  } 
  return result;
}
