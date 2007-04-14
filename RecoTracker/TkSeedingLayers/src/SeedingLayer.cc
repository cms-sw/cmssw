#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "HitExtractor.h"

using namespace ctfseeding;
using namespace std;

SeedingLayer::SeedingLayer( const DetLayer* layer, const string & name,
    const std::string & hitBuilder, const HitExtractor * hitExtractor)
  : theLayer(layer), theName(name), 
    theTTRHBuilderName(hitBuilder), theHitExtractor(hitExtractor), theTTRHBuilder(0)
{ }

SeedingLayer::~SeedingLayer()
{ 
//  delete theHitExtractor;
}

const TransientTrackingRecHitBuilder * SeedingLayer::hitBuilder(const edm::EventSetup& es) const 
{
  if (!theTTRHBuilder) {
    edm::ESHandle<TransientTrackingRecHitBuilder> builder;
    es.get<TransientRecHitRecord>().get(theTTRHBuilderName,builder);
    theTTRHBuilder = &(*builder);
  }
  return theTTRHBuilder;
}

std::vector<SeedingHit> SeedingLayer::hits(const edm::Event& ev, const edm::EventSetup& es) const
{
  return theHitExtractor->hits(ev,es);
}
