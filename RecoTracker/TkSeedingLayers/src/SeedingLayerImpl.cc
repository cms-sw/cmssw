#include "SeedingLayerImpl.h"

//#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
//#include "TrackingTools/DetLayers/interface/DetLayer.h"

//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//#include "DataFormats/Common/interface/Handle.h"

//#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
//#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "HitExtractor.h"

using namespace ctfseeding;
using namespace std;

SeedingLayerImpl::SeedingLayerImpl( 
    const string & name,
    const DetLayer* layer, 
    const TransientTrackingRecHitBuilder * hitBuilder, 
    const HitExtractor * hitExtractor)
  : 
    theName(name), 
    theLayer(layer), 
    theTTRHBuilder(hitBuilder), 
    theHitExtractor(hitExtractor), 
    theHasPredefinedHitErrors(false),thePredefinedHitErrorRZ(0.),thePredefinedHitErrorRPhi(0.)
{
}
SeedingLayerImpl::SeedingLayerImpl( 
    const string & name,
    const DetLayer* layer, 
    const TransientTrackingRecHitBuilder * hitBuilder, 
    const HitExtractor * hitExtractor,
    float hitErrorRZ, float hitErrorRPhi)
  : theName(name), theLayer(layer), 
    theTTRHBuilder(hitBuilder), theHitExtractor(hitExtractor),
    theHasPredefinedHitErrors(true), 
    thePredefinedHitErrorRZ(hitErrorRZ), thePredefinedHitErrorRPhi(hitErrorRPhi)
{
}

SeedingLayerImpl::~SeedingLayerImpl()
{ 
  delete theHitExtractor;
}

vector<SeedingHit> SeedingLayerImpl::hits(const SeedingLayer &sl, const edm::Event& ev, const edm::EventSetup& es) const
{
  return theHitExtractor->hits(sl,ev,es);
}
