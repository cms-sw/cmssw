#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "HitExtractor.h"


using namespace ctfseeding;

SeedingLayer::SeedingLayer(
			   const std::string & name,
			   const DetLayer* layer,
			   const TransientTrackingRecHitBuilder * hitBuilder,
			   const HitExtractor * hitExtractor,
			   bool usePredefinedErrors,
			   float hitErrorRZ, float hitErrorRPhi)
  : theName(name), theLayer(layer),
    theTTRHBuilder(hitBuilder), theHitExtractor(hitExtractor),
    theHasPredefinedHitErrors(usePredefinedErrors),
    thePredefinedHitErrorRZ(hitErrorRZ), thePredefinedHitErrorRPhi(hitErrorRPhi) { }

SeedingLayer::~SeedingLayer() { delete theHitExtractor; }

  SeedingLayer::Hits SeedingLayer::hits(const edm::Event& ev, 
			  const edm::EventSetup& es) const { return theHitExtractor->hits(*this,ev,es);  }



