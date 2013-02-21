#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "HitExtractor.h"


using namespace ctfseeding;

SeedingLayer::SeedingLayer(SeedingLayer && rh) noexcept :
theName(rh.theName), theLayer(rh.theLayer),
theTTRHBuilder(rh.theTTRHBuilder), theHitExtractor(std::move(rh.theHitExtractor)),
theHasPredefinedHitErrors(rh.theHasPredefinedHitErrors),
thePredefinedHitErrorRZ(rh.thePredefinedHitErrorRZ), thePredefinedHitErrorRPhi(rh.thePredefinedHitErrorRPhi) { }


SeedingLayer::SeedingLayer(
			   const std::string & name,
			   const DetLayer* layer,
			   const TransientTrackingRecHitBuilder * hitBuilder,
			   const HitExtractor * hitExtractor,
			   bool usePredefinedErrors,
			   float hitErrorRZ, float hitErrorRPhi)
  : theName(name), theLayer(layer),
    theTTRHBuilder(hitBuilder), theHitExtractor(const_cast<HitExtractor *>(hitExtractor)),
    theHasPredefinedHitErrors(usePredefinedErrors),
    thePredefinedHitErrorRZ(hitErrorRZ), thePredefinedHitErrorRPhi(hitErrorRPhi) { }

SeedingLayer::~SeedingLayer() {}

  SeedingLayer::Hits SeedingLayer::hits(const edm::Event& ev, 
			  const edm::EventSetup& es) const { return theHitExtractor->hits(*this,ev,es);  }



