#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "SeedingLayerImpl.h"

using namespace ctfseeding;

SeedingLayer::SeedingLayer( 
    const std::string & name, 
    const DetLayer* layer, 
    const TransientTrackingRecHitBuilder * hitBuilder,
    const HitExtractor * hitExtractor,
    bool usePredefinedErrors, float hitErrorRZ, float hitErrorRPhi)
{
  SeedingLayerImpl * l = 
      usePredefinedErrors ? 
      new SeedingLayerImpl(name,layer,hitBuilder,hitExtractor,hitErrorRZ,hitErrorRPhi)
    : new SeedingLayerImpl(name,layer,hitBuilder,hitExtractor);
  theImpl = boost::shared_ptr<SeedingLayerImpl> (l);
}

std::string SeedingLayer::name() const
{
  return theImpl->name();
}

const DetLayer*  SeedingLayer::detLayer() const
{
  return theImpl->detLayer();
}

const TransientTrackingRecHitBuilder * SeedingLayer::hitBuilder() const 
{
  return theImpl->hitBuilder();
}

std::vector<SeedingHit> SeedingLayer::hits(const edm::Event& ev, const edm::EventSetup& es) const
{
  return  theImpl->hits( *this,ev,es);
}


bool SeedingLayer::hasPredefinedHitErrors() const 
{
  return theImpl->hasPredefinedHitErrors();
}

float SeedingLayer::predefinedHitErrorRZ() const
{
  return theImpl->predefinedHitErrorRZ();
}

float SeedingLayer::predefinedHitErrorRPhi() const
{
  return theImpl->predefinedHitErrorRPhi();
}
