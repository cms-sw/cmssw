#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "HitExtractor.h"


using namespace ctfseeding;
using namespace std;


class SeedingLayer::SeedingLayerImpl {
public:
  SeedingLayerImpl(
                const std::string & name,
                const DetLayer* layer,
                const TransientTrackingRecHitBuilder * hitBuilder,
                const HitExtractor * hitExtractor)
  : theName(name),
    theLayer(layer),
    theTTRHBuilder(hitBuilder),
    theHitExtractor(hitExtractor),
    theHasPredefinedHitErrors(false),thePredefinedHitErrorRZ(0.),thePredefinedHitErrorRPhi(0.) { }

  SeedingLayerImpl(
    const string & name,
    const DetLayer* layer,
    const TransientTrackingRecHitBuilder * hitBuilder,
    const HitExtractor * hitExtractor,
    float hitErrorRZ, float hitErrorRPhi)
  : theName(name), theLayer(layer),
    theTTRHBuilder(hitBuilder), theHitExtractor(hitExtractor),
    theHasPredefinedHitErrors(true),
    thePredefinedHitErrorRZ(hitErrorRZ), thePredefinedHitErrorRPhi(hitErrorRPhi) { }

  ~SeedingLayerImpl() { delete theHitExtractor; }

  SeedingLayer::Hits hits(const SeedingLayer &sl, const edm::Event& ev, 
			  const edm::EventSetup& es) const { return theHitExtractor->hits(sl,ev,es);  }

  std::string name() const { return theName; }

  const DetLayer*  detLayer() const { return theLayer; }
  const TransientTrackingRecHitBuilder * hitBuilder() const { return theTTRHBuilder; }

  bool  hasPredefinedHitErrors() const { return theHasPredefinedHitErrors; }
  float predefinedHitErrorRZ() const { return thePredefinedHitErrorRZ; }
  float predefinedHitErrorRPhi() const { return thePredefinedHitErrorRPhi; }

private:
  SeedingLayerImpl(const SeedingLayerImpl &);

private:
  std::string theName;
  const DetLayer* theLayer;
  const TransientTrackingRecHitBuilder *theTTRHBuilder;
  const HitExtractor * theHitExtractor;
  bool theHasPredefinedHitErrors;
  float thePredefinedHitErrorRZ, thePredefinedHitErrorRPhi;
};




SeedingLayer::SeedingLayer( 
    const std::string & name, 
    const DetLayer* layer, 
    const TransientTrackingRecHitBuilder * hitBuilder,
    const HitExtractor * hitExtractor,
    bool usePredefinedErrors, float hitErrorRZ, float hitErrorRPhi)
{
  SeedingLayerImpl * l = usePredefinedErrors ? 
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

SeedingLayer::Hits SeedingLayer::hits(const edm::Event& ev, const edm::EventSetup& es) const
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
