#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "HitExtractor.h"


using namespace ctfseeding;
using namespace std;


class SeedingLayer::SeedingLayerImpl {
public:
  SeedingLayerImpl(
                const std::string & name, int seqNum,
                const DetLayer* layer,
                const TransientTrackingRecHitBuilder * hitBuilder,
                const HitExtractor * hitExtractor)
  : theName(name),
    theSeqNum(seqNum),
    theLayer(layer),
    theTTRHBuilder(hitBuilder),
    theHitExtractor(hitExtractor) { }

  ~SeedingLayerImpl() {  }

  SeedingLayer::Hits hits(const SeedingLayer &sl, const edm::Event& ev, 
			  const edm::EventSetup& es) const { return theHitExtractor->hits(*theTTRHBuilder, ev, es);  }

  std::string name() const { return theName; }

  int seqNum() const { return theSeqNum; }

  const DetLayer*  detLayer() const { return theLayer; }
  const TransientTrackingRecHitBuilder * hitBuilder() const { return theTTRHBuilder; }

private:
  SeedingLayerImpl(const SeedingLayerImpl &);

private:
  std::string theName;
  int theSeqNum;
  const DetLayer* theLayer;
  const TransientTrackingRecHitBuilder *theTTRHBuilder;
  const HitExtractor * theHitExtractor;
};




SeedingLayer::SeedingLayer( 
    const std::string & name, int seqNum,
    const DetLayer* layer, 
    const TransientTrackingRecHitBuilder * hitBuilder,
    const HitExtractor * hitExtractor)
{
  theImpl = std::make_shared<SeedingLayerImpl> (name,seqNum,layer,hitBuilder,hitExtractor);
}

std::string SeedingLayer::name() const
{
  return theImpl->name();
}

int SeedingLayer::seqNum() const
{
  return theImpl->seqNum();
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
