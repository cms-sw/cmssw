#ifndef RecoTracker_TkSeedingLayers_HitExtractorPIX_H
#define RecoTracker_TkSeedingLayers_HitExtractorPIX_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "HitExtractor.h"

#include <string>
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

namespace ctfseeding {
class HitExtractorPIX : public HitExtractor {
public:
  HitExtractorPIX( SeedingLayer::Side & side, int idLayer, const std::string & hitProducer, edm::ConsumesCollector& iC);
  virtual ~HitExtractorPIX(){}
  virtual HitExtractor::Hits hits(const TransientTrackingRecHitBuilder &ttrhBuilder, const edm::Event& , const edm::EventSetup& ) const;
  virtual HitExtractorPIX * clone() const { return new HitExtractorPIX(*this); }

private:
  typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > SkipClustersCollection;
  void useSkipClusters_(const edm::InputTag & m, edm::ConsumesCollector& iC) override;

  edm::EDGetTokenT<SkipClustersCollection> theSkipClusters;
  edm::EDGetTokenT<SiPixelRecHitCollection> theHitProducer;
  SeedingLayer::Side theSide;
  int theIdLayer;
};
}
#endif
