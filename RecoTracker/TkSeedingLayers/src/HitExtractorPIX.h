#ifndef RecoTracker_TkSeedingLayers_HitExtractorPIX_H
#define RecoTracker_TkSeedingLayers_HitExtractorPIX_H

#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "HitExtractor.h"

#include <string>
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

namespace ctfseeding {
  class HitExtractorPIX final : public HitExtractor {
  public:
    HitExtractorPIX(TrackerDetSide side, int idLayer, const std::string& hitProducer, edm::ConsumesCollector& iC);
    ~HitExtractorPIX() override {}
    HitExtractor::Hits hits(const TkTransientTrackingRecHitBuilder& ttrhBuilder,
                            const edm::Event&,
                            const edm::EventSetup&) const override;
    HitExtractorPIX* clone() const override { return new HitExtractorPIX(*this); }

  private:
    typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > SkipClustersCollection;
    void useSkipClusters_(const edm::InputTag& m, edm::ConsumesCollector& iC) override;

    edm::EDGetTokenT<SkipClustersCollection> theSkipClusters;
    edm::EDGetTokenT<SiPixelRecHitCollection> theHitProducer;
    TrackerDetSide theSide;
    int theIdLayer;
  };
}  // namespace ctfseeding
#endif
