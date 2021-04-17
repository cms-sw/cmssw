#ifndef RecoTracker_TkSeedingLayers_HitExtractorSTRP_H
#define RecoTracker_TkSeedingLayers_HitExtractorSTRP_H

#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HitExtractor.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"

#include <vector>
#include <tuple>
class DetLayer;

namespace edm {
  template <typename T>
  class ContainerMask;
}

namespace ctfseeding {

  class HitExtractorSTRP final : public HitExtractor {
  public:
    typedef SiStripRecHit2D::ClusterRef SiStripClusterRef;

    HitExtractorSTRP(GeomDetEnumerators::SubDetector subdet, TrackerDetSide side, int idLayer, float iminGoodCharge);
    ~HitExtractorSTRP() override {}

    HitExtractor::Hits hits(const TkTransientTrackingRecHitBuilder& ttrhBuilder,
                            const edm::Event&,
                            const edm::EventSetup&) const override;
    HitExtractorSTRP* clone() const override { return new HitExtractorSTRP(*this); }

    void useMatchedHits(const edm::InputTag& m, edm::ConsumesCollector& iC) {
      hasMatchedHits = true;
      theMatchedHits = iC.consumes<SiStripMatchedRecHit2DCollection>(m);
    }
    void useRPhiHits(const edm::InputTag& m, edm::ConsumesCollector& iC) {
      hasRPhiHits = true;
      theRPhiHits = iC.consumes<SiStripRecHit2DCollection>(m);
    }
    void useStereoHits(const edm::InputTag& m, edm::ConsumesCollector& iC) {
      hasStereoHits = true;
      theStereoHits = iC.consumes<SiStripRecHit2DCollection>(m);
    }

    void useVectorHits(const edm::InputTag& m, edm::ConsumesCollector& iC) {
      hasVectorHits = true;
      theVectorHits = iC.consumes<VectorHitCollection>(m);
    }
    void useRingSelector(int minRing, int maxRing);
    void useSimpleRphiHitsCleaner(bool use) { hasSimpleRphiHitsCleaner = use; }

    void cleanedOfClusters(const TkTransientTrackingRecHitBuilder& ttrhBuilder,
                           const edm::Event& ev,
                           HitExtractor::Hits& hits,
                           bool matched,
                           unsigned int cleanFrom = 0) const;

    std::pair<bool, ProjectedSiStripRecHit2D*> skipThis(
        const TkTransientTrackingRecHitBuilder& ttrhBuilder,
        TkHitRef matched,
        edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >& stripClusterMask) const;

    bool skipThis(DetId id,
                  OmniClusterRef const& clus,
                  edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >& stripClusterMask) const;

    void setNoProjection() { failProjection = true; }
    void setMinAbsZ(double minZToSet) { minAbsZ = minZToSet; }

    bool useRingSelector() const { return hasRingSelector; }
    std::tuple<int, int> getMinMaxRing() const { return std::make_tuple(theMinRing, theMaxRing); }

  private:
    bool ringRange(int ring) const;

    typedef edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > SkipClustersCollection;
    typedef edm::ContainerMask<Phase2TrackerCluster1DCollectionNew> SkipPhase2ClustersCollection;
    void useSkipClusters_(const edm::InputTag& m, edm::ConsumesCollector& iC) override;

  private:
    const GeomDetEnumerators::SubDetector theLayerSubDet;
    TrackerDetSide theSide;
    int theIdLayer;
    double minAbsZ;
    int theMinRing, theMaxRing;
    edm::EDGetTokenT<SkipClustersCollection> theSkipClusters;
    edm::EDGetTokenT<SkipPhase2ClustersCollection> theSkipPhase2Clusters;
    edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> theMatchedHits;
    edm::EDGetTokenT<SiStripRecHit2DCollection> theRPhiHits;
    edm::EDGetTokenT<SiStripRecHit2DCollection> theStereoHits;
    edm::EDGetTokenT<VectorHitCollection> theVectorHits;
    bool hasMatchedHits;
    bool hasRPhiHits;
    bool hasStereoHits;
    bool hasVectorHits;
    bool hasRingSelector;
    bool hasSimpleRphiHitsCleaner;
    bool failProjection;
  };

}  // namespace ctfseeding
#endif
