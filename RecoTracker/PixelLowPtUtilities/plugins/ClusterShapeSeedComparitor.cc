#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include <cstdio>
#include <cassert>

#include <iostream>

class PixelClusterShapeSeedComparitor : public SeedComparitor {
public:
  PixelClusterShapeSeedComparitor(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC);
  ~PixelClusterShapeSeedComparitor() override;
  void init(const edm::Event &ev, const edm::EventSetup &es) override;
  bool compatible(const SeedingHitSet &hits) const override { return true; }
  bool compatible(const TrajectoryStateOnSurface &, SeedingHitSet::ConstRecHitPointer hit) const override;
  bool compatible(const SeedingHitSet &hits,
                  const GlobalTrajectoryParameters &helixStateAtVertex,
                  const FastHelix &helix) const override;

private:
  bool compatibleHit(const TrackingRecHit &hit, const GlobalVector &direction) const;

  std::string filterName_;
  const edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> clusterShapeHitFilterESToken_;
  const ClusterShapeHitFilter *clusterShapeHitFilter_;
  edm::EDGetTokenT<SiPixelClusterShapeCache> pixelClusterShapeCacheToken_;
  const SiPixelClusterShapeCache *pixelClusterShapeCache_;
  const bool filterAtHelixStage_;
  const bool filterPixelHits_, filterStripHits_;
};

PixelClusterShapeSeedComparitor::PixelClusterShapeSeedComparitor(const edm::ParameterSet &cfg,
                                                                 edm::ConsumesCollector &iC)
    : filterName_(cfg.getParameter<std::string>("ClusterShapeHitFilterName")),
      clusterShapeHitFilterESToken_(iC.esConsumes(edm::ESInputTag("", filterName_))),
      pixelClusterShapeCache_(nullptr),
      filterAtHelixStage_(cfg.getParameter<bool>("FilterAtHelixStage")),
      filterPixelHits_(cfg.getParameter<bool>("FilterPixelHits")),
      filterStripHits_(cfg.getParameter<bool>("FilterStripHits")) {
  if (filterPixelHits_) {
    pixelClusterShapeCacheToken_ =
        iC.consumes<SiPixelClusterShapeCache>(cfg.getParameter<edm::InputTag>("ClusterShapeCacheSrc"));
  }
}

PixelClusterShapeSeedComparitor::~PixelClusterShapeSeedComparitor() {}

void PixelClusterShapeSeedComparitor::init(const edm::Event &ev, const edm::EventSetup &es) {
  clusterShapeHitFilter_ = &es.getData(clusterShapeHitFilterESToken_);
  if (filterPixelHits_) {
    edm::Handle<SiPixelClusterShapeCache> hcache;
    ev.getByToken(pixelClusterShapeCacheToken_, hcache);
    pixelClusterShapeCache_ = hcache.product();
  }
}

bool PixelClusterShapeSeedComparitor::compatible(const TrajectoryStateOnSurface &tsos,
                                                 SeedingHitSet::ConstRecHitPointer hit) const {
  if (filterAtHelixStage_)
    return true;
  assert(hit->isValid() && tsos.isValid());
  return compatibleHit(*hit, tsos.globalDirection());
}

bool PixelClusterShapeSeedComparitor::compatible(const SeedingHitSet &hits,
                                                 const GlobalTrajectoryParameters &helixStateAtVertex,
                                                 const FastHelix &helix) const {
  if (!filterAtHelixStage_)
    return true;

  if (!helix.isValid()              //check still if it's a straight line, which are OK
      && !helix.circle().isLine())  //complain if it's not even a straight line
    edm::LogWarning("InvalidHelix") << "PixelClusterShapeSeedComparitor helix is not valid, result is bad";

  float xc = helix.circle().x0(), yc = helix.circle().y0();

  GlobalPoint vertex = helixStateAtVertex.position();
  GlobalVector momvtx = helixStateAtVertex.momentum();
  float x0 = vertex.x(), y0 = vertex.y();
  for (unsigned int i = 0, n = hits.size(); i < n; ++i) {
    auto const &hit = *hits[i];
    GlobalPoint pos = hit.globalPosition();
    float x1 = pos.x(), y1 = pos.y(), dx1 = x1 - xc, dy1 = y1 - yc;

    // now figure out the proper tangent vector
    float perpx = -dy1, perpy = dx1;
    if (perpx * (x1 - x0) + perpy * (y1 - y0) < 0) {
      perpy = -perpy;
      perpx = -perpx;
    }

    // now normalize (perpx, perpy, 1.0) to momentum (px, py, pz)
    float perp2 = perpx * perpx + perpy * perpy;
    float pmom2 = momvtx.x() * momvtx.x() + momvtx.y() * momvtx.y(), momz2 = momvtx.z() * momvtx.z(),
          mom2 = pmom2 + momz2;
    float perpscale = sqrt(pmom2 / mom2 / perp2), zscale = sqrt((1 - pmom2 / mom2));
    GlobalVector gdir(perpx * perpscale, perpy * perpscale, (momvtx.z() > 0 ? zscale : -zscale));

    if (!compatibleHit(hit, gdir)) {
      return false;  // not yet
    }
  }
  return true;
}

bool PixelClusterShapeSeedComparitor::compatibleHit(const TrackingRecHit &hit, const GlobalVector &direction) const {
  if (hit.geographicalId().subdetId() <= 2) {
    if (!filterPixelHits_)
      return true;
    const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit *>(&hit);
    if (pixhit == nullptr)
      throw cms::Exception("LogicError", "Found a valid hit on the pixel detector which is not a SiPixelRecHit\n");
    //printf("Cheching hi hit on detid %10d, local direction is x = %9.6f, y = %9.6f, z = %9.6f\n", hit.geographicalId().rawId(), direction.x(), direction.y(), direction.z());
    return clusterShapeHitFilter_->isCompatible(*pixhit, direction, *pixelClusterShapeCache_);
  } else {
    if (!filterStripHits_)
      return true;
    const std::type_info &tid = typeid(*&hit);
    if (tid == typeid(SiStripMatchedRecHit2D)) {
      const SiStripMatchedRecHit2D *matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(&hit);
      assert(matchedHit != nullptr);
      return (
          clusterShapeHitFilter_->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), direction) &&
          clusterShapeHitFilter_->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), direction));
    } else if (tid == typeid(SiStripRecHit2D)) {
      const SiStripRecHit2D *recHit = dynamic_cast<const SiStripRecHit2D *>(&hit);
      assert(recHit != nullptr);
      return clusterShapeHitFilter_->isCompatible(*recHit, direction);
    } else if (tid == typeid(ProjectedSiStripRecHit2D)) {
      const ProjectedSiStripRecHit2D *precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(&hit);
      assert(precHit != nullptr);
      return clusterShapeHitFilter_->isCompatible(precHit->originalHit(), direction);
    } else {
      //printf("Questo e' un %s, che ci fo?\n", tid.name());
      return true;
    }
  }
}

DEFINE_EDM_PLUGIN(SeedComparitorFactory, PixelClusterShapeSeedComparitor, "PixelClusterShapeSeedComparitor");
