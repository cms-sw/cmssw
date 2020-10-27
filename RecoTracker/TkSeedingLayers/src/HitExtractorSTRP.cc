#include "HitExtractorSTRP.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/ContainerMask.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

#include <tuple>

#include <iostream>

using namespace ctfseeding;
using namespace std;
using namespace edm;

HitExtractorSTRP::HitExtractorSTRP(GeomDetEnumerators::SubDetector subdet,
                                   TrackerDetSide side,
                                   int idLayer,
                                   float iminGoodCharge)
    : theLayerSubDet(subdet),
      theSide(side),
      theIdLayer(idLayer),
      minAbsZ(0),
      theMinRing(1),
      theMaxRing(0),
      hasMatchedHits(false),
      hasRPhiHits(false),
      hasStereoHits(false),
      hasVectorHits(false),
      hasRingSelector(false),
      hasSimpleRphiHitsCleaner(true) {
  minGoodCharge = iminGoodCharge;
  if (minGoodCharge > 0)
    skipClusters = true;
}

void HitExtractorSTRP::useSkipClusters_(const edm::InputTag& m, edm::ConsumesCollector& iC) {
  theSkipClusters = iC.consumes<SkipClustersCollection>(m);
  theSkipPhase2Clusters = iC.consumes<SkipPhase2ClustersCollection>(m);
}

void HitExtractorSTRP::useRingSelector(int minRing, int maxRing) {
  hasRingSelector = true;
  theMinRing = minRing;
  theMaxRing = maxRing;
}

bool HitExtractorSTRP::ringRange(int ring) const {
  if (!hasRingSelector)
    return true;
  return (ring >= theMinRing) & (ring <= theMaxRing);
}

bool HitExtractorSTRP::skipThis(
    DetId id,
    OmniClusterRef const& clus,
    edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >& stripClusterMask) const {
  if (maskCluster && (stripClusterMask->mask(clus.key())))
    return true;

  if UNLIKELY (minGoodCharge <= 0)
    return false;
  return siStripClusterTools::chargePerCM(id, *clus.cluster_strip()) <= minGoodCharge;
}

std::pair<bool, ProjectedSiStripRecHit2D*> HitExtractorSTRP::skipThis(
    const TkTransientTrackingRecHitBuilder& ttrhBuilder,
    TkHitRef matched,
    edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > >& stripClusterMask) const {
  const SiStripMatchedRecHit2D& hit = (SiStripMatchedRecHit2D const&)(matched);

  assert(dynamic_cast<SiStripMatchedRecHit2D const*>(&matched));

  auto id = hit.geographicalId();
  ProjectedSiStripRecHit2D* replaceMe = nullptr;
  bool rejectSt = skipThis(id, hit.stereoClusterRef(), stripClusterMask);
  bool rejectMono = skipThis(id, hit.monoClusterRef(), stripClusterMask);

  if ((!rejectSt) & (!rejectMono)) {
    // keepit
    return std::make_pair(false, replaceMe);
  }

  if (failProjection || (rejectSt & rejectMono)) {
    //only skip if both hits are done
    return std::make_pair(true, replaceMe);
  }

  // replace with one

  auto cloner = ttrhBuilder.cloner();
  replaceMe = cloner.project(hit, rejectSt, TrajectoryStateOnSurface()).release();
  if (rejectSt)
    LogDebug("HitExtractorSTRP") << "a matched hit is partially masked, and the mono hit got projected onto: "
                                 << replaceMe->geographicalId().rawId() << " key: " << hit.monoClusterRef().key();
  else
    LogDebug("HitExtractorSTRP") << "a matched hit is partially masked, and the stereo hit got projected onto: "
                                 << replaceMe->geographicalId().rawId() << " key: " << hit.stereoClusterRef().key();

  return std::make_pair(true, replaceMe);
}

void HitExtractorSTRP::cleanedOfClusters(const TkTransientTrackingRecHitBuilder& ttrhBuilder,
                                         const edm::Event& ev,
                                         HitExtractor::Hits& hits,
                                         bool matched,
                                         unsigned int cleanFrom) const {
  unsigned int skipped = 0;
  unsigned int projected = 0;
  if (hasMatchedHits || hasRPhiHits || hasStereoHits) {
    LogTrace("HitExtractorSTRP") << "getting " << hits.size() << " strip hit in input.";
    edm::Handle<SkipClustersCollection> stripClusterMask;
    if (maskCluster)
      ev.getByToken(theSkipClusters, stripClusterMask);
    for (unsigned int iH = cleanFrom; iH < hits.size(); ++iH) {
      assert(hits[iH]->isValid());
      auto id = hits[iH]->geographicalId();
      if (matched) {
        auto [replace, replaceMe] = skipThis(ttrhBuilder, *hits[iH], stripClusterMask);
        if (replace) {
          if (!replaceMe) {
            LogTrace("HitExtractorSTRP") << "skipping a matched hit on :" << hits[iH]->geographicalId().rawId();
            skipped++;
          } else {
            projected++;
          }
          hits[iH].reset(replaceMe);
          if (replaceMe == nullptr)
            assert(hits[iH].empty());
          else
            assert(hits[iH].isOwn());
        }
      } else if (skipThis(id, hits[iH]->firstClusterRef(), stripClusterMask)) {
        LogTrace("HitExtractorSTRP") << "skipping a hit on :" << hits[iH]->geographicalId().rawId() << " key: ";
        skipped++;
        hits[iH].reset();
      }
    }
  }
  if (hasVectorHits) {
    LogTrace("HitExtractorSTRP") << "getting " << hits.size() << " vector hit in input.";
    edm::Handle<SkipPhase2ClustersCollection> ph2ClusterMask;
    if (maskCluster)
      ev.getByToken(theSkipPhase2Clusters, ph2ClusterMask);
    for (unsigned int iH = cleanFrom; iH < hits.size(); ++iH) {
      LogTrace("HitExtractorSTRP") << "analizing hit on :" << hits[iH]->geographicalId().rawId();
      assert(hits[iH]->isValid());
      const VectorHit& vhit = dynamic_cast<VectorHit const&>(*hits[iH]);
      LogTrace("HitExtractorSTRP") << " key lower: " << vhit.lowerClusterRef().key()
                                   << " and key upper: " << vhit.upperClusterRef().key();
      LogTrace("HitExtractorSTRP") << " key lower: " << hits[iH]->firstClusterRef().key();

      //FIXME:: introduce a "projected" version later?
      if (maskCluster &&
          (ph2ClusterMask->mask(vhit.lowerClusterRef().key()) || ph2ClusterMask->mask(vhit.upperClusterRef().key()))) {
        LogTrace("HitExtractorSTRP") << "skipping a vector hit on :" << hits[iH]->geographicalId().rawId()
                                     << " key lower: " << vhit.lowerClusterRef().key()
                                     << " and key upper: " << vhit.upperClusterRef().key();
        skipped++;
        hits[iH].reset();
      }
    }
  }

  //  remove empty elements...
  auto last = std::remove_if(hits.begin() + cleanFrom, hits.end(), [](HitPointer const& p) { return p.empty(); });
  hits.resize(last - hits.begin());

  LogTrace("HitExtractorSTRP") << "skipped :" << skipped << " rechits because of clusters and projected: " << projected;
}

HitExtractor::Hits HitExtractorSTRP::hits(const TkTransientTrackingRecHitBuilder& ttrhBuilder,
                                          const edm::Event& ev,
                                          const edm::EventSetup& es) const {
  LogDebug("HitExtractorSTRP") << "HitExtractorSTRP::hits";
  HitExtractor::Hits result;
  unsigned int cleanFrom = 0;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //
  // TIB
  //
  if (theLayerSubDet == GeomDetEnumerators::TIB) {
    LogTrace("HitExtractorSTRP") << "Getting hits into the TIB";
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByToken(theMatchedHits, matchedHits);
      if (skipClusters)
        cleanFrom = result.size();
      range2SeedingHits(*matchedHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, true, cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByToken(theRPhiHits, rphiHits);
      if (hasMatchedHits) {
        if (!hasSimpleRphiHitsCleaner) {  // this is a brutal "cleaning". Add something smarter in the future
          if (skipClusters)
            cleanFrom = result.size();
          range2SeedingHits(*rphiHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
          if (skipClusters)
            cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
        }
      } else {
        if (skipClusters)
          cleanFrom = result.size();
        range2SeedingHits(*rphiHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
        if (skipClusters)
          cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByToken(theStereoHits, stereoHits);
      if (skipClusters)
        cleanFrom = result.size();
      range2SeedingHits(*stereoHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
    if (hasVectorHits) {
      LogError("HitExtractorSTRP") << "TIB is not supposed to be in Phase2 TRK detector configuration. What follows "
                                      "have never been checked before! ";
      auto const& vectorHits = ev.get(theVectorHits);
      if (skipClusters)
        cleanFrom = result.size();
      range2SeedingHits(vectorHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }

  }

  //
  // TID
  //
  else if (theLayerSubDet == GeomDetEnumerators::TID) {
    LogTrace("HitExtractorSTRP") << "Getting hits into the TID";
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByToken(theMatchedHits, matchedHits);
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tidDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
      for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tidRing(it->detId());
        if (!ringRange(ring))
          continue;
        for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end;
             ++hit) {
          result.emplace_back(*hit);
        }
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, true, cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByToken(theRPhiHits, rphiHits);
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tidDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
      for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tidRing(it->detId());
        if (!ringRange(ring))
          continue;
        if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner)
          continue;  // this is a brutal "cleaning". Add something smarter in the future
        for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
          result.emplace_back(*hit);
        }
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByToken(theStereoHits, stereoHits);
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tidDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
      for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tidRing(it->detId());
        if (!ringRange(ring))
          continue;
        for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
          result.emplace_back(*hit);
        }
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
    if (hasVectorHits) {
      LogTrace("HitExtractorSTRP") << "Getting vector hits for IdLayer " << theIdLayer;
      auto const& vectorHits = ev.get(theVectorHits);
      //FIXME: check the skipClusters with VHits
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tidDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      VectorHitCollection::Range range = vectorHits.equal_range(getter.first, getter.second);
      for (VectorHitCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tidRing(it->detId());
        if (!ringRange(ring))
          continue;
        for (VectorHitCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
          result.emplace_back(*hit);
        }
      }
      LogTrace("HitExtractorSTRP") << "result size value:" << result.size();
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
  }
  //
  // TOB
  //
  else if (theLayerSubDet == GeomDetEnumerators::TOB) {
    LogTrace("HitExtractorSTRP") << "Getting hits into the TOB";
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByToken(theMatchedHits, matchedHits);
      if (skipClusters)
        cleanFrom = result.size();
      if (minAbsZ > 0.) {
        auto getter = tTopo->tobDetIdLayerComparator(theIdLayer);
        SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
        for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
          for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end;
               ++hit) {
            if (fabs(hit->globalPosition().z()) >= minAbsZ)
              result.emplace_back(*hit);
          }
        }
      } else {
        range2SeedingHits(*matchedHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, true, cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByToken(theRPhiHits, rphiHits);
      if (hasMatchedHits) {
        if (!hasSimpleRphiHitsCleaner) {  // this is a brutal "cleaning". Add something smarter in the future
          if (skipClusters)
            cleanFrom = result.size();
          range2SeedingHits(*rphiHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
          if (skipClusters)
            cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
        }
      } else {
        if (skipClusters)
          cleanFrom = result.size();
        range2SeedingHits(*rphiHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
        if (skipClusters)
          cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByToken(theStereoHits, stereoHits);
      if (skipClusters)
        cleanFrom = result.size();
      range2SeedingHits(*stereoHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
    if (hasVectorHits) {
      LogTrace("HitExtractorSTRP") << "Getting vector hits for IdLayer " << theIdLayer;
      edm::Handle<VectorHitCollection> vectorHits;
      ev.getByToken(theVectorHits, vectorHits);
      //FIXME: check the skipClusters with VHits
      if (skipClusters)
        cleanFrom = result.size();
      range2SeedingHits(*vectorHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }

  }

  //
  // TEC
  //
  else if (theLayerSubDet == GeomDetEnumerators::TEC) {
    LogTrace("HitExtractorSTRP") << "Getting hits into the TEC";
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByToken(theMatchedHits, matchedHits);
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tecDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
      for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tecRing(it->detId());
        if (!ringRange(ring))
          continue;
        for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end;
             ++hit) {
          result.emplace_back(*hit);
        }
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, true, cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByToken(theRPhiHits, rphiHits);
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tecDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
      for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tecRing(it->detId());
        if (!ringRange(ring))
          continue;
        if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner)
          continue;  // this is a brutal "cleaning". Add something smarter in the future
        for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
          result.emplace_back(*hit);
        }
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByToken(theStereoHits, stereoHits);
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tecDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
      for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tecRing(it->detId());
        if (!ringRange(ring))
          continue;
        for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
          result.emplace_back(*hit);
        }
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
    if (hasVectorHits) {
      LogError("HitExtractorSTRP") << "TEC is not supposed to be in Phase2 TRK detector configuration. What follows "
                                      "have never been checked before! ";
      edm::Handle<VectorHitCollection> vectorHits;
      ev.getByToken(theVectorHits, vectorHits);
      if (skipClusters)
        cleanFrom = result.size();
      auto getter = tTopo->tidDetIdWheelComparator(static_cast<unsigned int>(theSide), theIdLayer);
      VectorHitCollection::Range range = vectorHits->equal_range(getter.first, getter.second);
      for (VectorHitCollection::const_iterator it = range.first; it != range.second; ++it) {
        int ring = tTopo->tidRing(it->detId());
        if (!ringRange(ring))
          continue;
        for (VectorHitCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
          result.emplace_back(*hit);
        }
      }
      if (skipClusters)
        cleanedOfClusters(ttrhBuilder, ev, result, false, cleanFrom);
    }
  }

  LogDebug("HitExtractorSTRP") << " giving: " << result.size() << " out for charge cut " << minGoodCharge;
  return result;
}
