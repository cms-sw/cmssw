#ifndef HitPairGeneratorFromLayerPair_h
#define HitPairGeneratorFromLayerPair_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include <optional>

class DetLayer;
class IdealMagneticFieldRecord;
class MagneticField;
class MultipleScatteringParametrisationMaker;
class TrackerMultipleScatteringRecord;
class TrackingRegion;

class HitPairGeneratorFromLayerPair {
public:
  typedef LayerHitMapCache LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;

  HitPairGeneratorFromLayerPair(edm::ConsumesCollector iC,
                                unsigned int inner,
                                unsigned int outer,
                                LayerCacheType* layerCache,
                                unsigned int max = 0);

  ~HitPairGeneratorFromLayerPair();

  std::optional<HitDoublets> doublets(const TrackingRegion& reg,
                                      const edm::Event& ev,
                                      const edm::EventSetup& es,
                                      Layers layers) {
    assert(theLayerCache);
    return doublets(reg, ev, es, layers, *theLayerCache);
  }
  std::optional<HitDoublets> doublets(const TrackingRegion& reg,
                                      const edm::Event& ev,
                                      const edm::EventSetup& es,
                                      const Layer& innerLayer,
                                      const Layer& outerLayer) {
    assert(theLayerCache);
    return doublets(reg, ev, es, innerLayer, outerLayer, *theLayerCache);
  }
  std::optional<HitDoublets> doublets(const TrackingRegion& reg,
                                      const edm::Event& ev,
                                      const edm::EventSetup& es,
                                      Layers layers,
                                      LayerCacheType& layerCache) {
    Layer innerLayerObj = innerLayer(layers);
    Layer outerLayerObj = outerLayer(layers);
    return doublets(reg, ev, es, innerLayerObj, outerLayerObj, layerCache);
  }
  std::optional<HitDoublets> doublets(const TrackingRegion& reg,
                                      const edm::Event& ev,
                                      const edm::EventSetup& es,
                                      const Layer& innerLayer,
                                      const Layer& outerLayer,
                                      LayerCacheType& layerCache);

  bool hitPairs(
      const TrackingRegion& reg, OrderedHitPairs& prs, const edm::Event& ev, const edm::EventSetup& es, Layers layers);
  static bool doublets(const TrackingRegion& region,
                       const DetLayer& innerHitDetLayer,
                       const DetLayer& outerHitDetLayer,
                       const RecHitsSortedInPhi& innerHitsMap,
                       const RecHitsSortedInPhi& outerHitsMap,
                       const MagneticField& field,
                       const MultipleScatteringParametrisationMaker& msmaker,
                       const unsigned int theMaxElement,
                       HitDoublets& result);

  Layer innerLayer(const Layers& layers) const { return layers[theInnerLayer]; }
  Layer outerLayer(const Layers& layers) const { return layers[theOuterLayer]; }

private:
  LayerCacheType* theLayerCache;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  const edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> theMSMakerToken;
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
  const unsigned int theMaxElement;
};

#endif
