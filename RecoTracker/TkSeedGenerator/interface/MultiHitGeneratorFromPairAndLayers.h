#ifndef MultiHitGeneratorFromPairAndLayers_H
#define MultiHitGeneratorFromPairAndLayers_H

/** A MultiHitGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/OrderedMultiHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ParameterSetDescription;
}  // namespace edm
class TrackingRegion;
class HitPairGeneratorFromLayerPair;

class MultiHitGeneratorFromPairAndLayers {
public:
  typedef LayerHitMapCache LayerCacheType;

  explicit MultiHitGeneratorFromPairAndLayers(const edm::ParameterSet& pset);
  virtual ~MultiHitGeneratorFromPairAndLayers();

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  virtual void initES(const edm::EventSetup& es) = 0;

  void init(std::unique_ptr<HitPairGeneratorFromLayerPair>&& pairGenerator, LayerCacheType* layerCache);

  virtual void hitSets(const TrackingRegion& region,
                       OrderedMultiHits& trs,
                       const edm::Event& ev,
                       const edm::EventSetup& es,
                       SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                       std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers) = 0;

  virtual void hitTriplets(const TrackingRegion& region,
                           OrderedMultiHits& result,
                           const edm::EventSetup& es,
                           const HitDoublets& doublets,
                           const RecHitsSortedInPhi** thirdHitMap,
                           const std::vector<const DetLayer*>& thirdLayerDetLayer,
                           const int nThirdLayers) = 0;

  const HitPairGeneratorFromLayerPair& pairGenerator() const { return *thePairGenerator; }

  void clear();

protected:
  using cacheHitPointer = std::unique_ptr<BaseTrackerRecHit>;
  using cacheHits = std::vector<cacheHitPointer>;
  cacheHits cache;  // ownes what is by reference above...

  std::unique_ptr<HitPairGeneratorFromLayerPair> thePairGenerator;
  LayerCacheType* theLayerCache;
  const unsigned int theMaxElement;
};
#endif
