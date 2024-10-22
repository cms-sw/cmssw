#ifndef RecoTracker_PixelSeeding_plugins_CombinedHitTripletGenerator_h
#define RecoTracker_PixelSeeding_plugins_CombinedHitTripletGenerator_h

/** A HitTripletGenerator consisting of a set of 
 *  triplet generators of type HitTripletGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */

#include <vector>
#include <memory>
#include "RecoTracker/PixelSeeding/interface/HitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <string>
#include <memory>

class TrackingRegion;
class HitTripletGeneratorFromPairAndLayers;
class SeedingLayerSetsHits;

namespace edm {
  class Event;
}
namespace edm {
  class EventSetup;
}

class CombinedHitTripletGenerator : public HitTripletGenerator {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitTripletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~CombinedHitTripletGenerator() override;

  /// from base class
  void hitTriplets(const TrackingRegion& reg,
                   OrderedHitTriplets& triplets,
                   const edm::Event& ev,
                   const edm::EventSetup& es) override;

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;

  LayerCacheType theLayerCache;

  std::unique_ptr<HitTripletGeneratorFromPairAndLayers> theGenerator;
};
#endif
