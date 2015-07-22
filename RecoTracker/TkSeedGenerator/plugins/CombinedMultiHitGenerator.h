#ifndef CombinedMultiHitGenerator_H
#define CombinedMultiHitGenerator_H

/** A MultiHitGenerator consisting of a set of 
 *  triplet generators of type MultiHitGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */ 

#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include<memory>

class TrackingRegion;
class dso_hidden SeedingLayerSetsHits;
// class MultiHitGeneratorFromPairAndLayers;

namespace edm { class Event; }
namespace edm { class EventSetup; }

class dso_hidden CombinedMultiHitGenerator final : public MultiHitGenerator {
public:
  typedef LayerHitMapCache  LayerCacheType;

public:

  CombinedMultiHitGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~CombinedMultiHitGenerator();

  /// from base class
  virtual void hitSets( const TrackingRegion& reg, OrderedMultiHits & result,
      const edm::Event & ev,  const edm::EventSetup& es);

  virtual void clear() override {
    MultiHitGenerator::clear();
    theGenerator->clear();
  }

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;

  LayerCacheType            theLayerCache;

  std::unique_ptr<MultiHitGeneratorFromPairAndLayers> theGenerator;
};
#endif
