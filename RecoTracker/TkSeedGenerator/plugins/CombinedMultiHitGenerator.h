#ifndef CombinedMultiHitGenerator_H
#define CombinedMultiHitGenerator_H

/** A MultiHitGenerator consisting of a set of 
 *  triplet generators of type MultiHitGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */ 

#include <vector>
#include <memory>
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegion;
class MultiHitGeneratorFromPairAndLayers;

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CombinedMultiHitGenerator : public MultiHitGenerator {
public:
  typedef LayerHitMapCache  LayerCacheType;

public:

  CombinedMultiHitGenerator( const edm::ParameterSet& cfg);

  virtual ~CombinedMultiHitGenerator();

  /// from base class
  virtual void hitSets( const TrackingRegion& reg, OrderedMultiHits & result,
      const edm::Event & ev,  const edm::EventSetup& es);

private:
  edm::InputTag theSeedingLayerSrc;

  LayerCacheType            theLayerCache;

  std::unique_ptr<MultiHitGeneratorFromPairAndLayers> theGenerator;
};
#endif
