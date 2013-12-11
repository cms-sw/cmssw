#ifndef CombinedMultiHitGenerator_H
#define CombinedMultiHitGenerator_H

/** A MultiHitGenerator consisting of a set of 
 *  triplet generators of type MultiHitGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */ 

#include <vector>
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegion;
class MultiHitGeneratorFromPairAndLayers;
namespace ctfseeding { class SeedingLayer;}

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CombinedMultiHitGenerator : public MultiHitGenerator {
public:
  typedef LayerHitMapCache  LayerCacheType;

public:

  CombinedMultiHitGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~CombinedMultiHitGenerator();

  /// from base class
  virtual void hitSets( const TrackingRegion& reg, OrderedMultiHits & result,
      const edm::Event & ev,  const edm::EventSetup& es);

private:
  void init(const edm::ParameterSet & cfg, const edm::EventSetup& es);

  mutable bool initialised;

  edm::ParameterSet         theConfig;
  LayerCacheType            theLayerCache;

  typedef std::vector<MultiHitGeneratorFromPairAndLayers* > GeneratorContainer;
  GeneratorContainer        theGenerators;
};
#endif
