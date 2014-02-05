#ifndef MultiHitGeneratorFromPairAndLayers_H
#define MultiHitGeneratorFromPairAndLayers_H

/** A MultiHitGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include <vector>
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"

namespace edm { class EventSetup; }

class MultiHitGeneratorFromPairAndLayers : public MultiHitGenerator {

public:
  typedef LayerHitMapCache  LayerCacheType;

  virtual ~MultiHitGeneratorFromPairAndLayers() {}
  virtual void init( const HitPairGenerator & pairs, 
		     const std::vector<ctfseeding::SeedingLayer>& layers, 
		     LayerCacheType* layerCache,  
		     const edm::EventSetup& es) = 0; 
};
#endif


