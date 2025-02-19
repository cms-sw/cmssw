#ifndef TkSeedingLayers_SeedingLayerSets_H
#define TkSeedingLayers_SeedingLayerSets_H

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

namespace ctfseeding {
  typedef std::vector<SeedingLayer>               SeedingLayers;
  typedef std::vector<std::vector<SeedingLayer> > SeedingLayerSets;
}

  
#endif
