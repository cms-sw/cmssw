#ifndef TkSeedingLayers_SeedingLayerSetsBuilder_H
#define TkSeedingLayers_SeedingLayerSetsBuilder_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include <string>
#include <vector>
namespace edm { class EventSetup; }
namespace edm { class ParameterSet; }

namespace ctfseeding {
class SeedingLayerSetsBuilder {
public:
  SeedingLayerSetsBuilder(){}
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg);
  SeedingLayerSets layers(const edm::EventSetup& es) const; 
private:
  void init(const std::vector<std::string> & layerNames);
private:
  std::vector<std::vector<std::string> > theLayersInSetNames; 
};
}
#endif
