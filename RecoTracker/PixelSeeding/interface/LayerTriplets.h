#ifndef LayerTriplets_H
#define LayerTriplets_H

/** A class grouping pixel layers in pairs and associating a vector
    of layers to each layer pair. The layer pair is used to generate
    hit pairs and the associated vector of layers to generate
    a third hit confirming layer pair
 */

#include <vector>
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

namespace LayerTriplets {
  using Layer = SeedingLayerSetsHits::SeedingLayer;
  using LayerSet = SeedingLayerSetsHits::SeedingLayerSet;
  using LayerSetAndLayers = std::pair<LayerSet, std::vector<Layer> >;

  std::vector<LayerSetAndLayers> layers(const SeedingLayerSetsHits& sets);
}  // namespace LayerTriplets

#endif
