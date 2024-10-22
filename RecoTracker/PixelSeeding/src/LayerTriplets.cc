#include "RecoTracker/PixelSeeding/interface/LayerTriplets.h"

namespace LayerTriplets {
  std::vector<LayerSetAndLayers> layers(const SeedingLayerSetsHits& sets) {
    std::vector<LayerSetAndLayers> result;
    if (sets.numberOfLayersInSet() < 3)
      return result;

    for (LayerSet set : sets) {
      bool added = false;

      for (auto ir = result.begin(); ir < result.end(); ++ir) {
        const LayerSet& resSet = ir->first;
        if (resSet[0].index() == set[0].index() && resSet[1].index() == set[1].index()) {
          std::vector<Layer>& thirds = ir->second;
          // 3rd layer can already be there if we are dealing with quadruplet layer sets
          auto found =
              std::find_if(thirds.begin(), thirds.end(), [&](const Layer& l) { return l.index() == set[2].index(); });
          if (found == thirds.end())
            thirds.push_back(set[2]);
          added = true;
          break;
        }
      }
      if (!added) {
        LayerSetAndLayers lpl = std::make_pair(set, std::vector<Layer>(1, set[2]));
        result.push_back(lpl);
      }
    }
    return result;
  }
}  // namespace LayerTriplets
