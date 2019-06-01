#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <limits>
#include <sstream>

SeedingLayerSetsHits::SeedingLayerSetsHits(unsigned short nlayers,
                                           const std::vector<LayerSetIndex> *layerSetIndices,
                                           const std::vector<std::string> *layerNames,
                                           const std::vector<const DetLayer *> *layerDets)
    : nlayers_(nlayers), layerSetIndices_(layerSetIndices), layerNames_(layerNames), layerDets_(layerDets) {
  layerHitIndices_.reserve(layerNames->size());
}

void SeedingLayerSetsHits::shrink_to_fit() { rechits_.shrink_to_fit(); }

void SeedingLayerSetsHits::addHits(LayerIndex layerIndex, OwnedHits &&hits) {
  if (layerIndex != layerHitIndices_.size()) {
    throw cms::Exception("Assert")
        << "SeedingLayerSetsHits::addHits() must be called in the order of the layers, got layer " << layerIndex
        << " while was expecting " << layerHitIndices_.size();
  }

  layerHitIndices_.push_back(rechits_.size());
  std::move(hits.begin(), hits.end(), std::back_inserter(rechits_));
}

SeedingLayerSetsHits::Hits SeedingLayerSetsHits::hits(LayerIndex layerIndex) const {
  HitIndex begin = layerHitIndices_[layerIndex];
  ++layerIndex;
  HitIndex end = layerIndex < layerHitIndices_.size() ? layerHitIndices_[layerIndex] : rechits_.size();

  Hits ret;
  ret.reserve(end - begin);
  std::transform(rechits_.begin() + begin, rechits_.begin() + end, std::back_inserter(ret), [](HitPointer const &p) {
    return p.get();
  });
  return ret;
}

void SeedingLayerSetsHits::print() const {
  std::stringstream ss;
  ss << "SeedingLayerSetsHits with " << numberOfLayersInSet() << " layers in each LayerSets, LayerSets has " << size()
     << " items\n";
  for (LayerSetIndex iLayers = 0; iLayers < size(); ++iLayers) {
    ss << " " << iLayers << ": ";
    SeedingLayerSet layers = operator[](iLayers);
    for (unsigned iLayer = 0; iLayer < layers.size(); ++iLayer) {
      SeedingLayer layer = layers[iLayer];
      ss << layer.name() << " (" << layer.index() << ", nhits " << layer.hits().size() << ") ";
    }
    ss << "\n";
  }
  LogDebug("SeedingLayerSetsHits") << ss.str();
}
