#include "RecoTracker/SeedingLayerSetsHits/interface/SeedingLayerSetsHits.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <limits>
#include <sstream>

SeedingLayerSetsHits::SeedingLayerSetsHits(): SeedingLayerSetsHits(0) {}
SeedingLayerSetsHits::SeedingLayerSetsHits(unsigned int nlayers): nlayers_(nlayers) {}
SeedingLayerSetsHits::~SeedingLayerSetsHits() {}

std::pair<SeedingLayerSetsHits::LayerIndex, bool> SeedingLayerSetsHits::insertLayer(const std::string& layerName, const DetLayer *layerDet) {
  std::pair<LayerIndex, bool> index = insertLayer_(layerName, layerDet);
  layerSetIndices_.push_back(index.first);
  return index;
}

std::pair<SeedingLayerSetsHits::LayerIndex, bool> SeedingLayerSetsHits::insertLayer_(const std::string& layerName, const DetLayer *layerDet) {
  auto found = std::find(layerNames_.begin(), layerNames_.end(), layerName);
  // insert if not found
  if(found == layerNames_.end()) {
    layerNames_.push_back(layerName);
    layerDets_.push_back(layerDet);
    const auto max = std::numeric_limits<unsigned int>::max();
    layerHitRanges_.emplace_back(max, max);
    //std::cout << "Inserted layer " << layerName << " to index " << layerNames_.size()-1 << std::endl;
    return std::make_pair(layerNames_.size()-1, true);
  }
  //std::cout << "Encountered layer " << layerName << " index " << found-layerNames_.begin() << std::endl;
  return std::make_pair(found-layerNames_.begin(), false);
}

void SeedingLayerSetsHits::insertLayerHits(LayerIndex layerIndex, const Hits& hits) {
  assert(layerIndex < layerHitRanges_.size());
  Range& range = layerHitRanges_[layerIndex];
  range.first = rechits_.size();
  std::copy(hits.begin(), hits.end(), std::back_inserter(rechits_));
  range.second = rechits_.size();

  //std::cout << "  added " << hits.size() << " hits to layer " << layerIndex << " range " << range.first << " " << range.second << std::endl;
}

SeedingLayerSetsHits::Hits SeedingLayerSetsHits::hits(LayerIndex layerIndex) const {
  const Range& range = layerHitRanges_[layerIndex];

  Hits ret;
  ret.reserve(range.second-range.first);
  std::copy(rechits_.begin()+range.first, rechits_.begin()+range.second, std::back_inserter(ret));
  return ret;
}


void SeedingLayerSetsHits::print() const {
  std::stringstream ss;
  ss << "SeedingLayerSetsHits with " << numberOfLayersInSet() << " layers in each LayerSets, LayerSets has " << size() << " items\n";
  for(LayerSetIndex iLayers=0; iLayers<size(); ++iLayers) {
    ss << " " << iLayers << ": ";
    SeedingLayerSet layers = operator[](iLayers);
    for(unsigned iLayer=0; iLayer<layers.size(); ++iLayer) {
      SeedingLayer layer = layers[iLayer];
      ss << layer.name() << " (" << layer.index() << ", nhits " << layer.hits().size() << ") ";
    }
    ss << "\n";
  }
  LogDebug("SeedingLayerSetsHits") << ss.str();
}
