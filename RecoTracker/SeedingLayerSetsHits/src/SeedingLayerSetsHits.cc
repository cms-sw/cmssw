#include "RecoTracker/SeedingLayerSetsHits/interface/SeedingLayerSetsHits.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <limits>
#include <sstream>

SeedingLayerSetsHits::SeedingLayerSetsHits(): nlayers_(0), layerSetIndices_(nullptr), layerNames_(nullptr) {}
SeedingLayerSetsHits::SeedingLayerSetsHits(unsigned short nlayers, const std::vector<LayerSetIndex> *layerSetIndices, const std::vector<std::string> *layerNames, const std::vector<const DetLayer *>& layerDets):
  nlayers_(nlayers),
  layerSetIndices_(layerSetIndices),
  layerHitRanges_(layerNames->size(), std::make_pair(std::numeric_limits<unsigned int>::max(), std::numeric_limits<unsigned int>::max())),
  layerNames_(layerNames),
  layerDets_(layerDets)
{}
SeedingLayerSetsHits::~SeedingLayerSetsHits() {}

void SeedingLayerSetsHits::setHits(LayerIndex layerIndex, const Hits& hits) {
  assert(layerIndex < layerHitRanges_.size());
  Range& range = layerHitRanges_[layerIndex];
  range.first = rechits_.size();
  rechits_.reserve(rechits_.size()+hits.size());
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
