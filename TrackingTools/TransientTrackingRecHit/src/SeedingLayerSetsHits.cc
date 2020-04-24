#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <limits>
#include <sstream>

SeedingLayerSetsHits::SeedingLayerSetsHits(): nlayers_(0), layerSetIndices_(nullptr), layerNames_(nullptr) {}
SeedingLayerSetsHits::SeedingLayerSetsHits(unsigned short nlayers,
                                           const std::vector<LayerSetIndex> *layerSetIndices,
                                           const std::vector<std::string> *layerNames,
                                           const std::vector<const DetLayer *>& layerDets):
  nlayers_(nlayers),
  layerSetIndices_(layerSetIndices),
  layerNames_(layerNames),
  layerDets_(layerDets)
{}
SeedingLayerSetsHits::~SeedingLayerSetsHits() {
//   std::cout << "deleting eedingLayerSetsHits " << rechits_.size() << std::endl;
}



void SeedingLayerSetsHits::swapHits(std::vector<HitIndex>& layerHitIndices, OwnedHits& hits) {
  layerHitIndices_.swap(layerHitIndices);
  rechits_.swap(hits);
}

SeedingLayerSetsHits::Hits SeedingLayerSetsHits::hits(LayerIndex layerIndex) const {
  HitIndex begin = layerHitIndices_[layerIndex];
  ++layerIndex;
  HitIndex end = layerIndex < layerHitIndices_.size() ? layerHitIndices_[layerIndex] : rechits_.size();

  Hits ret;
  ret.reserve(end-begin);
  std::transform(rechits_.begin()+begin, rechits_.begin()+end, std::back_inserter(ret),[](HitPointer const &p){return p.get();});
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
