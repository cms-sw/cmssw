#ifndef TkSeedingLayers_SeedingLayerSetsBuilder_H
#define TkSeedingLayers_SeedingLayerSetsBuilder_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include <string>
#include <vector>
namespace edm { class Event; class EventSetup; class ConsumesCollector;}
class TrackerRecoGeometryRecord;
class TransientRecHitRecord;
class TransientTrackingRecHitBuilder;

class SeedingLayerSetsBuilder {

public:

  SeedingLayerSetsBuilder();
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector&& iC);
  ~SeedingLayerSetsBuilder();

  ctfseeding::SeedingLayerSets layers(const edm::EventSetup& es); // only for backwards-compatibility

  bool check(const edm::EventSetup& es);
  void updateEventSetup(const edm::EventSetup& es);

  typedef unsigned short LayerSetIndex;
  unsigned short numberOfLayersInSet() const { return theNumberOfLayersInSet; }
  const std::vector<LayerSetIndex>& layerSetIndices() const { return theLayerSetIndices; }

  unsigned short numberOfLayers() const { return theLayers.size(); }
  const std::vector<std::string>& layerNames() const { return theLayerNames; }
  const std::vector<const DetLayer *>& layerDets() const { return theLayerDets; }
  std::pair<std::vector<unsigned int>, ctfseeding::SeedingLayer::Hits> hits(const edm::Event& ev, const edm::EventSetup& es) const;

private:
  std::vector<std::vector<std::string> > layerNamesInSets(
    const std::vector<std::string> & namesPSet) ;
  edm::ParameterSet layerConfig(const std::string & nameLayer,const edm::ParameterSet& cfg) const;

  edm::ESWatcher<TrackerRecoGeometryRecord> geometryWatcher_;
  edm::ESWatcher<TransientRecHitRecord> trhWatcher_;

  struct LayerSpec { 
    LayerSpec(unsigned short index, const std::string& layerName, const edm::ParameterSet& cfgLayer, edm::ConsumesCollector& iC);
    ~LayerSpec();
    const unsigned short nameIndex;
    std::string pixelHitProducer;
    bool usePixelHitProducer;
    const std::string hitBuilder;

    GeomDetEnumerators::SubDetector subdet;
    ctfseeding::SeedingLayer::Side side;
    int idLayer;
    std::shared_ptr<ctfseeding::HitExtractor> extractor;

    std::string print(const std::vector<std::string>& names) const;
  }; 
  unsigned short theNumberOfLayersInSet;
  std::vector<LayerSetIndex> theLayerSetIndices; // indices to theLayers to form the layer sets
  std::vector<std::string> theLayerNames;
  std::vector<const DetLayer *> theLayerDets;
  std::vector<const TransientTrackingRecHitBuilder *> theTTRHBuilders;
  std::vector<LayerSpec> theLayers;
};
#endif
