#ifndef TkSeedingLayers_SeedingLayerSetsBuilder_H
#define TkSeedingLayers_SeedingLayerSetsBuilder_H

#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"

#include <string>
#include <vector>
namespace edm { class Event; class EventSetup; class ConsumesCollector;}
namespace ctfseeding {class HitExtractor; }
class TrackerRecoGeometryRecord;
class TransientRecHitRecord;
class TransientTrackingRecHitBuilder;
class DetLayer;

class SeedingLayerSetsBuilder {

public:

  SeedingLayerSetsBuilder() = default;
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector&& iC);
  ~SeedingLayerSetsBuilder();

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  unsigned short numberOfLayers() const { return theLayers.size(); }
  std::unique_ptr<SeedingLayerSetsHits> hits(const edm::Event& ev, const edm::EventSetup& es);

  using SeedingLayerId = std::tuple<GeomDetEnumerators::SubDetector, TrackerDetSide, int>;
  static SeedingLayerId nameToEnumId(const std::string& name);
  static std::vector<std::vector<std::string> > layerNamesInSets(const std::vector<std::string> & namesPSet) ;

private:
  edm::ParameterSet layerConfig(const std::string & nameLayer,const edm::ParameterSet& cfg) const;
  void updateEventSetup(const edm::EventSetup& es);

  edm::ESWatcher<TrackerRecoGeometryRecord> geometryWatcher_;
  edm::ESWatcher<TransientRecHitRecord> trhWatcher_;

  struct LayerSpec { 
    LayerSpec(unsigned short index, const std::string& layerName, const edm::ParameterSet& cfgLayer, edm::ConsumesCollector& iC);
    ~LayerSpec() = default;
    LayerSpec(const LayerSpec&) = delete;
    LayerSpec& operator=(const LayerSpec&) = delete;
    LayerSpec(LayerSpec &&) = default;
    LayerSpec& operator=(LayerSpec &&) = default;
    const unsigned short nameIndex;
    std::string pixelHitProducer;
    bool usePixelHitProducer;
    const std::string hitBuilder;

    GeomDetEnumerators::SubDetector subdet;
    TrackerDetSide side;
    int idLayer;
    std::unique_ptr<ctfseeding::HitExtractor> extractor;

    std::string print(const std::vector<std::string>& names) const;
  }; 
  unsigned short theNumberOfLayersInSet;
  std::vector<SeedingLayerSetsHits::LayerSetIndex> theLayerSetIndices; // indices to theLayers to form the layer sets
  std::vector<std::string> theLayerNames;
  std::vector<const DetLayer *> theLayerDets;
  std::vector<const TransientTrackingRecHitBuilder *> theTTRHBuilders;
  std::vector<LayerSpec> theLayers;
};
#endif
