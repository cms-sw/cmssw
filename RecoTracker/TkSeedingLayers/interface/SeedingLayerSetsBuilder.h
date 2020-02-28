#ifndef TkSeedingLayers_SeedingLayerSetsBuilder_H
#define TkSeedingLayers_SeedingLayerSetsBuilder_H

#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsLooper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include <string>
#include <vector>
namespace edm {
  class Event;
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm
namespace ctfseeding {
  class HitExtractor;
}
class TrackerRecoGeometryRecord;
class TransientRecHitRecord;
class TransientTrackingRecHitBuilder;
class DetLayer;

class SeedingLayerSetsBuilder {
public:
  using SeedingLayerId = std::tuple<GeomDetEnumerators::SubDetector, TrackerDetSide, int>;

  SeedingLayerSetsBuilder() = default;
  SeedingLayerSetsBuilder(const edm::ParameterSet& cfg,
                          edm::ConsumesCollector& iC,
                          const edm::InputTag& fastsimHitTag);  //FastSim specific constructor
  SeedingLayerSetsBuilder(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);
  SeedingLayerSetsBuilder(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC);
  ~SeedingLayerSetsBuilder();

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  unsigned short numberOfLayers() const { return theLayers.size(); }
  unsigned short numberOfLayerSets() const {
    return theNumberOfLayersInSet > 0 ? theLayerSetIndices.size() / theNumberOfLayersInSet : 0;
  }
  std::vector<SeedingLayerId> layers() const;  // please call at most once per job per client
  SeedingLayerSetsLooper seedingLayerSetsLooper() const {
    return SeedingLayerSetsLooper(theNumberOfLayersInSet, &theLayerSetIndices);
  }

  const std::vector<SeedingLayerSetsHits::LayerSetIndex>& layerSetIndices() const { return theLayerSetIndices; }

  std::unique_ptr<SeedingLayerSetsHits> hits(const edm::Event& ev, const edm::EventSetup& es);
  //new function for FastSim only
  std::unique_ptr<SeedingLayerSetsHits> makeSeedingLayerSetsHitsforFastSim(const edm::Event& ev,
                                                                           const edm::EventSetup& es);

  static SeedingLayerId nameToEnumId(const std::string& name);
  static std::vector<std::vector<std::string> > layerNamesInSets(const std::vector<std::string>& namesPSet);

private:
  edm::ParameterSet layerConfig(const std::string& nameLayer, const edm::ParameterSet& cfg) const;
  void updateEventSetup(const edm::EventSetup& es);

  edm::ESWatcher<TrackerRecoGeometryRecord> geometryWatcher_;
  edm::ESWatcher<TransientRecHitRecord> trhWatcher_;
  edm::EDGetTokenT<FastTrackerRecHitCollection> fastSimrecHitsToken_;
  struct LayerSpec {
    LayerSpec(unsigned short index,
              const std::string& layerName,
              const edm::ParameterSet& cfgLayer,
              edm::ConsumesCollector& iC);
    ~LayerSpec() = default;
    LayerSpec(const LayerSpec&) = delete;
    LayerSpec& operator=(const LayerSpec&) = delete;
    LayerSpec(LayerSpec&&) = default;
    LayerSpec& operator=(LayerSpec&&) = delete;
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
  std::vector<SeedingLayerSetsHits::LayerSetIndex> theLayerSetIndices;  // indices to theLayers to form the layer sets
  std::vector<std::string> theLayerNames;
  std::vector<const DetLayer*> theLayerDets;
  std::vector<const TransientTrackingRecHitBuilder*> theTTRHBuilders;
  std::vector<LayerSpec> theLayers;
};

#endif
