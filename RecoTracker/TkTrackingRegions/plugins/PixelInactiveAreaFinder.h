#ifndef RecoTracker_TkTrackingRegions_PixelInactiveAreaFinder_H
#define RecoTracker_TkTrackingRegions_PixelInactiveAreaFinder_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "AreaSeededTrackingRegionsBuilder.h"

class SiPixelQuality;
class TrackerGeometry;
class TrackerTopology;

// Adapted from a summer student project of Niko Leskinen (HIP, Finland)

class PixelInactiveAreaFinder {
public:
  using Area = AreaSeededTrackingRegionsBuilder::Area;
  using SeedingLayerId = SeedingLayerSetsBuilder::SeedingLayerId;
  using LayerSetIndex = SeedingLayerSetsHits::LayerSetIndex;

  struct DetGroupSpan {
    int subdetId;
    std::pair<float,float> phiSpan;
    std::pair<float,float> zSpan;
    std::pair<float,float> rSpan;
    unsigned int layer;
    unsigned int disk;
    DetGroupSpan():
      subdetId(0),
      phiSpan(0,0),
      zSpan(0,0),
      rSpan(0,0),
      layer(0),disk(0)
    {}
  };
  using DetGroupSpanContainer = std::vector<DetGroupSpan>;

  class InactiveAreas {
  public:
    InactiveAreas(const std::vector<SeedingLayerId> *layers,
                  std::vector<DetGroupSpanContainer>&& spans,
                  const std::vector<std::pair<unsigned short, unsigned short> > *layerPairs,
                  const std::vector<std::vector<LayerSetIndex> > *layerSetIndexInactiveToActive):
      layers_(layers),
      spans_(std::move(spans)),
      layerPairIndices_(layerPairs),
      layerSetIndexInactiveToActive_(layerSetIndexInactiveToActive)
    {}

    std::vector<std::pair<std::vector<Area>, std::vector<LayerSetIndex> > > areasAndLayerSets(const GlobalPoint& point, float zwidth) const;
    std::vector<std::pair<std::vector<DetGroupSpan>, std::vector<LayerSetIndex> > > spansAndLayerSets(const GlobalPoint& point, float zwidth) const;

  private:
    const std::vector<SeedingLayerId> *layers_;   // pointer to PixelInactiveAreaFinder::layers_
    std::vector<DetGroupSpanContainer> spans_;    // inactive areas for each layer, indexing corresponds to layers_
    const std::vector<std::pair<unsigned short, unsigned short> > *layerPairIndices_; // indices to the layer pair within the input SeedingLayerSetsHits
    const std::vector<std::vector<LayerSetIndex> > *layerSetIndexInactiveToActive_; // mapping from index in "inactive" seeding layers to "active" seeding layers
  };


  PixelInactiveAreaFinder(const edm::ParameterSet& iConfig, const std::vector<SeedingLayerId>& seedingLayers,
                          const SeedingLayerSetsLooper& seedingLayerSetsLooper);
  ~PixelInactiveAreaFinder() = default;

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  InactiveAreas inactiveAreas(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  // Configuration
  const bool debug_;
  const bool createPlottingFiles_;

  std::vector<SeedingLayerId> layers_;
  std::vector<std::pair<unsigned short, unsigned short> > layerSetIndices_; // indices within layers_
  std::vector<std::vector<LayerSetIndex> > layerSetIndexInactiveToActive_; // mapping from index in layers_ to constructor seedingLayers+seedingLayerSetsLooper

  // Output type aliases
  using DetGroupSpanContainerPair = std::pair<DetGroupSpanContainer,DetGroupSpanContainer>;
  using OverlapSpans = std::vector<DetGroupSpan>;
  using OverlapSpansContainer = std::vector<OverlapSpans>;
  // static data members; TODO see if these could be obtained from the geometry
  const static unsigned int nLayer1Ladders = 12;
  const static unsigned int nLayer2Ladders = 28;
  const static unsigned int nLayer3Ladders = 44;
  const static unsigned int nLayer4Ladders = 64;
  const static unsigned int nModulesPerLadder = 8;
  // type aliases
  using det_t = uint32_t;
  using Span_t = std::pair<float,float>;
  using DetContainer = std::vector<uint32_t>;
  using DetGroup = std::vector<uint32_t>;
  using DetGroupContainer = std::vector<DetGroup>;
  using DetectorSet = std::set<uint32_t>;
  using Stream = std::stringstream;
  // data handles and containers;
  edm::ESWatcher<TrackerDigiGeometryRecord> geometryWatcher_;

  const SiPixelQuality *pixelQuality_ = nullptr;
  const TrackerGeometry *trackerGeometry_ = nullptr;
  const TrackerTopology *trackerTopology_ = nullptr;

  DetContainer pixelDetsBarrel;
  DetContainer pixelDetsEndcap;
  DetContainer badPixelDetsBarrel;
  DetContainer badPixelDetsEndcap;
  // functions for fetching date from handles
  void updatePixelDets(const edm::EventSetup& iSetup);
  void getBadPixelDets();
  // Printing functions
  void detInfo(const det_t & det, Stream & ss);
  void printPixelDets();
  void printBadPixelDets();
  void printBadDetGroups();
  void printBadDetGroupSpans();
  void createPlottingFiles();
  // Functions for finding bad detGroups
  static bool phiRangesOverlap(const float x1,const float x2, const float y1,const float y2);
  static bool phiRangesOverlap(const Span_t&phiSpanA, const Span_t&phiSpanB);
  bool detWorks(det_t det);
  DetGroup badAdjecentDetsBarrel(const det_t & det);
  DetGroup badAdjecentDetsEndcap(const det_t & det);
  DetGroup reachableDetGroup(const det_t & initDet, DetectorSet & foundDets);
  DetGroupContainer badDetGroupsBarrel();
  DetGroupContainer badDetGroupsEndcap();
  // Functions for finding ranges that detGroups cover
  static bool phiMoreClockwise(float phiA, float phiB);
  static bool phiMoreCounterclockwise(float phiA, float phiB);
  void getPhiSpanBarrel(const DetGroup & detGroup, DetGroupSpan & cspan);
  void getPhiSpanEndcap(const DetGroup & detGroup, DetGroupSpan & cspan);
  void getZSpan(const DetGroup & detGroup, DetGroupSpan & cspan);
  void getRSpan(const DetGroup & detGroup, DetGroupSpan & cspan);
  void getSpan(const DetGroup & detGroup, DetGroupSpan & cspan);
  DetGroupSpanContainerPair detGroupSpans();
};

#endif
