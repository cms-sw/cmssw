#ifndef RecoTracker_TkTrackingRegions_PixelInactiveAreaFinder_H
#define RecoTracker_TkTrackingRegions_PixelInactiveAreaFinder_H

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/VecArray.h"
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
    std::pair<float, float> phiSpan;
    std::pair<float, float> zSpan;
    std::pair<float, float> rSpan;
    unsigned int layer;
    unsigned int disk;
    DetGroupSpan() : subdetId(0), phiSpan(0, 0), zSpan(0, 0), rSpan(0, 0), layer(0), disk(0) {}
  };
  using DetGroupSpanContainer = std::vector<DetGroupSpan>;

  class InactiveAreas {
  public:
    InactiveAreas(const std::vector<SeedingLayerId>* inactiveLayers,
                  std::vector<DetGroupSpanContainer>&& inactiveSpans,
                  const std::vector<std::pair<unsigned short, unsigned short> >* inactiveLayerPairIndices,
                  const std::vector<std::vector<LayerSetIndex> >* layerSetIndexInactiveToActive)
        : inactiveLayers_(inactiveLayers),
          inactiveSpans_(std::move(inactiveSpans)),
          inactiveLayerPairIndices_(inactiveLayerPairIndices),
          layerSetIndexInactiveToActive_(layerSetIndexInactiveToActive) {}

    template <typename T>
    using VecArray2 = edm::VecArray<
        T,
        2>;  // 2 inactive layers (using VecArray for possible extension to 1 inactive layer, i.e. triplet mitigation)
    std::vector<std::pair<VecArray2<Area>, std::vector<LayerSetIndex> > > areasAndLayerSets(const GlobalPoint& point,
                                                                                            float zwidth) const;
    std::vector<std::pair<VecArray2<DetGroupSpan>, std::vector<LayerSetIndex> > > spansAndLayerSets(
        const GlobalPoint& point, float zwidth) const;

  private:
    const std::vector<SeedingLayerId>* inactiveLayers_;  // pointer to PixelInactiveAreaFinder::layers_
    std::vector<DetGroupSpanContainer> inactiveSpans_;  // inactive areas for each layer, indexing corresponds to layers_
    const std::vector<std::pair<unsigned short, unsigned short> >*
        inactiveLayerPairIndices_;  // indices to the layer pair within the input SeedingLayerSetsHits for pairs of layers to check for correlated inactive regions
    const std::vector<std::vector<LayerSetIndex> >*
        layerSetIndexInactiveToActive_;  // mapping from index in "inactive" seeding layers to "active" seeding layers
  };

  PixelInactiveAreaFinder(const edm::ParameterSet& iConfig,
                          const std::vector<SeedingLayerId>& seedingLayers,
                          const SeedingLayerSetsLooper& seedingLayerSetsLooper,
                          edm::ConsumesCollector&& iC);
  ~PixelInactiveAreaFinder() = default;

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  InactiveAreas inactiveAreas(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  // Configuration
  const bool debug_;
  const bool createPlottingFiles_;
  const bool ignoreSingleFPixPanelModules_;

  std::vector<SeedingLayerId> inactiveLayers_;  // layers to check for inactive regions
  std::vector<std::pair<unsigned short, unsigned short> > inactiveLayerSetIndices_;  // indices within inactiveLayers_
  std::vector<std::vector<LayerSetIndex> >
      layerSetIndexInactiveToActive_;  // mapping from index in inactiveLayers_ to constructor seedingLayers+seedingLayerSetsLooper

  std::vector<edm::EDGetTokenT<DetIdCollection> > inactivePixelDetectorTokens_;
  std::vector<edm::EDGetTokenT<PixelFEDChannelCollection> > badPixelFEDChannelsTokens_;

  // Output type aliases
  using DetGroupSpanContainerPair = std::pair<DetGroupSpanContainer, DetGroupSpanContainer>;
  using OverlapSpans = std::vector<DetGroupSpan>;
  using OverlapSpansContainer = std::vector<OverlapSpans>;
  // static data members; TODO see if these could be obtained from the geometry
  std::array<unsigned short, 4> nBPixLadders;
  unsigned short nModulesPerLadder;
  // type aliases
  using det_t = uint32_t;
  using DetContainer = std::vector<uint32_t>;
  using DetGroup = std::vector<uint32_t>;
  using DetGroupContainer = std::vector<DetGroup>;
  using DetectorSet = std::set<uint32_t>;
  using Stream = std::stringstream;
  // data handles and containers;
  edm::ESWatcher<TrackerDigiGeometryRecord> geometryWatcher_;

  const SiPixelQuality* pixelQuality_ = nullptr;
  const TrackerGeometry* trackerGeometry_ = nullptr;
  const TrackerTopology* trackerTopology_ = nullptr;

  DetContainer pixelDetsBarrel_;
  DetContainer pixelDetsEndcap_;
  DetContainer badPixelDetsBarrel_;
  DetContainer badPixelDetsEndcap_;
  // functions for fetching date from handles
  void updatePixelDets(const edm::EventSetup& iSetup);
  void getBadPixelDets(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  // Printing functions
  void detInfo(const det_t& det, Stream& ss);
  void printPixelDets();
  void printBadPixelDets();
  void printBadDetGroups();
  void printBadDetGroupSpans();
  void createPlottingFiles();
  // Functions for finding bad detGroups
  bool detWorks(det_t det);
  DetGroup badAdjecentDetsBarrel(const det_t& det);
  DetGroup badAdjecentDetsEndcap(const det_t& det);
  DetGroup reachableDetGroup(const det_t& initDet, DetectorSet& foundDets);
  DetGroupContainer badDetGroupsBarrel();
  DetGroupContainer badDetGroupsEndcap();
  // Functions for finding ranges that detGroups cover
  void getPhiSpanBarrel(const DetGroup& detGroup, DetGroupSpan& cspan);
  void getPhiSpanEndcap(const DetGroup& detGroup, DetGroupSpan& cspan);
  void getZSpan(const DetGroup& detGroup, DetGroupSpan& cspan);
  void getRSpan(const DetGroup& detGroup, DetGroupSpan& cspan);
  void getSpan(const DetGroup& detGroup, DetGroupSpan& cspan);
  DetGroupSpanContainerPair detGroupSpans();
};

#endif
