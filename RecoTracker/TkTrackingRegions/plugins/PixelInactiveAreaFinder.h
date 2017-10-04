#ifndef RecoTracker_TkTrackingRegions_PixelInactiveAreaFinder_H
#define RecoTracker_TkTrackingRegions_PixelInactiveAreaFinder_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "AreaSeededTrackingRegionsBuilder.h"

// Adapted from a summer student project of Niko Leskinen (HIP, Finland)

class PixelInactiveAreaFinder {
public:
  using Area = AreaSeededTrackingRegionsBuilder::Area;
  using SeedingLayerId = SeedingLayerSetsBuilder::SeedingLayerId;

  class AreaLayers {
  public:
    const std::vector<Area>& areas() const { return areas_; }
    const std::vector<SeedingLayerId> layers() const { return layers_; }

  private:
    std::vector<Area> areas_;            // inactive areas related to these two layers
    std::vector<SeedingLayerId> layers_; // inner and outer active layer, so size always 2; is vector only because of client interface
  };

  PixelInactiveAreaFinder(const edm::ParameterSet& iConfig, const std::vector<SeedingLayerSetsBuilder::SeedingLayerId>& seedingLayers,
                          const std::vector<SeedingLayerSetsHits::LayerSetIndex>& layerSetIndices);
  ~PixelInactiveAreaFinder() = default;

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  std::vector<AreaLayers> inactiveAreas(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  // Configuration
  const bool debug_;
  const bool createPlottingFiles_;

  // Output types
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
  // Output type aliases
  using DetGroupSpanContainer = std::vector<DetGroupSpan>;
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
  // data handles and containers; TODO convert containers to pointers
  edm::ESHandle<SiPixelQuality> pixelQuality;
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  edm::ESHandle<TrackerTopology> trackerTopology;
  DetContainer pixelDetsBarrel;
  DetContainer pixelDetsEndcap;
  DetContainer badPixelDetsBarrel;
  DetContainer badPixelDetsEndcap;
  // functions for fetching date from handles
  void getPixelDetsBarrel();
  void getPixelDetsEndcap();
  void getBadPixelDets();
  // Printing functions
  void detInfo(const det_t & det, Stream & ss);
  void detGroupSpanInfo(const DetGroupSpan & cspan, Stream & ss);
  void printPixelDets();
  void printBadPixelDets();
  void printBadDetGroups();
  void printBadDetGroupSpans();
  void printOverlapSpans();
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
  // Functions for findind overlapping functions
  static float zAxisIntersection(const float zrPointA[2], const float zrPoint[2]);
  bool getZAxisOverlapRangeBarrel(const DetGroupSpan & cspanA, const DetGroupSpan & cspanB,std::pair<float,float> & range);
  bool getZAxisOverlapRangeEndcap(const DetGroupSpan & cspanA, const DetGroupSpan & cspanB,std::pair<float,float> & range);
  bool getZAxisOverlapRangeBarrelEndcap(const DetGroupSpan & cspanA, const DetGroupSpan & cspanB,std::pair<float,float> & range);
  void compareDetGroupSpansBarrel();
  OverlapSpansContainer overlappingSpans(float zAxisThreshold = std::numeric_limits<float>::infinity());
};

#endif
