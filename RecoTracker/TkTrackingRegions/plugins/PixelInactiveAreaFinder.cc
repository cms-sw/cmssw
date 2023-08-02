#include "PixelInactiveAreaFinder.h"

#include "FWCore/Utilities/interface/VecArray.h"
#include "FWCore/Utilities/interface/transform.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsLooper.h"

#include <fstream>
#include <queue>
#include <algorithm>

std::ostream& operator<<(std::ostream& os, SeedingLayerSetsBuilder::SeedingLayerId layer) {
  if (std::get<0>(layer) == GeomDetEnumerators::PixelBarrel) {
    os << "BPix";
  } else {
    os << "FPix";
  }
  os << std::get<2>(layer);
  if (std::get<1>(layer) == TrackerDetSide::PosEndcap) {
    os << "_pos";
  } else if (std::get<1>(layer) == TrackerDetSide::NegEndcap) {
    os << "_neg";
  }
  return os;
}

namespace {
  using LayerPair = std::pair<SeedingLayerSetsBuilder::SeedingLayerId, SeedingLayerSetsBuilder::SeedingLayerId>;
  using ActiveLayerSetToInactiveSetsMap = std::map<LayerPair, edm::VecArray<LayerPair, 5>>;
  using Stream = std::stringstream;
  using Span_t = std::pair<float, float>;

  ActiveLayerSetToInactiveSetsMap createActiveToInactiveMap() {
    ActiveLayerSetToInactiveSetsMap map;

    auto bpix = [](int layer) {
      return SeedingLayerSetsBuilder::SeedingLayerId(GeomDetEnumerators::PixelBarrel, TrackerDetSide::Barrel, layer);
    };
    auto fpix_pos = [](int disk) {
      return SeedingLayerSetsBuilder::SeedingLayerId(GeomDetEnumerators::PixelEndcap, TrackerDetSide::PosEndcap, disk);
    };
    auto fpix_neg = [](int disk) {
      return SeedingLayerSetsBuilder::SeedingLayerId(GeomDetEnumerators::PixelEndcap, TrackerDetSide::NegEndcap, disk);
    };

    auto add_permutations = [&](std::array<SeedingLayerSetsBuilder::SeedingLayerId, 4> quads) {
      do {
        // skip permutations like BPix2+BPix1 or FPix1+BPix1
        // operator> works automatically
        if (quads[0] > quads[1] || quads[2] > quads[3])
          continue;

        map[std::make_pair(quads[0], quads[1])].emplace_back(quads[2], quads[3]);
      } while (std::next_permutation(quads.begin(), quads.end()));
    };

    // 4 barrel
    add_permutations({{bpix(1), bpix(2), bpix(3), bpix(4)}});

    // 3 barrel, 1 forward
    add_permutations({{bpix(1), bpix(2), bpix(3), fpix_pos(1)}});
    add_permutations({{bpix(1), bpix(2), bpix(3), fpix_neg(1)}});

    // 2 barrel, 2 forward
    add_permutations({{bpix(1), bpix(2), fpix_pos(1), fpix_pos(2)}});
    add_permutations({{bpix(1), bpix(2), fpix_neg(1), fpix_neg(2)}});

    // 1 barrel, 3 forward
    add_permutations({{bpix(1), fpix_pos(1), fpix_pos(2), fpix_pos(3)}});
    add_permutations({{bpix(1), fpix_neg(1), fpix_neg(2), fpix_neg(3)}});

#ifdef EDM_ML_DEBUG
    LogDebug("PixelInactiveAreaFinder") << "Active to inactive mapping";
    for (const auto& elem : map) {
      std::stringstream ss;
      for (const auto& layerPair : elem.second) {
        ss << layerPair.first << "+" << layerPair.second << ",";
      }
      LogTrace("PixelInactiveAreaFinder") << " " << elem.first.first << "+" << elem.first.second << " => " << ss.str();
    }
#endif

    return map;
  }

  void detGroupSpanInfo(const PixelInactiveAreaFinder::DetGroupSpan& cspan, Stream& ss) {
    using std::fixed;
    using std::left;
    using std::noshowpos;
    using std::right;
    using std::setfill;
    using std::setprecision;
    using std::setw;
    using std::showpos;
    std::string deli = "; ";
    ss << "subdetid:[" << cspan.subdetId << "]" << deli;
    if (cspan.subdetId == PixelSubdetector::PixelBarrel) {
      ss << "layer:[" << cspan.layer << "]" << deli;
    } else {
      ss << "disk:[" << cspan.disk << "]" << deli;
    }
    ss
        //<< setfill(' ') << setw(36) << " "
        << setprecision(16) << showpos << "phi:<" << right << setw(12) << cspan.phiSpan.first << "," << left << setw(12)
        << cspan.phiSpan.second << ">" << deli << "z:<" << right << setw(7) << cspan.zSpan.first << "," << left
        << setw(7) << cspan.zSpan.second << ">" << deli << noshowpos << "r:<" << right << setw(10) << cspan.rSpan.first
        << "," << left << setw(10) << cspan.rSpan.second << ">" << deli;
  }
  void printOverlapSpans(const PixelInactiveAreaFinder::InactiveAreas& areasLayers) {
    auto spansLayerSets = areasLayers.spansAndLayerSets(GlobalPoint(0, 0, 0), std::numeric_limits<float>::infinity());

    Stream ss;
    for (auto const& spansLayers : spansLayerSets) {
      ss << "Overlapping detGroups:\n";
      for (auto const& cspan : spansLayers.first) {
        detGroupSpanInfo(cspan, ss);
        ss << std::endl;
      }
    }
    edm::LogPrint("PixelInactiveAreaFinder") << ss.str();
  }

  // Functions for finding bad detGroups
  bool phiRangesOverlap(const Span_t& phiSpanA, const Span_t& phiSpanB) {
    float x1, x2, y1, y2;
    std::tie(x1, x2) = phiSpanA;
    std::tie(y1, y2) = phiSpanB;
    // assuming phi ranges are [x1,x2] and [y1,y2] and xi,yi in [-pi,pi]
    if (x1 <= x2 && y1 <= y2) {
      return x1 <= y2 && y1 <= x2;
    } else if ((x1 > x2 && y1 <= y2) || (y1 > y2 && x1 <= x2)) {
      return y1 <= x2 || x1 <= y2;
    } else if (x1 > x2 && y1 > y2) {
      return true;
    } else {
      return false;
    }
  }

  // Functions for finding ranges that detGroups cover
  bool phiMoreClockwise(float phiA, float phiB) {
    // return true if a is more clockwise than b
    return reco::deltaPhi(phiA, phiB) <= 0.f;
  }
  bool phiMoreCounterclockwise(float phiA, float phiB) {
    // return true if a is more counterclockwise than b
    return reco::deltaPhi(phiA, phiB) >= 0.f;
  }

  // Functions for findind overlapping functions
  float zAxisIntersection(const float zrPointA[2], const float zrPointB[2]) {
    return (zrPointB[0] - zrPointA[0]) / (zrPointB[1] - zrPointA[1]) * (-zrPointA[1]) + zrPointA[0];
  }
  bool getZAxisOverlapRangeBarrel(const PixelInactiveAreaFinder::DetGroupSpan& cspanA,
                                  const PixelInactiveAreaFinder::DetGroupSpan& cspanB,
                                  std::pair<float, float>& range) {
    PixelInactiveAreaFinder::DetGroupSpan cspanUpper;
    PixelInactiveAreaFinder::DetGroupSpan cspanLower;
    if (cspanA.rSpan.second < cspanB.rSpan.first) {
      cspanLower = cspanA;
      cspanUpper = cspanB;
    } else if (cspanA.rSpan.first > cspanB.rSpan.second) {
      cspanUpper = cspanA;
      cspanLower = cspanB;
    } else {
      return false;
    }
    float lower = 0;
    float upper = 0;
    if (cspanUpper.zSpan.second < cspanLower.zSpan.first) {
      // lower intersectionpoint, point = {z,r} in cylindrical coordinates
      const float pointUpperDetGroupL[2] = {cspanUpper.zSpan.second, cspanUpper.rSpan.second};
      const float pointLowerDetGroupL[2] = {cspanLower.zSpan.first, cspanLower.rSpan.first};
      lower = zAxisIntersection(pointUpperDetGroupL, pointLowerDetGroupL);
      // upper intersectionpoint
      const float pointUpperDetGroupU[2] = {cspanUpper.zSpan.first, cspanUpper.rSpan.first};
      const float pointLowerDetGroupU[2] = {cspanLower.zSpan.second, cspanLower.rSpan.second};
      upper = zAxisIntersection(pointUpperDetGroupU, pointLowerDetGroupU);
    } else if (cspanUpper.zSpan.first <= cspanLower.zSpan.second && cspanLower.zSpan.first <= cspanUpper.zSpan.second) {
      // lower intersectionpoint, point = {z,r} in cylindrical coordinates
      const float pointUpperDetGroupL[2] = {cspanUpper.zSpan.second, cspanUpper.rSpan.first};
      const float pointLowerDetGroupL[2] = {cspanLower.zSpan.first, cspanLower.rSpan.second};
      lower = zAxisIntersection(pointUpperDetGroupL, pointLowerDetGroupL);
      // upper intersectionpoint
      const float pointUpperDetGroupU[2] = {cspanUpper.zSpan.first, cspanUpper.rSpan.first};
      const float pointLowerDetGroupU[2] = {cspanLower.zSpan.second, cspanLower.rSpan.second};
      upper = zAxisIntersection(pointUpperDetGroupU, pointLowerDetGroupU);
    } else if (cspanUpper.zSpan.first > cspanLower.zSpan.second) {
      // lower intersectionpoint, point = {z,r} in cylindrical coordinates
      const float pointUpperDetGroupL[2] = {cspanUpper.zSpan.second, cspanUpper.rSpan.first};
      const float pointLowerDetGroupL[2] = {cspanLower.zSpan.first, cspanLower.rSpan.second};
      lower = zAxisIntersection(pointUpperDetGroupL, pointLowerDetGroupL);
      // upper intersectionpoint
      const float pointUpperDetGroupU[2] = {cspanUpper.zSpan.first, cspanUpper.rSpan.second};
      const float pointLowerDetGroupU[2] = {cspanLower.zSpan.second, cspanLower.rSpan.first};
      upper = zAxisIntersection(pointUpperDetGroupU, pointLowerDetGroupU);
    } else {
      //something wrong
      return false;
    }
    range = std::pair<float, float>(lower, upper);
    return true;
  }
  bool getZAxisOverlapRangeEndcap(const PixelInactiveAreaFinder::DetGroupSpan& cspanA,
                                  const PixelInactiveAreaFinder::DetGroupSpan& cspanB,
                                  std::pair<float, float>& range) {
    // While on left hand side of pixel detector
    PixelInactiveAreaFinder::DetGroupSpan cspanNearer;
    PixelInactiveAreaFinder::DetGroupSpan cspanFurther;
    float lower = 0;
    float upper = 0;
    if (cspanA.zSpan.first < 0 && cspanB.zSpan.first < 0) {
      if (cspanA.zSpan.second < cspanB.zSpan.first) {
        cspanFurther = cspanA;
        cspanNearer = cspanB;
      } else if (cspanB.zSpan.second < cspanA.zSpan.first) {
        cspanFurther = cspanB;
        cspanNearer = cspanA;
      } else {
#ifdef EDM_ML_DEBUG
        LogTrace("PixelInactiveAreaFinder") << "No overlap, same disk propably. Spans:";
        Stream ss;
        detGroupSpanInfo(cspanA, ss);
        ss << std::endl;
        detGroupSpanInfo(cspanB, ss);
        ss << std::endl;
        LogTrace("PixelInactiveAreaFinder") << ss.str();
        ss.str(std::string());
        LogTrace("PixelInactiveAreaFinder") << "**";
#endif
        return false;
      }
      if (cspanFurther.rSpan.second > cspanNearer.rSpan.first) {
        const float pointA[2] = {cspanFurther.zSpan.second, cspanFurther.rSpan.second};
        const float pointB[2] = {cspanNearer.zSpan.first, cspanNearer.rSpan.first};
        lower = zAxisIntersection(pointA, pointB);
        if (cspanFurther.rSpan.first > cspanNearer.rSpan.second) {
          const float pointC[2] = {cspanFurther.zSpan.first, cspanFurther.rSpan.first};
          const float pointD[2] = {cspanNearer.zSpan.second, cspanFurther.rSpan.second};
          upper = zAxisIntersection(pointC, pointD);
        } else {
          upper = std::numeric_limits<float>::infinity();
        }
      } else {
#ifdef EDM_ML_DEBUG
        LogTrace("PixelInactiveAreaFinder") << "No overlap, further detGroup is lower. Spans:";
        Stream ss;
        detGroupSpanInfo(cspanA, ss);
        ss << std::endl;
        detGroupSpanInfo(cspanB, ss);
        ss << std::endl;
        LogTrace("PixelInactiveAreaFinder") << ss.str();
        ss.str(std::string());
        LogTrace("PixelInactiveAreaFinder") << "**";
#endif
        return false;
      }
    } else if (cspanA.zSpan.first > 0 && cspanB.zSpan.first > 0) {
      if (cspanA.zSpan.first > cspanB.zSpan.second) {
        cspanFurther = cspanA;
        cspanNearer = cspanB;
      } else if (cspanB.zSpan.first > cspanA.zSpan.second) {
        cspanFurther = cspanB;
        cspanNearer = cspanA;
      } else {
#ifdef EDM_ML_DEBUG
        LogTrace("PixelInactiveAreaFinder") << "No overlap, same disk propably. Spans:";
        Stream ss;
        detGroupSpanInfo(cspanA, ss);
        ss << std::endl;
        detGroupSpanInfo(cspanB, ss);
        ss << std::endl;
        LogTrace("PixelInactiveAreaFinder") << ss.str();
        ss.str(std::string());
        LogTrace("PixelInactiveAreaFinder") << "**";
#endif
        return false;
      }
      if (cspanFurther.rSpan.second > cspanNearer.rSpan.first) {
        const float pointA[2] = {cspanFurther.zSpan.first, cspanFurther.rSpan.second};
        const float pointB[2] = {cspanNearer.zSpan.second, cspanNearer.rSpan.first};
        upper = zAxisIntersection(pointA, pointB);
        if (cspanFurther.rSpan.first > cspanNearer.rSpan.second) {
          const float pointC[2] = {cspanFurther.zSpan.second, cspanFurther.rSpan.first};
          const float pointD[2] = {cspanNearer.zSpan.first, cspanFurther.rSpan.second};
          lower = zAxisIntersection(pointC, pointD);
        } else {
          lower = -std::numeric_limits<float>::infinity();
        }
      } else {
#ifdef EDM_ML_DEBUG
        LogTrace("PixelInactiveAreaFinder") << "No overlap, further detGroup lower. Spans:";
        Stream ss;
        detGroupSpanInfo(cspanA, ss);
        ss << std::endl;
        detGroupSpanInfo(cspanB, ss);
        ss << std::endl;
        LogTrace("PixelInactiveAreaFinder") << ss.str();
        ss.str(std::string());
        LogTrace("PixelInactiveAreaFinder") << "**";
#endif
        return false;
      }
    } else {
#ifdef EDM_ML_DEBUG
      LogTrace("PixelInactiveAreaFinder") << "No overlap, different sides of z axis. Spans:";
      Stream ss;
      detGroupSpanInfo(cspanA, ss);
      ss << std::endl;
      detGroupSpanInfo(cspanB, ss);
      ss << std::endl;
      LogTrace("PixelInactiveAreaFinder") << ss.str();
      ss.str(std::string());
      LogTrace("PixelInactiveAreaFinder") << "**";
#endif
      return false;
    }
    range = std::pair<float, float>(lower, upper);
    return true;
  }

  bool getZAxisOverlapRangeBarrelEndcap(const PixelInactiveAreaFinder::DetGroupSpan& cspanBar,
                                        const PixelInactiveAreaFinder::DetGroupSpan& cspanEnd,
                                        std::pair<float, float>& range) {
    float lower = 0;
    float upper = 0;
    if (cspanEnd.rSpan.second > cspanBar.rSpan.first) {
      if (cspanEnd.zSpan.second < cspanBar.zSpan.first) {
        // if we are on the left hand side of pixel detector
        const float pointA[2] = {cspanEnd.zSpan.second, cspanEnd.rSpan.second};
        const float pointB[2] = {cspanBar.zSpan.first, cspanBar.rSpan.first};
        lower = zAxisIntersection(pointA, pointB);
        if (cspanEnd.rSpan.first > cspanBar.rSpan.second) {
          // if does not overlap, then there is also upper limit
          const float pointC[2] = {cspanEnd.zSpan.first, cspanEnd.rSpan.first};
          const float pointD[2] = {cspanBar.zSpan.second, cspanBar.rSpan.second};
          upper = zAxisIntersection(pointC, pointD);
        } else {
          upper = std::numeric_limits<float>::infinity();
        }
      } else if (cspanEnd.zSpan.first > cspanBar.zSpan.second) {
        // if we are on the right hand side of pixel detector
        const float pointA[2] = {cspanEnd.zSpan.first, cspanEnd.rSpan.second};
        const float pointB[2] = {cspanBar.zSpan.second, cspanBar.rSpan.first};
        upper = zAxisIntersection(pointA, pointB);
        if (cspanEnd.rSpan.first > cspanBar.rSpan.second) {
          const float pointC[2] = {cspanEnd.zSpan.second, cspanEnd.rSpan.first};
          const float pointD[2] = {cspanBar.zSpan.first, cspanBar.rSpan.second};
          lower = zAxisIntersection(pointC, pointD);
        } else {
          lower = -std::numeric_limits<float>::infinity();
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
    range = std::pair<float, float>(lower, upper);
    return true;
  }
}  // namespace

std::vector<
    std::pair<edm::VecArray<PixelInactiveAreaFinder::Area, 2>, std::vector<PixelInactiveAreaFinder::LayerSetIndex>>>
PixelInactiveAreaFinder::InactiveAreas::areasAndLayerSets(const GlobalPoint& point, float zwidth) const {
  auto spansLayerSets = spansAndLayerSets(point, zwidth);

  // TODO: try to remove this conversion...
  std::vector<std::pair<VecArray2<Area>, std::vector<LayerSetIndex>>> ret;
  for (auto& item : spansLayerSets) {
    auto& innerSpan = item.first[0];
    auto& outerSpan = item.first[1];
    VecArray2<Area> areas;
    areas.emplace_back(innerSpan.rSpan.first,
                       innerSpan.rSpan.second,
                       innerSpan.phiSpan.first,
                       innerSpan.phiSpan.second,
                       innerSpan.zSpan.first,
                       innerSpan.zSpan.second);
    areas.emplace_back(outerSpan.rSpan.first,
                       outerSpan.rSpan.second,
                       outerSpan.phiSpan.first,
                       outerSpan.phiSpan.second,
                       outerSpan.zSpan.first,
                       outerSpan.zSpan.second);
    ret.emplace_back(areas, std::move(item.second));
  }

  return ret;
}

std::vector<std::pair<edm::VecArray<PixelInactiveAreaFinder::DetGroupSpan, 2>,
                      std::vector<PixelInactiveAreaFinder::LayerSetIndex>>>
PixelInactiveAreaFinder::InactiveAreas::spansAndLayerSets(const GlobalPoint& point, float zwidth) const {
  // TODO: in the future use 2D-r for the origin for the phi overlap check
  const float zmin = point.z() - zwidth;
  const float zmax = point.z() + zwidth;

  std::vector<std::pair<VecArray2<DetGroupSpan>, std::vector<LayerSetIndex>>> ret;

  LogDebug("PixelInactiveAreaFinder") << "Origin at " << point.x() << "," << point.y() << "," << point.z()
                                      << " z half width " << zwidth;

  for (LayerSetIndex i = 0, end = inactiveLayerPairIndices_->size(); i < end; ++i) {
    const auto& layerIdxPair = (*inactiveLayerPairIndices_)[i];
    const auto& innerSpans = inactiveSpans_[layerIdxPair.first];
    const auto& outerSpans = inactiveSpans_[layerIdxPair.second];

    for (const auto& innerSpan : innerSpans) {
      for (const auto& outerSpan : outerSpans) {
        if (phiRangesOverlap(innerSpan.phiSpan, outerSpan.phiSpan)) {
          std::pair<float, float> range(0, 0);

          bool zOverlap = false;
          const auto innerDet = std::get<0>((*inactiveLayers_)[layerIdxPair.first]);
          const auto outerDet = std::get<0>((*inactiveLayers_)[layerIdxPair.second]);
          if (innerDet == GeomDetEnumerators::PixelBarrel) {
            if (outerDet == GeomDetEnumerators::PixelBarrel)
              zOverlap = getZAxisOverlapRangeBarrel(innerSpan, outerSpan, range);
            else
              zOverlap = getZAxisOverlapRangeBarrelEndcap(innerSpan, outerSpan, range);
          } else {
            if (outerDet == GeomDetEnumerators::PixelEndcap)
              zOverlap = getZAxisOverlapRangeEndcap(innerSpan, outerSpan, range);
            else
              throw cms::Exception("LogicError") << "Forward->barrel transition is not supported";
          }

          if (zOverlap && zmin <= range.second && range.first <= zmax) {
#ifdef EDM_ML_DEBUG
            Stream ss;
            for (auto ind : (*layerSetIndexInactiveToActive_)[i]) {
              ss << ind << ",";
            }
            ss << "\n  ";
            detGroupSpanInfo(innerSpan, ss);
            ss << "\n  ";
            detGroupSpanInfo(outerSpan, ss);
            LogTrace("PixelInactiveAreaFinder") << " adding areas for active layer sets " << ss.str();
#endif
            VecArray2<DetGroupSpan> vec;
            vec.emplace_back(innerSpan);
            vec.emplace_back(outerSpan);
            ret.emplace_back(std::move(vec), (*layerSetIndexInactiveToActive_)[i]);
          }
        }
      }
    }
  }

  return ret;
}

PixelInactiveAreaFinder::PixelInactiveAreaFinder(
    const edm::ParameterSet& iConfig,
    const std::vector<SeedingLayerSetsBuilder::SeedingLayerId>& seedingLayers,
    const SeedingLayerSetsLooper& seedingLayerSetsLooper,
    edm::ConsumesCollector&& iC)
    : debug_(iConfig.getUntrackedParameter<bool>("debug")),
      createPlottingFiles_(iConfig.getUntrackedParameter<bool>("createPlottingFiles")),
      ignoreSingleFPixPanelModules_(iConfig.getParameter<bool>("ignoreSingleFPixPanelModules")),
      inactivePixelDetectorTokens_(
          edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("inactivePixelDetectorLabels"),
                                [&](const auto& tag) { return iC.consumes<DetIdCollection>(tag); })),
      badPixelFEDChannelsTokens_(
          edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("badPixelFEDChannelCollectionLabels"),
                                [&](const auto& tag) { return iC.consumes<PixelFEDChannelCollection>(tag); })),
      trackerGeometryToken_(iC.esConsumes()),
      trackerTopologyToken_(iC.esConsumes()),
      pixelQualityToken_(iC.esConsumes()) {
#ifdef EDM_ML_DEBUG
  for (const auto& layer : seedingLayers) {
    LogTrace("PixelInactiveAreaFinder") << "Input layer subdet " << std::get<0>(layer) << " side "
                                        << static_cast<unsigned int>(std::get<1>(layer)) << " layer "
                                        << std::get<2>(layer);
  }
#endif

  auto findOrAdd = [&](SeedingLayerId layer) -> unsigned short {
    auto found = std::find(inactiveLayers_.cbegin(), inactiveLayers_.cend(), layer);
    if (found == inactiveLayers_.cend()) {
      auto ret = inactiveLayers_.size();
      inactiveLayers_.push_back(layer);
      return ret;
    }
    return std::distance(inactiveLayers_.cbegin(), found);
  };

  // mapping from active layer pairs to inactive layer pairs
  const auto activeToInactiveMap = createActiveToInactiveMap();

  // convert input layer pairs (that are for active layers) to layer
  // pairs to look for inactive areas
  LayerSetIndex i = 0;
  for (const auto& layerSet : seedingLayerSetsLooper.makeRange(seedingLayers)) {
    assert(layerSet.size() == 2);
    auto found = activeToInactiveMap.find(std::make_pair(layerSet[0], layerSet[1]));
    if (found == activeToInactiveMap.end()) {
      throw cms::Exception("Configuration")
          << "Encountered layer pair " << layerSet[0] << "+" << layerSet[1]
          << " not found from the internal 'active layer pairs' to 'inactive layer pairs' mapping; either fix the "
             "input or the mapping (in PixelInactiveAreaFinder.cc)";
    }

    LogTrace("PixelInactiveAreaFinder") << "Input layer set " << layerSet[0] << "+" << layerSet[1];
    for (const auto& inactiveLayerSet : found->second) {
      auto innerInd = findOrAdd(inactiveLayerSet.first);
      auto outerInd = findOrAdd(inactiveLayerSet.second);

      auto found = std::find(
          inactiveLayerSetIndices_.cbegin(), inactiveLayerSetIndices_.cend(), std::make_pair(innerInd, outerInd));
      if (found == inactiveLayerSetIndices_.end()) {
        inactiveLayerSetIndices_.emplace_back(innerInd, outerInd);
        layerSetIndexInactiveToActive_.push_back(std::vector<LayerSetIndex>{i});
      } else {
        layerSetIndexInactiveToActive_.at(std::distance(inactiveLayerSetIndices_.cbegin(), found))
            .push_back(i);  // TODO: move to operator[] once finished
      }

      LogTrace("PixelInactiveAreaFinder")
          << " inactive layer set " << inactiveLayerSet.first << "+" << inactiveLayerSet.second;
    }

    ++i;
  }

#ifdef EDM_ML_DEBUG
  LogDebug("PixelInactiveAreaFinder") << "All inactive layer sets";
  for (const auto& idxPair : inactiveLayerSetIndices_) {
    LogTrace("PixelInactiveAreaFinder") << " " << inactiveLayers_[idxPair.first] << "+"
                                        << inactiveLayers_[idxPair.second];
  }
#endif
}

void PixelInactiveAreaFinder::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<std::vector<edm::InputTag>>("inactivePixelDetectorLabels",
                                       std::vector<edm::InputTag>{{edm::InputTag("siPixelDigis")}})
      ->setComment("One or more DetIdCollections of modules to mask on the fly for a given event");
  desc.add<std::vector<edm::InputTag>>("badPixelFEDChannelCollectionLabels",
                                       std::vector<edm::InputTag>{{edm::InputTag("siPixelDigis")}})
      ->setComment("One or more PixelFEDChannelCollections of modules+ROCs to mask on the fly for a given event");
  desc.add<bool>("ignoreSingleFPixPanelModules", false);

  desc.addUntracked<bool>("debug", false);
  desc.addUntracked<bool>("createPlottingFiles", false);
}

PixelInactiveAreaFinder::InactiveAreas PixelInactiveAreaFinder::inactiveAreas(const edm::Event& iEvent,
                                                                              const edm::EventSetup& iSetup) {
  // Set data to handles
  trackerGeometry_ = &iSetup.getData(trackerGeometryToken_);
  trackerTopology_ = &iSetup.getData(trackerTopologyToken_);

  // assign data to instance variables
  updatePixelDets(iSetup);

  // clear the list of bad pixel modules at each event!
  badPixelDetsBarrel_.clear();
  badPixelDetsEndcap_.clear();
  getBadPixelDets(iEvent, iSetup);

  //write files for plotting
  if (createPlottingFiles_) {
    createPlottingFiles();
  }

  // find detGroupSpans ie phi,r,z limits for detector detGroups that are not working
  // returns pair where first is barrel spans and second endcap spans
  DetGroupSpanContainerPair cspans = detGroupSpans();

  // map spans to a vector with consistent indexing with inactiveLayers_ and inactiveLayerSetIndices_
  // TODO: try to move the inner logic towards this direction as well
  std::vector<DetGroupSpanContainer> spans(inactiveLayers_.size());

  auto doWork = [&](const DetGroupSpanContainer& container) {
    for (const auto& span : container) {
      const auto subdet = span.subdetId == PixelSubdetector::PixelBarrel ? GeomDetEnumerators::PixelBarrel
                                                                         : GeomDetEnumerators::PixelEndcap;
      const auto side = (subdet == GeomDetEnumerators::PixelBarrel
                             ? TrackerDetSide::Barrel
                             : (span.zSpan.first < 0 ? TrackerDetSide::NegEndcap : TrackerDetSide::PosEndcap));
      const auto layer = subdet == GeomDetEnumerators::PixelBarrel ? span.layer : span.disk;
      auto found = std::find(inactiveLayers_.begin(), inactiveLayers_.end(), SeedingLayerId(subdet, side, layer));
      if (found != inactiveLayers_.end()) {  // it is possible that this layer is ignored by the configuration
        spans[std::distance(inactiveLayers_.begin(), found)].push_back(span);
      }
    }
  };
  doWork(cspans.first);
  doWork(cspans.second);

  auto ret =
      InactiveAreas(&inactiveLayers_, std::move(spans), &inactiveLayerSetIndices_, &layerSetIndexInactiveToActive_);

  if (debug_) {
    printOverlapSpans(ret);
  }

  return ret;
}

// Functions for fetching date from handles
void PixelInactiveAreaFinder::updatePixelDets(const edm::EventSetup& iSetup) {
  if (!geometryWatcher_.check(iSetup))
    return;

  // deduce the number of ladders per layer and the number of modules per ladder
  {
    // sanity checks
    if (trackerGeometry_->numberOfLayers(PixelSubdetector::PixelBarrel) != 4) {
      throw cms::Exception("NotImplemented")
          << "This module supports only a detector with 4 pixel barrel layers, the current geometry has "
          << trackerGeometry_->numberOfLayers(PixelSubdetector::PixelBarrel);
    }
    if (trackerGeometry_->numberOfLayers(PixelSubdetector::PixelEndcap) != 3) {
      throw cms::Exception("NotImplemented")
          << "This module supports only a detector with 3 pixel forward disks, the current geometry has "
          << trackerGeometry_->numberOfLayers(PixelSubdetector::PixelEndcap);
    }

    std::array<std::array<unsigned short, 100>, 4> counts = {};  // assume at most 100 ladders per layer
    for (const auto& det : trackerGeometry_->detsPXB()) {
      const auto layer = trackerTopology_->layer(det->geographicalId());
      const auto ladder = trackerTopology_->pxbLadder(det->geographicalId());
      if (ladder >= 100) {
        throw cms::Exception("LogicError")
            << "Got a ladder with number " << ladder
            << " while the expected maximum was 100; either something is wrong or the maximum has to be increased.";
      }
      counts[layer - 1][ladder - 1] += 1;  // numbering of layer and ladder starts at 1
    }

    // take number of modules per ladder from the first ladder of the first layer (such better exist)
    // other ladders better have the same number
    nModulesPerLadder = counts[0][0];
    if (nModulesPerLadder == 0) {
      throw cms::Exception("LogicError") << "Ladder 1 of layer 1 has 0 modules, something fishy is going on.";
    }

    LogDebug("PixelInactiveAreaFinder") << "Number of modules per ladder " << nModulesPerLadder
                                        << "; below are number of ladders per layer";

    // number of ladders
    for (unsigned layer = 0; layer < 4; ++layer) {
      nBPixLadders[layer] =
          std::count_if(counts[layer].begin(), counts[layer].end(), [](unsigned short val) { return val > 0; });
      LogTrace("PixelInactiveAreaFinder")
          << "BPix layer " << (layer + 1) << " has " << nBPixLadders[layer] << " ladders";

      auto fail = std::count_if(counts[layer].begin(), counts[layer].end(), [&](unsigned short val) {
        return val != nModulesPerLadder && val > 0;
      });
      if (fail != 0) {
        throw cms::Exception("LogicError")
            << "Layer " << (layer + 1) << " had " << fail
            << " ladders whose number of modules/ladder differed from the ladder 1 of layer 1 (" << nModulesPerLadder
            << "). Something fishy is going on.";
      }
    }
  }

  // don't bother with the rest if not needed
  if (!createPlottingFiles_)
    return;

  pixelDetsBarrel_.clear();
  pixelDetsEndcap_.clear();

  for (auto const& geomDetPtr : trackerGeometry_->detsPXB()) {
    if (geomDetPtr->geographicalId().subdetId() == PixelSubdetector::PixelBarrel) {
      pixelDetsBarrel_.push_back(geomDetPtr->geographicalId().rawId());
    }
  }
  for (auto const& geomDetPtr : trackerGeometry_->detsPXF()) {
    if (geomDetPtr->geographicalId().subdetId() == PixelSubdetector::PixelEndcap) {
      pixelDetsEndcap_.push_back(geomDetPtr->geographicalId().rawId());
    }
  }
  std::sort(pixelDetsBarrel_.begin(), pixelDetsBarrel_.end());
  std::sort(pixelDetsEndcap_.begin(), pixelDetsEndcap_.end());
}
void PixelInactiveAreaFinder::getBadPixelDets(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto addDetId = [&](const auto id) {
    const auto detid = DetId(id);
    const auto subdet = detid.subdetId();
    if (subdet == PixelSubdetector::PixelBarrel) {
      badPixelDetsBarrel_.push_back(detid.rawId());
    } else if (subdet == PixelSubdetector::PixelEndcap) {
      badPixelDetsEndcap_.push_back(detid.rawId());
    }
  };

  // SiPixelQuality
  auto const& pixelQuality = iSetup.getData(pixelQualityToken_);

  for (auto const& disabledModule : pixelQuality.getBadComponentList()) {
    addDetId(disabledModule.DetID);
  }

  // dynamic bad modules
  for (const auto& token : inactivePixelDetectorTokens_) {
    edm::Handle<DetIdCollection> detIds;
    iEvent.getByToken(token, detIds);
    for (const auto& id : *detIds) {
      addDetId(id);
    }
  }

  // dynamic bad ROCs ("Fed25")
  // TODO: consider moving to finer-grained areas for inactive ROCs
  for (const auto& token : badPixelFEDChannelsTokens_) {
    edm::Handle<PixelFEDChannelCollection> pixelFEDChannelCollectionHandle;
    iEvent.getByToken(token, pixelFEDChannelCollectionHandle);
    for (const auto& disabledChannels : *pixelFEDChannelCollectionHandle) {
      addDetId(disabledChannels.detId());
    }
  }

  // remove duplicates
  std::sort(badPixelDetsBarrel_.begin(), badPixelDetsBarrel_.end());
  std::sort(badPixelDetsEndcap_.begin(), badPixelDetsEndcap_.end());
  badPixelDetsBarrel_.erase(std::unique(badPixelDetsBarrel_.begin(), badPixelDetsBarrel_.end()),
                            badPixelDetsBarrel_.end());
  badPixelDetsEndcap_.erase(std::unique(badPixelDetsEndcap_.begin(), badPixelDetsEndcap_.end()),
                            badPixelDetsEndcap_.end());
}
// Printing functions
void PixelInactiveAreaFinder::detInfo(const det_t& det, Stream& ss) {
  using std::fixed;
  using std::left;
  using std::noshowpos;
  using std::right;
  using std::setfill;
  using std::setprecision;
  using std::setw;
  using std::showpos;
  using std::tie;
  std::string deli = "; ";
  ss << "id:[" << det << "]" << deli;
  ss << "subdetid:[" << DetId(det).subdetId() << "]" << deli;
  if (DetId(det).subdetId() == PixelSubdetector::PixelBarrel) {
    unsigned int layer = trackerTopology_->pxbLayer(DetId(det));
    unsigned int ladder = trackerTopology_->pxbLadder(DetId(det));
    unsigned int module = trackerTopology_->pxbModule(DetId(det));
    ss << "layer:[" << layer << "]" << deli << "ladder:[" << right << setw(2) << ladder << "]" << deli << "module:["
       << module << "]" << deli;
  } else if (DetId(det).subdetId() == PixelSubdetector::PixelEndcap) {
    unsigned int disk = trackerTopology_->pxfDisk(DetId(det));
    unsigned int blade = trackerTopology_->pxfBlade(DetId(det));
    unsigned int panel = trackerTopology_->pxfPanel(DetId(det));
    ss << left << setw(6) << "disk:"
       << "[" << right << disk << "]" << deli << left << setw(7) << "blade:"
       << "[" << setw(2) << right << blade << "]" << deli << left << setw(7) << "panel:"
       << "[" << right << panel << "]" << deli;
  }
  float phiA, phiB, zA, zB, rA, rB;
  auto detSurface = trackerGeometry_->idToDet(DetId(det))->surface();
  tie(phiA, phiB) = detSurface.phiSpan();
  tie(zA, zB) = detSurface.zSpan();
  tie(rA, rB) = detSurface.rSpan();
  ss << setprecision(16) << fixed << showpos << setfill(' ') << "phi:[" << right << setw(12) << phiA << "," << left
     << setw(12) << phiB << "]" << deli << "z:[" << right << setw(7) << zA << "," << left << setw(7) << zB << "]"
     << deli << noshowpos << "r:[" << right << setw(10) << rA << "," << left << setw(10) << rB << "]" << deli;
}
void PixelInactiveAreaFinder::printPixelDets() {
  edm::LogPrint("PixelInactiveAreaFinder") << "Barrel detectors:";
  Stream ss;
  for (auto const& det : pixelDetsBarrel_) {
    detInfo(det, ss);
    edm::LogPrint("PixelInactiveAreaFinder") << ss.str();
    ss.str(std::string());
  }
  edm::LogPrint("PixelInactiveAreaFinder") << "Endcap detectors;";
  for (auto const& det : pixelDetsEndcap_) {
    detInfo(det, ss);
    edm::LogPrint("PixelInactiveAreaFinder") << ss.str();
    ss.str(std::string());
  }
}
void PixelInactiveAreaFinder::printBadPixelDets() {
  edm::LogPrint("PixelInactiveAreaFinder") << "Bad barrel detectors:";
  Stream ss;
  for (auto const& det : badPixelDetsBarrel_) {
    detInfo(det, ss);
    edm::LogPrint("PixelInactiveAreaFinder") << ss.str();
    ss.str(std::string());
  }
  edm::LogPrint("PixelInactiveAreaFinder") << "Endcap detectors;";
  for (auto const& det : badPixelDetsEndcap_) {
    detInfo(det, ss);
    edm::LogPrint("PixelInactiveAreaFinder") << ss.str();
    ss.str(std::string());
  }
}
void PixelInactiveAreaFinder::printBadDetGroups() {
  DetGroupContainer badDetGroupsBar = badDetGroupsBarrel();
  DetGroupContainer badDetGroupsEnd = badDetGroupsEndcap();
  Stream ss;
  for (auto const& detGroup : badDetGroupsBar) {
    ss << std::setfill(' ') << std::left << std::setw(16) << "DetGroup:";
    DetGroupSpan cspan;
    getPhiSpanBarrel(detGroup, cspan);
    getZSpan(detGroup, cspan);
    getRSpan(detGroup, cspan);
    detGroupSpanInfo(cspan, ss);
    ss << std::endl;
    for (auto const& det : detGroup) {
      detInfo(det, ss);
      ss << std::endl;
    }
    ss << std::endl;
  }
  for (auto const& detGroup : badDetGroupsEnd) {
    ss << std::setfill(' ') << std::left << std::setw(16) << "DetGroup:";
    DetGroupSpan cspan;
    getPhiSpanEndcap(detGroup, cspan);
    getZSpan(detGroup, cspan);
    getRSpan(detGroup, cspan);
    detGroupSpanInfo(cspan, ss);
    ss << std::endl;
    for (auto const& det : detGroup) {
      detInfo(det, ss);
      ss << std::endl;
    }
    ss << std::endl;
  }
  edm::LogPrint("PixelInactiveAreaFinder") << ss.str();
}
void PixelInactiveAreaFinder::printBadDetGroupSpans() {
  DetGroupSpanContainerPair cspans = detGroupSpans();
  Stream ss;
  for (auto const& cspan : cspans.first) {
    detGroupSpanInfo(cspan, ss);
    ss << std::endl;
  }
  for (auto const& cspan : cspans.second) {
    detGroupSpanInfo(cspan, ss);
    ss << std::endl;
  }
  edm::LogPrint("PixelInactiveAreaFinder") << ss.str();
}
void PixelInactiveAreaFinder::createPlottingFiles() {
  // All detectors to file DETECTORS
  Stream ss;
  std::ofstream fsDet("DETECTORS.txt");
  for (auto const& det : pixelDetsBarrel_) {
    detInfo(det, ss);
    ss << std::endl;
  }
  edm::LogPrint("PixelInactiveAreaFinder") << "Endcap detectors;";
  for (auto const& det : pixelDetsEndcap_) {
    detInfo(det, ss);
    ss << std::endl;
  }
  fsDet << ss.rdbuf();
  ss.str(std::string());
  // Bad detectors
  std::ofstream fsBadDet("BADDETECTORS.txt");
  for (auto const& det : badPixelDetsBarrel_) {
    detInfo(det, ss);
    ss << std::endl;
  }
  for (auto const& det : badPixelDetsEndcap_) {
    detInfo(det, ss);
    ss << std::endl;
  }
  fsBadDet << ss.rdbuf();
  ss.str(std::string());
  // detgroupspans
  std::ofstream fsSpans("DETGROUPSPANS.txt");
  DetGroupSpanContainerPair cspans = detGroupSpans();
  for (auto const& cspan : cspans.first) {
    detGroupSpanInfo(cspan, ss);
    ss << std::endl;
  }
  for (auto const& cspan : cspans.second) {
    detGroupSpanInfo(cspan, ss);
    ss << std::endl;
  }
  fsSpans << ss.rdbuf();
  ss.str(std::string());
}
// Functions for finding bad detGroups
bool PixelInactiveAreaFinder::detWorks(det_t det) {
  return std::find(badPixelDetsBarrel_.begin(), badPixelDetsBarrel_.end(), det) == badPixelDetsBarrel_.end() &&
         std::find(badPixelDetsEndcap_.begin(), badPixelDetsEndcap_.end(), det) == badPixelDetsEndcap_.end();
}
PixelInactiveAreaFinder::DetGroup PixelInactiveAreaFinder::badAdjecentDetsBarrel(const det_t& det) {
  using std::remove_if;

  DetGroup adj;
  auto const tTopo = trackerTopology_;
  auto const& detId = DetId(det);
  unsigned int layer = tTopo->pxbLayer(detId);
  unsigned int ladder = tTopo->pxbLadder(detId);
  unsigned int module = tTopo->pxbModule(detId);
  unsigned int nLads = nBPixLadders[layer];
  //add detectors from next and previous ladder
  adj.push_back(tTopo->pxbDetId(layer, ((ladder - 1) + 1) % nLads + 1, module)());
  adj.push_back(tTopo->pxbDetId(layer, ((ladder - 1) - 1 + nLads) % nLads + 1, module)());
  //add adjecent detectors from same ladder
  if (module == 1) {
    adj.push_back(tTopo->pxbDetId(layer, ladder, module + 1)());
  } else if (module == nModulesPerLadder) {
    adj.push_back(tTopo->pxbDetId(layer, ladder, module - 1)());
  } else {
    adj.push_back(tTopo->pxbDetId(layer, ladder, module + 1)());
    adj.push_back(tTopo->pxbDetId(layer, ladder, module - 1)());
  }
  //remove working detectors from list
  adj.erase(remove_if(adj.begin(), adj.end(), [&](auto c) { return this->detWorks(c); }), adj.end());
  return adj;
}
PixelInactiveAreaFinder::DetGroup PixelInactiveAreaFinder::badAdjecentDetsEndcap(const det_t& det) {
  // this might be faster if adjecent
  using std::ignore;
  using std::tie;
  DetGroup adj;
  Span_t phiSpan, phiSpanComp;
  float z, zComp;
  unsigned int disk, diskComp;
  auto const& detSurf = trackerGeometry_->idToDet(DetId(det))->surface();
  phiSpan = detSurf.phiSpan();
  tie(z, ignore) = detSurf.zSpan();
  disk = trackerTopology_->pxfDisk(DetId(det));
  // add detectors from same disk whose phi ranges overlap to the adjecent list
  for (auto const& detComp : badPixelDetsEndcap_) {
    auto const& detIdComp = DetId(detComp);
    auto const& detSurfComp = trackerGeometry_->idToDet(detIdComp)->surface();
    diskComp = trackerTopology_->pxfDisk(detIdComp);
    phiSpanComp = detSurfComp.phiSpan();
    tie(zComp, ignore) = detSurfComp.zSpan();
    if (det != detComp && disk == diskComp && z * zComp > 0 && phiRangesOverlap(phiSpan, phiSpanComp)) {
      adj.push_back(detComp);
    }
  }
  return adj;
}
PixelInactiveAreaFinder::DetGroup PixelInactiveAreaFinder::reachableDetGroup(const det_t& initDet,
                                                                             DetectorSet& foundDets) {
  DetGroup reachableDetGroup;
  std::queue<det_t> workQueue;
  det_t workDet;
  DetGroup badAdjDets;
  foundDets.insert(initDet);
  workQueue.push(initDet);
  reachableDetGroup.push_back(initDet);
  while (!workQueue.empty()) {
    workDet = workQueue.front();
    workQueue.pop();
    if (DetId(workDet).subdetId() == PixelSubdetector::PixelBarrel) {
      badAdjDets = this->badAdjecentDetsBarrel(workDet);
    } else if (DetId(workDet).subdetId() == PixelSubdetector::PixelEndcap) {
      badAdjDets = this->badAdjecentDetsEndcap(workDet);
    } else {
      badAdjDets = {};
    }
    for (auto const& badDet : badAdjDets) {
      if (foundDets.find(badDet) == foundDets.end()) {
        reachableDetGroup.push_back(badDet);
        foundDets.insert(badDet);
        workQueue.push(badDet);
      }
    }
  }
  return reachableDetGroup;
}
PixelInactiveAreaFinder::DetGroupContainer PixelInactiveAreaFinder::badDetGroupsBarrel() {
  DetGroupContainer detGroups;
  DetectorSet foundDets;
  for (auto const& badDet : badPixelDetsBarrel_) {
    if (foundDets.find(badDet) == foundDets.end()) {
      detGroups.push_back(this->reachableDetGroup(badDet, foundDets));
    }
  }
  return detGroups;
}
PixelInactiveAreaFinder::DetGroupContainer PixelInactiveAreaFinder::badDetGroupsEndcap() {
  DetGroupContainer detGroups;
  DetectorSet foundDets;
  for (auto const& badDet : badPixelDetsEndcap_) {
    if (foundDets.find(badDet) == foundDets.end()) {
      auto adjacentDets = this->reachableDetGroup(badDet, foundDets);
      if (ignoreSingleFPixPanelModules_ && adjacentDets.size() == 1) {
        // size==1 means that only a single panel of a blade was inactive
        // because of the large overlap with the other panel (i.e.
        // redundancy in the detectory) ignoring these may help to decrease fakes
        continue;
      }
      detGroups.push_back(adjacentDets);
    }
  }
  return detGroups;
}
// Functions for finding DetGroupSpans
void PixelInactiveAreaFinder::getPhiSpanBarrel(const DetGroup& detGroup, DetGroupSpan& cspan) {
  // find phiSpan using ordered vector of unique ladders in detGroup
  if (detGroup.empty()) {
    cspan = DetGroupSpan();
    return;
  } else {
    cspan.layer = trackerTopology_->pxbLayer(DetId(detGroup[0]));
    cspan.disk = 0;
  }
  using uint = unsigned int;
  using LadderSet = std::set<uint>;
  using LadVec = std::vector<uint>;
  LadderSet lads;
  for (auto const& det : detGroup) {
    lads.insert(trackerTopology_->pxbLadder(DetId(det)));
  }
  LadVec ladv(lads.begin(), lads.end());
  uint nLadders = nBPixLadders[cspan.layer];
  // find start ladder of detGroup
  uint i = 0;
  uint currentLadder = ladv[0];
  uint previousLadder = ladv[(ladv.size() + i - 1) % ladv.size()];
  // loop until discontinuity is found from vector
  while ((nLadders + currentLadder - 1) % nLadders == previousLadder) {
    ++i;
    currentLadder = ladv[i % ladv.size()];
    previousLadder = ladv[(ladv.size() + i - 1) % ladv.size()];
    if (i == ladv.size()) {
      cspan.phiSpan.first = std::numeric_limits<float>::epsilon();
      cspan.phiSpan.second = -std::numeric_limits<float>::epsilon();
      return;
    }
  }
  uint startLadder = currentLadder;
  uint endLadder = previousLadder;
  auto detStart = trackerTopology_->pxbDetId(cspan.layer, startLadder, 1);
  auto detEnd = trackerTopology_->pxbDetId(cspan.layer, endLadder, 1);
  cspan.phiSpan.first = trackerGeometry_->idToDet(detStart)->surface().phiSpan().first;
  cspan.phiSpan.second = trackerGeometry_->idToDet(detEnd)->surface().phiSpan().second;
}
void PixelInactiveAreaFinder::getPhiSpanEndcap(const DetGroup& detGroup, DetGroupSpan& cspan) {
  // this is quite naive/bruteforce method
  // 1) it starts by taking one detector from detGroup and starts to compare it to others
  // 2) when it finds overlapping detector in clockwise direction it starts comparing
  //    found detector to others
  // 3) search stops until no overlapping detectors in clockwise detector or all detectors
  //    have been work detector
  Stream ss;
  bool found = false;
  auto const tGeom = trackerGeometry_;
  DetGroup::const_iterator startDetIter = detGroup.begin();
  Span_t phiSpan, phiSpanComp;
  unsigned int counter = 0;
  while (!found) {
    phiSpan = tGeom->idToDet(DetId(*startDetIter))->surface().phiSpan();
    for (DetGroup::const_iterator compDetIter = detGroup.begin(); compDetIter != detGroup.end(); ++compDetIter) {
      phiSpanComp = tGeom->idToDet(DetId(*compDetIter))->surface().phiSpan();
      if (phiRangesOverlap(phiSpan, phiSpanComp) && phiMoreClockwise(phiSpanComp.first, phiSpan.first) &&
          startDetIter != compDetIter) {
        ++counter;
        if (counter > detGroup.size()) {
          cspan.phiSpan.first = std::numeric_limits<float>::epsilon();
          cspan.phiSpan.second = -std::numeric_limits<float>::epsilon();
          return;
        }
        startDetIter = compDetIter;
        break;
      } else if (compDetIter == detGroup.end() - 1) {
        found = true;
      }
    }
  }
  cspan.phiSpan.first = phiSpan.first;
  // second with same method}
  found = false;
  DetGroup::const_iterator endDetIter = detGroup.begin();
  counter = 0;
  while (!found) {
    phiSpan = tGeom->idToDet(DetId(*endDetIter))->surface().phiSpan();
    for (DetGroup::const_iterator compDetIter = detGroup.begin(); compDetIter != detGroup.end(); ++compDetIter) {
      phiSpanComp = tGeom->idToDet(DetId(*compDetIter))->surface().phiSpan();
      if (phiRangesOverlap(phiSpan, phiSpanComp) && phiMoreCounterclockwise(phiSpanComp.second, phiSpan.second) &&
          endDetIter != compDetIter) {
        ++counter;
        if (counter > detGroup.size()) {
          cspan.phiSpan.first = std::numeric_limits<float>::epsilon();
          cspan.phiSpan.second = -std::numeric_limits<float>::epsilon();
          return;
        }
        endDetIter = compDetIter;
        break;
      } else if (compDetIter == detGroup.end() - 1) {
        found = true;
      }
    }
  }
  cspan.phiSpan.second = phiSpan.second;
}
void PixelInactiveAreaFinder::getZSpan(const DetGroup& detGroup, DetGroupSpan& cspan) {
  auto cmpFun = [this](det_t detA, det_t detB) {
    return trackerGeometry_->idToDet(DetId(detA))->surface().zSpan().first <
           trackerGeometry_->idToDet(DetId(detB))->surface().zSpan().first;
  };

  auto minmaxIters = std::minmax_element(detGroup.begin(), detGroup.end(), cmpFun);
  cspan.zSpan.first = trackerGeometry_->idToDet(DetId(*(minmaxIters.first)))->surface().zSpan().first;
  cspan.zSpan.second = trackerGeometry_->idToDet(DetId(*(minmaxIters.second)))->surface().zSpan().second;
}
void PixelInactiveAreaFinder::getRSpan(const DetGroup& detGroup, DetGroupSpan& cspan) {
  auto cmpFun = [this](det_t detA, det_t detB) {
    return trackerGeometry_->idToDet(DetId(detA))->surface().rSpan().first <
           trackerGeometry_->idToDet(DetId(detB))->surface().rSpan().first;
  };

  auto minmaxIters = std::minmax_element(detGroup.begin(), detGroup.end(), cmpFun);
  cspan.rSpan.first = trackerGeometry_->idToDet(DetId(*(minmaxIters.first)))->surface().rSpan().first;
  cspan.rSpan.second = trackerGeometry_->idToDet(DetId(*(minmaxIters.second)))->surface().rSpan().second;
}
void PixelInactiveAreaFinder::getSpan(const DetGroup& detGroup, DetGroupSpan& cspan) {
  auto firstDetIt = detGroup.begin();
  if (firstDetIt != detGroup.end()) {
    cspan.subdetId = DetId(*firstDetIt).subdetId();
    if (cspan.subdetId == 1) {
      cspan.layer = trackerTopology_->pxbLayer(DetId(*firstDetIt));
      cspan.disk = 0;
      getPhiSpanBarrel(detGroup, cspan);
    } else if (cspan.subdetId == 2) {
      cspan.disk = trackerTopology_->pxfDisk(DetId(*firstDetIt));
      cspan.layer = 0;
      getPhiSpanEndcap(detGroup, cspan);
    }
    getZSpan(detGroup, cspan);
    getRSpan(detGroup, cspan);
  }
}
PixelInactiveAreaFinder::DetGroupSpanContainerPair PixelInactiveAreaFinder::detGroupSpans() {
  DetGroupSpanContainer cspansBarrel;
  DetGroupSpanContainer cspansEndcap;
  DetGroupContainer badDetGroupsBar = badDetGroupsBarrel();
  DetGroupContainer badDetGroupsEnd = badDetGroupsEndcap();
  for (auto const& detGroup : badDetGroupsBar) {
    DetGroupSpan cspan;
    getSpan(detGroup, cspan);
    cspansBarrel.push_back(cspan);
  }
  for (auto const& detGroup : badDetGroupsEnd) {
    DetGroupSpan cspan;
    getSpan(detGroup, cspan);
    cspansEndcap.push_back(cspan);
  }
  return DetGroupSpanContainerPair(cspansBarrel, cspansEndcap);
}
