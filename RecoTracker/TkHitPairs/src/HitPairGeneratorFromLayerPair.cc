#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"

#include "FWCore/Framework/interface/Event.h"

using namespace GeomDetEnumerators;
using namespace std;

typedef PixelRecoRange<float> Range;

namespace {
  template <class T>
  inline T sqr(T t) {
    return t * t;
  }
}  // namespace

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"

HitPairGeneratorFromLayerPair::HitPairGeneratorFromLayerPair(
    edm::ConsumesCollector iC, unsigned int inner, unsigned int outer, LayerCacheType* layerCache, unsigned int max)
    : theLayerCache(layerCache),
      theFieldToken(iC.esConsumes()),
      theMSMakerToken(iC.esConsumes()),
      theOuterLayer(outer),
      theInnerLayer(inner),
      theMaxElement(max) {}

HitPairGeneratorFromLayerPair::~HitPairGeneratorFromLayerPair() {}

// devirtualizer
#include <tuple>
namespace {

  template <typename Algo>
  struct Kernel {
    using Base = HitRZCompatibility;
    void set(Base const* a) {
      assert(a->algo() == Algo::me);
      checkRZ = reinterpret_cast<Algo const*>(a);
    }

    void operator()(int b, int e, const RecHitsSortedInPhi& innerHitsMap, bool* ok) const {
      constexpr float nSigmaRZ = 3.46410161514f;  // std::sqrt(12.f);
      for (int i = b; i != e; ++i) {
        Range allowed = checkRZ->range(innerHitsMap.u[i]);
        float vErr = nSigmaRZ * innerHitsMap.dv[i];
        Range hitRZ(innerHitsMap.v[i] - vErr, innerHitsMap.v[i] + vErr);
        Range crossRange = allowed.intersection(hitRZ);
        ok[i - b] = !crossRange.empty();
      }
    }
    Algo const* checkRZ;
  };

  template <typename... Args>
  using Kernels = std::tuple<Kernel<Args>...>;

}  // namespace

bool HitPairGeneratorFromLayerPair::hitPairs(const TrackingRegion& region,
                                             OrderedHitPairs& result,
                                             const edm::Event& iEvent,
                                             const edm::EventSetup& iSetup,
                                             Layers layers) {
  auto const& ds = doublets(region, iEvent, iSetup, layers);
  if (not ds) {
    return false;
  }
  for (std::size_t i = 0; i != ds->size(); ++i) {
    result.push_back(OrderedHitPair(ds->hit(i, HitDoublets::inner), ds->hit(i, HitDoublets::outer)));
  }
  if (theMaxElement != 0 && result.size() >= theMaxElement) {
    result.clear();
    edm::LogError("TooManyPairs") << "number of pairs exceed maximum, no pairs produced";
    return false;
  }
  return true;
}

std::optional<HitDoublets> HitPairGeneratorFromLayerPair::doublets(const TrackingRegion& region,
                                                                   const edm::Event& iEvent,
                                                                   const edm::EventSetup& iSetup,
                                                                   const Layer& innerLayer,
                                                                   const Layer& outerLayer,
                                                                   LayerCacheType& layerCache) {
  const RecHitsSortedInPhi& innerHitsMap = layerCache(innerLayer, region);
  if (innerHitsMap.empty())
    return HitDoublets(innerHitsMap, innerHitsMap);

  const RecHitsSortedInPhi& outerHitsMap = layerCache(outerLayer, region);
  if (outerHitsMap.empty())
    return HitDoublets(innerHitsMap, outerHitsMap);
  const auto& field = iSetup.getData(theFieldToken);
  const auto& msmaker = iSetup.getData(theMSMakerToken);
  HitDoublets result(innerHitsMap, outerHitsMap);
  result.reserve(std::max(innerHitsMap.size(), outerHitsMap.size()));
  bool succeeded = doublets(region,
                            *innerLayer.detLayer(),
                            *outerLayer.detLayer(),
                            innerHitsMap,
                            outerHitsMap,
                            field,
                            msmaker,
                            theMaxElement,
                            result);
  if (succeeded) {
    return result;
  } else {
    return std::nullopt;
  }
}

bool HitPairGeneratorFromLayerPair::doublets(const TrackingRegion& region,
                                             const DetLayer& innerHitDetLayer,
                                             const DetLayer& outerHitDetLayer,
                                             const RecHitsSortedInPhi& innerHitsMap,
                                             const RecHitsSortedInPhi& outerHitsMap,
                                             const MagneticField& field,
                                             const MultipleScatteringParametrisationMaker& msmaker,
                                             const unsigned int theMaxElement,
                                             HitDoublets& result) {
  //  HitDoublets result(innerHitsMap,outerHitsMap); result.reserve(std::max(innerHitsMap.size(),outerHitsMap.size()));
  typedef RecHitsSortedInPhi::Hit Hit;
  InnerDeltaPhi deltaPhi(outerHitDetLayer, innerHitDetLayer, region, field, msmaker);

  // std::cout << "layers " << theInnerLayer.detLayer()->seqNum()  << " " << outerLayer.detLayer()->seqNum() << std::endl;

  // constexpr float nSigmaRZ = std::sqrt(12.f);
  constexpr float nSigmaPhi = 3.f;
  for (int io = 0; io != int(outerHitsMap.theHits.size()); ++io) {
    if (!deltaPhi.prefilter(outerHitsMap.x[io], outerHitsMap.y[io]))
      continue;
    Hit const& ohit = outerHitsMap.theHits[io].hit();
    PixelRecoRange<float> phiRange =
        deltaPhi(outerHitsMap.x[io], outerHitsMap.y[io], outerHitsMap.z[io], nSigmaPhi * outerHitsMap.drphi[io]);

    if (phiRange.empty())
      continue;

    std::unique_ptr<const HitRZCompatibility> checkRZ =
        region.checkRZ(&innerHitDetLayer,
                       ohit,
                       &outerHitDetLayer,
                       outerHitsMap.rv(io),
                       outerHitsMap.z[io],
                       outerHitsMap.isBarrel ? outerHitsMap.du[io] : outerHitsMap.dv[io],
                       outerHitsMap.isBarrel ? outerHitsMap.dv[io] : outerHitsMap.du[io]);
    if (!checkRZ)
      continue;

    Kernels<HitZCheck, HitRCheck, HitEtaCheck> kernels;

    auto innerRange = innerHitsMap.doubleRange(phiRange.min(), phiRange.max());
    LogDebug("HitPairGeneratorFromLayerPair")
        << "preparing for combination of: " << innerRange[1] - innerRange[0] + innerRange[3] - innerRange[2]
        << " inner and: " << outerHitsMap.theHits.size() << " outter";
    for (int j = 0; j < 3; j += 2) {
      auto b = innerRange[j];
      auto e = innerRange[j + 1];
      if (e == b)
        continue;
      bool ok[e - b];
      switch (checkRZ->algo()) {
        case (HitRZCompatibility::zAlgo):
          std::get<0>(kernels).set(checkRZ.get());
          std::get<0>(kernels)(b, e, innerHitsMap, ok);
          break;
        case (HitRZCompatibility::rAlgo):
          std::get<1>(kernels).set(checkRZ.get());
          std::get<1>(kernels)(b, e, innerHitsMap, ok);
          break;
        case (HitRZCompatibility::etaAlgo):
          std::get<2>(kernels).set(checkRZ.get());
          std::get<2>(kernels)(b, e, innerHitsMap, ok);
          break;
      }
      for (int i = 0; i != e - b; ++i) {
        if (!ok[i])
          continue;
        if (theMaxElement != 0 && result.size() >= theMaxElement) {
          result.clear();
          edm::LogError("TooManyPairs") << "number of pairs exceed maximum, no pairs produced";
          return false;
        }
        result.add(b + i, io);
      }
    }
  }
  LogDebug("HitPairGeneratorFromLayerPair") << " total number of pairs provided back: " << result.size();
  result.shrink_to_fit();
  return true;
}
