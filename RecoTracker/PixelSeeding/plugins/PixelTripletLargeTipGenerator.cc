#include "RecoTracker/PixelSeeding/plugins/PixelTripletLargeTipGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/PixelSeeding/interface/ThirdHitPredictionFromCircle.h"
#include "RecoTracker/PixelSeeding/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/PixelSeeding/plugins/ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

#include "MatchedHitRZCorrectionFromBending.h"
#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "DataFormats/Math/interface/normalizedPhi.h"

#include "CommonTools/Utils/interface/DynArray.h"

using namespace std;

using Range = PixelRecoRange<float>;
using HelixRZ = ThirdHitPredictionFromCircle::HelixRZ;

namespace {
  struct LayerRZPredictions {
    ThirdHitRZPrediction<PixelRecoLineRZ> line;
    ThirdHitRZPrediction<HelixRZ> helix1, helix2;
    MatchedHitRZCorrectionFromBending rzPositionFixup;
    ThirdHitCorrection correction;
  };
}  // namespace

constexpr double nSigmaRZ = 3.4641016151377544;  // sqrt(12.)
constexpr float nSigmaPhi = 3.;
constexpr float fnSigmaRZ = nSigmaRZ;

PixelTripletLargeTipGenerator::PixelTripletLargeTipGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : HitTripletGeneratorFromPairAndLayers(cfg),
      useFixedPreFiltering(cfg.getParameter<bool>("useFixedPreFiltering")),
      extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
      extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
      useMScat(cfg.getParameter<bool>("useMultScattering")),
      useBend(cfg.getParameter<bool>("useBending")),
      dphi(useFixedPreFiltering ? cfg.getParameter<double>("phiPreFiltering") : 0),
      trackerTopologyESToken_(iC.esConsumes()),
      fieldESToken_(iC.esConsumes()) {
  if (useMScat) {
    msmakerESToken_ = iC.esConsumes();
  }
}

PixelTripletLargeTipGenerator::~PixelTripletLargeTipGenerator() {}

void PixelTripletLargeTipGenerator::fillDescriptions(edm::ParameterSetDescription& desc) {
  HitTripletGeneratorFromPairAndLayers::fillDescriptions(desc);
  // Defaults for the extraHit*tolerance are set to 0 here since that
  // was already the case in practice in all offline occurrances.
  desc.add<double>("extraHitRPhitolerance", 0);  // default in old python was 0.032
  desc.add<double>("extraHitRZtolerance", 0);    // default in old python was 0.037
  desc.add<bool>("useMultScattering", true);
  desc.add<bool>("useBending", true);
  desc.add<bool>("useFixedPreFiltering", false);
  desc.add<double>("phiPreFiltering", 0.3);
}

// Disable bitwise-instead-of-logical warning, see discussion in
// https://github.com/cms-sw/cmssw/issues/39105

#if defined(__clang__) && defined(__has_warning)
#if __has_warning("-Wbitwise-instead-of-logical")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#endif
#endif

namespace {
  inline bool intersect(Range& range, const Range& second) {
    if ((range.min() > second.max()) | (range.max() < second.min()))
      return false;
    if (range.first < second.min())
      range.first = second.min();
    if (range.second > second.max())
      range.second = second.max();
    return range.first < range.second;
  }
}  // namespace

#if defined(__clang__) && defined(__has_warning)
#if __has_warning("-Wbitwise-instead-of-logical")
#pragma clang diagnostic pop
#endif
#endif

void PixelTripletLargeTipGenerator::hitTriplets(const TrackingRegion& region,
                                                OrderedHitTriplets& result,
                                                const edm::Event& ev,
                                                const edm::EventSetup& es,
                                                const SeedingLayerSetsHits::SeedingLayerSet& pairLayers,
                                                const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) {
  auto const& doublets = thePairGenerator->doublets(region, ev, es, pairLayers);

  if (doublets.empty())
    return;

  assert(theLayerCache);
  hitTriplets(region, result, ev, es, doublets, thirdLayers, nullptr, *theLayerCache);
}

void PixelTripletLargeTipGenerator::hitTriplets(const TrackingRegion& region,
                                                OrderedHitTriplets& result,
                                                const edm::Event& ev,
                                                const edm::EventSetup& es,
                                                const HitDoublets& doublets,
                                                const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
                                                std::vector<int>* tripletLastLayerIndex,
                                                LayerCacheType& layerCache) {
  int size = thirdLayers.size();
  const RecHitsSortedInPhi* thirdHitMap[size];
  vector<const DetLayer*> thirdLayerDetLayer(size, nullptr);
  for (int il = 0; il < size; ++il) {
    thirdHitMap[il] = &layerCache(thirdLayers[il], region);
    thirdLayerDetLayer[il] = thirdLayers[il].detLayer();
  }
  hitTriplets(region, result, es, doublets, thirdHitMap, thirdLayerDetLayer, size, tripletLastLayerIndex);
}

void PixelTripletLargeTipGenerator::hitTriplets(const TrackingRegion& region,
                                                OrderedHitTriplets& result,
                                                const edm::EventSetup& es,
                                                const HitDoublets& doublets,
                                                const RecHitsSortedInPhi** thirdHitMap,
                                                const std::vector<const DetLayer*>& thirdLayerDetLayer,
                                                const int nThirdLayers) {
  hitTriplets(region, result, es, doublets, thirdHitMap, thirdLayerDetLayer, nThirdLayers, nullptr);
}

void PixelTripletLargeTipGenerator::hitTriplets(const TrackingRegion& region,
                                                OrderedHitTriplets& result,
                                                const edm::EventSetup& es,
                                                const HitDoublets& doublets,
                                                const RecHitsSortedInPhi** thirdHitMap,
                                                const std::vector<const DetLayer*>& thirdLayerDetLayer,
                                                const int nThirdLayers,
                                                std::vector<int>* tripletLastLayerIndex) {
  const TrackerTopology* tTopo = &es.getData(trackerTopologyESToken_);
  const auto& field = es.getData(fieldESToken_);
  const MultipleScatteringParametrisationMaker* msmaker = nullptr;
  if (useMScat) {
    msmaker = &es.getData(msmakerESToken_);
  }

  auto outSeq = doublets.detLayer(HitDoublets::outer)->seqNum();

  using NodeInfo = KDTreeNodeInfo<unsigned int, 2>;
  std::vector<NodeInfo> layerTree;       // re-used throughout
  std::vector<unsigned int> foundNodes;  // re-used throughout
  foundNodes.reserve(100);

  declareDynArray(KDTreeLinkerAlgo<unsigned int>, nThirdLayers, hitTree);
  declareDynArray(LayerRZPredictions, nThirdLayers, mapPred);

  float rzError[nThirdLayers];  //save maximum errors

  const float maxDelphi = region.ptMin() < 0.3f ? float(M_PI) / 4.f : float(M_PI) / 8.f;  // FIXME move to config??
  const float maxphi = M_PI + maxDelphi, minphi = -maxphi;  // increase to cater for any range
  const float safePhi = M_PI - maxDelphi;                   // sideband

  for (int il = 0; il < nThirdLayers; il++) {
    auto const& hits = *thirdHitMap[il];

    const DetLayer* layer = thirdLayerDetLayer[il];
    LayerRZPredictions& predRZ = mapPred[il];
    predRZ.line.initLayer(layer);
    predRZ.helix1.initLayer(layer);
    predRZ.helix2.initLayer(layer);
    predRZ.line.initTolerance(extraHitRZtolerance);
    predRZ.helix1.initTolerance(extraHitRZtolerance);
    predRZ.helix2.initTolerance(extraHitRZtolerance);
    predRZ.rzPositionFixup = MatchedHitRZCorrectionFromBending(layer, tTopo);
    predRZ.correction.init(region.ptMin(),
                           *doublets.detLayer(HitDoublets::inner),
                           *doublets.detLayer(HitDoublets::outer),
                           *thirdLayerDetLayer[il],
                           useMScat,
                           msmaker,
                           false,
                           nullptr);

    layerTree.clear();
    float minv = 999999.0;
    float maxv = -999999.0;  // Initialise to extreme values in case no hits
    float maxErr = 0.0f;
    for (unsigned int i = 0; i != hits.size(); ++i) {
      auto angle = hits.phi(i);
      auto v = hits.gv(i);
      //use (phi,r) for endcaps rather than (phi,z)
      minv = std::min(minv, v);
      maxv = std::max(maxv, v);
      float myerr = hits.dv[i];
      maxErr = std::max(maxErr, myerr);
      layerTree.emplace_back(i, angle, v);  // save it
      // populate side-bands
      if (angle > safePhi)
        layerTree.emplace_back(i, angle - Geom::ftwoPi(), v);
      else if (angle < -safePhi)
        layerTree.emplace_back(i, angle + Geom::ftwoPi(), v);
    }
    KDTreeBox phiZ(minphi, maxphi, minv - 0.01f, maxv + 0.01f);  // declare our bounds
    //add fudge factors in case only one hit and also for floating-point inaccuracy
    hitTree[il].build(layerTree, phiZ);  // make KDtree
    rzError[il] = maxErr;                //save error
  }

  double curv = PixelRecoUtilities::curvature(1. / region.ptMin(), field);

  for (std::size_t ip = 0; ip != doublets.size(); ip++) {
    auto xi = doublets.x(ip, HitDoublets::inner);
    auto yi = doublets.y(ip, HitDoublets::inner);
    auto zi = doublets.z(ip, HitDoublets::inner);
    // auto rvi = doublets.rv(ip,HitDoublets::inner);
    auto xo = doublets.x(ip, HitDoublets::outer);
    auto yo = doublets.y(ip, HitDoublets::outer);
    auto zo = doublets.z(ip, HitDoublets::outer);
    // auto rvo = doublets.rv(ip,HitDoublets::outer);
    GlobalPoint gp1(xi, yi, zi);
    GlobalPoint gp2(xo, yo, zo);

    auto toPos = std::signbit(zo - zi);

    PixelRecoLineRZ line(gp1, gp2);
    PixelRecoPointRZ point2(gp2.perp(), zo);
    ThirdHitPredictionFromCircle predictionRPhi(gp1, gp2, extraHitRPhitolerance);

    Range generalCurvature = predictionRPhi.curvature(region.originRBound());
    if (!intersect(generalCurvature, Range(-curv, curv)))
      continue;

    for (int il = 0; il < nThirdLayers; il++) {
      if (hitTree[il].empty())
        continue;  // Don't bother if no hits
      const DetLayer* layer = thirdLayerDetLayer[il];
      bool barrelLayer = layer->isBarrel();

      if ((!barrelLayer) & (toPos != std::signbit(layer->position().z())))
        continue;

      Range curvature = generalCurvature;

      LayerRZPredictions& predRZ = mapPred[il];
      predRZ.line.initPropagator(&line);

      auto& correction = predRZ.correction;
      correction.init(line, point2, outSeq);

      Range rzRange;
      if (useBend) {
        // For the barrel region:
        // swiping the helix passing through the two points across from
        // negative to positive bending, can give us a sort of U-shaped
        // projection onto the phi-z (barrel) or r-z plane (forward)
        // so we checking minimum/maximum of all three possible extrema
        //
        // For the endcap region:
        // Checking minimum/maximum radius of the helix projection
        // onto an endcap plane, here we have to guard against
        // looping tracks, when phi(delta z) gets out of control.
        // HelixRZ::rAtZ should not follow looping tracks, but clamp
        // to the minimum reachable r with the next-best lower |curvature|.
        // So same procedure as for the barrel region can be applied.
        //
        // In order to avoid looking for potential looping tracks at all
        // we also clamp the allowed curvature range for this layer,
        // and potentially fail the layer entirely

        if (!barrelLayer) {
          Range z3s = predRZ.line.detRange();
          double z3 = z3s.first < 0 ? std::max(z3s.first, z3s.second) : std::min(z3s.first, z3s.second);
          double maxCurvature = HelixRZ::maxCurvature(&predictionRPhi, gp1.z(), gp2.z(), z3);
          if (!intersect(curvature, Range(-maxCurvature, maxCurvature)))
            continue;
        }

        HelixRZ helix1(&predictionRPhi, gp1.z(), gp2.z(), curvature.first);
        HelixRZ helix2(&predictionRPhi, gp1.z(), gp2.z(), curvature.second);

        predRZ.helix1.initPropagator(&helix1);
        predRZ.helix2.initPropagator(&helix2);

        Range rzRanges[2] = {predRZ.helix1(), predRZ.helix2()};
        predRZ.helix1.initPropagator(nullptr);
        predRZ.helix2.initPropagator(nullptr);

        rzRange.first = std::min(rzRanges[0].first, rzRanges[1].first);
        rzRange.second = std::max(rzRanges[0].second, rzRanges[1].second);

        // if the allowed curvatures include a straight line,
        // this can give us another extremum for allowed r/z
        if (curvature.first * curvature.second < 0.0) {
          Range rzLineRange = predRZ.line();
          rzRange.first = std::min(rzRange.first, rzLineRange.first);
          rzRange.second = std::max(rzRange.second, rzLineRange.second);
        }
      } else {
        rzRange = predRZ.line();
      }

      if (rzRange.first >= rzRange.second)
        continue;

      correction.correctRZRange(rzRange);

      Range phiRange;
      if (useFixedPreFiltering) {
        float phi0 = doublets.phi(ip, HitDoublets::outer);
        phiRange = Range(phi0 - dphi, phi0 + dphi);
      } else {
        Range radius;

        if (barrelLayer) {
          radius = predRZ.line.detRange();
          if (!intersect(rzRange, predRZ.line.detSize()))
            continue;
        } else {
          radius = rzRange;
          if (!intersect(radius, predRZ.line.detSize()))
            continue;
        }

        //gc: predictionRPhi uses the cosine rule to find the phi of the 3rd point at radius, assuming the pairCurvature range [-c,+c]
        if ((curvature.first < 0.0f) & (curvature.second < 0.0f)) {
          radius.swap();
        } else if ((curvature.first >= 0.0f) & (curvature.second >= 0.0f)) {
          ;
        } else {
          radius.first = radius.second;
        }
        auto phi12 = predictionRPhi.phi(curvature.first, radius.first);
        auto phi22 = predictionRPhi.phi(curvature.second, radius.second);
        phi12 = normalizedPhi(phi12);
        phi22 = proxim(phi22, phi12);
        phiRange = Range(phi12, phi22);
        phiRange.sort();
        auto rmean = radius.mean();
        phiRange.first *= rmean;
        phiRange.second *= rmean;
        correction.correctRPhiRange(phiRange);
        phiRange.first /= rmean;
        phiRange.second /= rmean;
      }

      foundNodes.clear();                                    // Now recover hits in bounding box...
      float prmin = phiRange.min(), prmax = phiRange.max();  //get contiguous range

      if (prmax - prmin > maxDelphi) {
        auto prm = phiRange.mean();
        prmin = prm - 0.5f * maxDelphi;
        prmax = prm + 0.5f * maxDelphi;
      }

      if (barrelLayer) {
        Range regMax = predRZ.line.detRange();
        Range regMin = predRZ.line(regMax.min());
        regMax = predRZ.line(regMax.max());
        correction.correctRZRange(regMin);
        correction.correctRZRange(regMax);
        if (regMax.min() < regMin.min()) {
          std::swap(regMax, regMin);
        }
        KDTreeBox phiZ(prmin, prmax, regMin.min() - fnSigmaRZ * rzError[il], regMax.max() + fnSigmaRZ * rzError[il]);
        hitTree[il].search(phiZ, foundNodes);
      } else {
        KDTreeBox phiZ(prmin, prmax, rzRange.min() - fnSigmaRZ * rzError[il], rzRange.max() + fnSigmaRZ * rzError[il]);
        hitTree[il].search(phiZ, foundNodes);
      }

      MatchedHitRZCorrectionFromBending l2rzFixup(doublets.hit(ip, HitDoublets::outer)->det()->geographicalId(), tTopo);
      MatchedHitRZCorrectionFromBending l3rzFixup = predRZ.rzPositionFixup;

      auto const& hits = *thirdHitMap[il];
      for (auto KDdata : foundNodes) {
        GlobalPoint p3 = hits.gp(KDdata);
        double p3_r = p3.perp();
        double p3_z = p3.z();
        float p3_phi = hits.phi(KDdata);

        Range rangeRPhi = predictionRPhi(curvature, p3_r);
        correction.correctRPhiRange(rangeRPhi);

        float ir = 1.f / p3_r;
        // limit error to 90 degree
        constexpr float maxPhiErr = 0.5 * M_PI;
        float phiErr = nSigmaPhi * hits.drphi[KDdata] * ir;
        phiErr = std::min(maxPhiErr, phiErr);
        if (!checkPhiInRange(p3_phi, rangeRPhi.first * ir - phiErr, rangeRPhi.second * ir + phiErr, maxPhiErr))
          continue;

        Basic2DVector<double> thc(p3.x(), p3.y());

        auto curv_ = predictionRPhi.curvature(thc);
        double p2_r = point2.r();
        double p2_z = point2.z();  // they will be modified!

        l2rzFixup(predictionRPhi, curv_, *doublets.hit(ip, HitDoublets::outer), p2_r, p2_z, tTopo);
        l3rzFixup(predictionRPhi, curv_, *hits.theHits[KDdata].hit(), p3_r, p3_z, tTopo);

        Range rangeRZ;
        if (useBend) {
          HelixRZ updatedHelix(&predictionRPhi, gp1.z(), p2_z, curv_);
          rangeRZ = predRZ.helix1(barrelLayer ? p3_r : p3_z, updatedHelix);
        } else {
          float tIP = predictionRPhi.transverseIP(thc);
          PixelRecoPointRZ updatedPoint2(p2_r, p2_z);
          PixelRecoLineRZ updatedLine(line.origin(), point2, tIP);
          rangeRZ = predRZ.line(barrelLayer ? p3_r : p3_z, line);
        }
        correction.correctRZRange(rangeRZ);

        double err = nSigmaRZ * hits.dv[KDdata];

        rangeRZ.first -= err, rangeRZ.second += err;

        if (!rangeRZ.inside(barrelLayer ? p3_z : p3_r))
          continue;

        if (theMaxElement != 0 && result.size() >= theMaxElement) {
          result.clear();
          if (tripletLastLayerIndex)
            tripletLastLayerIndex->clear();
          edm::LogError("TooManyTriplets") << " number of triples exceed maximum. no triplets produced.";
          return;
        }
        result.emplace_back(
            doublets.hit(ip, HitDoublets::inner), doublets.hit(ip, HitDoublets::outer), hits.theHits[KDdata].hit());
        // to bookkeep the triplets and 3rd layers in triplet EDProducer
        if (tripletLastLayerIndex)
          tripletLastLayerIndex->push_back(il);
      }
    }
  }
  // std::cout << "found triplets " << result.size() << std::endl;
}
