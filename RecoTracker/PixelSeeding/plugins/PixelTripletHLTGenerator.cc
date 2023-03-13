#include "RecoTracker/PixelSeeding/plugins/PixelTripletHLTGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "ThirdHitPredictionFromInvParabola.h"
#include "RecoTracker/PixelSeeding/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "DataFormats/Math/interface/normalizedPhi.h"

#include <cstdio>
#include <iostream>

using pixelrecoutilities::LongitudinalBendingCorrection;
using Range = PixelRecoRange<float>;

using namespace std;

PixelTripletHLTGenerator::PixelTripletHLTGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : HitTripletGeneratorFromPairAndLayers(cfg),
      fieldToken_(iC.esConsumes()),
      useFixedPreFiltering(cfg.getParameter<bool>("useFixedPreFiltering")),
      extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
      extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
      useMScat(cfg.getParameter<bool>("useMultScattering")),
      useBend(cfg.getParameter<bool>("useBending")),
      dphi(useFixedPreFiltering ? cfg.getParameter<double>("phiPreFiltering") : 0) {
  edm::ParameterSet comparitorPSet = cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  if (comparitorName != "none") {
    theComparitor = SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC);
  }
  if (useMScat) {
    msmakerToken_ = iC.esConsumes();
  }
}

PixelTripletHLTGenerator::~PixelTripletHLTGenerator() {}

void PixelTripletHLTGenerator::fillDescriptions(edm::ParameterSetDescription& desc) {
  HitTripletGeneratorFromPairAndLayers::fillDescriptions(desc);
  desc.add<double>("extraHitRPhitolerance", 0.032);
  desc.add<double>("extraHitRZtolerance", 0.037);
  desc.add<bool>("useMultScattering", true);
  desc.add<bool>("useBending", true);
  desc.add<bool>("useFixedPreFiltering", false);
  desc.add<double>("phiPreFiltering", 0.3);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything();  // until we have moved SeedComparitor too to EDProducers
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
}

void PixelTripletHLTGenerator::hitTriplets(const TrackingRegion& region,
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

void PixelTripletHLTGenerator::hitTriplets(const TrackingRegion& region,
                                           OrderedHitTriplets& result,
                                           const edm::Event& ev,
                                           const edm::EventSetup& es,
                                           const HitDoublets& doublets,
                                           const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
                                           std::vector<int>* tripletLastLayerIndex,
                                           LayerCacheType& layerCache) {
  if (theComparitor)
    theComparitor->init(ev, es);

  int size = thirdLayers.size();
  const RecHitsSortedInPhi* thirdHitMap[size];
  vector<const DetLayer*> thirdLayerDetLayer(size, nullptr);
  for (int il = 0; il < size; ++il) {
    thirdHitMap[il] = &layerCache(thirdLayers[il], region);
    thirdLayerDetLayer[il] = thirdLayers[il].detLayer();
  }
  hitTriplets(region, result, es, doublets, thirdHitMap, thirdLayerDetLayer, size, tripletLastLayerIndex);
}

void PixelTripletHLTGenerator::hitTriplets(const TrackingRegion& region,
                                           OrderedHitTriplets& result,
                                           const edm::EventSetup& es,
                                           const HitDoublets& doublets,
                                           const RecHitsSortedInPhi** thirdHitMap,
                                           const std::vector<const DetLayer*>& thirdLayerDetLayer,
                                           const int nThirdLayers) {
  hitTriplets(region, result, es, doublets, thirdHitMap, thirdLayerDetLayer, nThirdLayers, nullptr);
}

void PixelTripletHLTGenerator::hitTriplets(const TrackingRegion& region,
                                           OrderedHitTriplets& result,
                                           const edm::EventSetup& es,
                                           const HitDoublets& doublets,
                                           const RecHitsSortedInPhi** thirdHitMap,
                                           const std::vector<const DetLayer*>& thirdLayerDetLayer,
                                           const int nThirdLayers,
                                           std::vector<int>* tripletLastLayerIndex) {
  auto outSeq = doublets.detLayer(HitDoublets::outer)->seqNum();

  float regOffset = region.origin().perp();  //try to take account of non-centrality (?)

  declareDynArray(ThirdHitRZPrediction<PixelRecoLineRZ>, nThirdLayers, preds);
  declareDynArray(ThirdHitCorrection, nThirdLayers, corrections);

  typedef RecHitsSortedInPhi::Hit Hit;

  using NodeInfo = KDTreeNodeInfo<unsigned int, 2>;
  std::vector<NodeInfo> layerTree;       // re-used throughout
  std::vector<unsigned int> foundNodes;  // re-used thoughout
  foundNodes.reserve(100);

  declareDynArray(KDTreeLinkerAlgo<unsigned int>, nThirdLayers, hitTree);
  float rzError[nThirdLayers];  //save maximum errors

  const float maxDelphi = region.ptMin() < 0.3f ? float(M_PI) / 4.f : float(M_PI) / 8.f;  // FIXME move to config??
  const float maxphi = M_PI + maxDelphi, minphi = -maxphi;  // increase to cater for any range
  const float safePhi = M_PI - maxDelphi;                   // sideband

  const auto& field = es.getData(fieldToken_);
  const MultipleScatteringParametrisationMaker* msmaker = nullptr;
  if (useMScat) {
    msmaker = &es.getData(msmakerToken_);
  }

  // fill the prediction vector
  for (int il = 0; il < nThirdLayers; ++il) {
    auto const& hits = *thirdHitMap[il];
    ThirdHitRZPrediction<PixelRecoLineRZ>& pred = preds[il];
    pred.initLayer(thirdLayerDetLayer[il]);
    pred.initTolerance(extraHitRZtolerance);

    corrections[il].init(region.ptMin(),
                         *doublets.detLayer(HitDoublets::inner),
                         *doublets.detLayer(HitDoublets::outer),
                         *thirdLayerDetLayer[il],
                         useMScat,
                         msmaker,
                         useBend,
                         &field);

    layerTree.clear();
    float minv = 999999.0f, maxv = -minv;  // Initialise to extreme values in case no hits
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
    // std::cout << "layer " << thirdLayerDetLayer[il]->seqNum() << " " << layerTree.size() << std::endl;
  }

  float imppar = region.originRBound();
  float imppartmp = region.originRBound() + region.origin().perp();
  float curv = PixelRecoUtilities::curvature(1.f / region.ptMin(), field);

  for (std::size_t ip = 0; ip != doublets.size(); ip++) {
    auto xi = doublets.x(ip, HitDoublets::inner);
    auto yi = doublets.y(ip, HitDoublets::inner);
    auto zi = doublets.z(ip, HitDoublets::inner);
    auto rvi = doublets.rv(ip, HitDoublets::inner);
    auto xo = doublets.x(ip, HitDoublets::outer);
    auto yo = doublets.y(ip, HitDoublets::outer);
    auto zo = doublets.z(ip, HitDoublets::outer);
    auto rvo = doublets.rv(ip, HitDoublets::outer);

    auto toPos = std::signbit(zo - zi);

    PixelRecoPointRZ point1(rvi, zi);
    PixelRecoPointRZ point2(rvo, zo);
    PixelRecoLineRZ line(point1, point2);
    ThirdHitPredictionFromInvParabola predictionRPhi(xi - region.origin().x(),
                                                     yi - region.origin().y(),
                                                     xo - region.origin().x(),
                                                     yo - region.origin().y(),
                                                     imppar,
                                                     curv,
                                                     extraHitRPhitolerance);

    ThirdHitPredictionFromInvParabola predictionRPhitmp(xi, yi, xo, yo, imppartmp, curv, extraHitRPhitolerance);

    // printf("++Constr %f %f %f %f %f %f %f\n",xi,yi,xo,yo,imppartmp,curv,extraHitRPhitolerance);

    // std::cout << ip << ": " << point1.r() << ","<< point1.z() << " "
    //                        << point2.r() << ","<< point2.z() <<std::endl;

    for (int il = 0; il != nThirdLayers; ++il) {
      const DetLayer* layer = thirdLayerDetLayer[il];
      auto barrelLayer = layer->isBarrel();

      if ((!barrelLayer) & (toPos != std::signbit(layer->position().z())))
        continue;

      if (hitTree[il].empty())
        continue;  // Don't bother if no hits

      auto const& hits = *thirdHitMap[il];

      auto& correction = corrections[il];

      correction.init(line, point2, outSeq);

      auto& predictionRZ = preds[il];

      predictionRZ.initPropagator(&line);
      Range rzRange = predictionRZ();
      correction.correctRZRange(rzRange);

      Range phiRange;
      if (useFixedPreFiltering) {
        float phi0 = doublets.phi(ip, HitDoublets::outer);
        phiRange = Range(phi0 - dphi, phi0 + dphi);
      } else {
        Range radius;
        if (barrelLayer) {
          radius = predictionRZ.detRange();
        } else {
          radius =
              Range(max(rzRange.min(), predictionRZ.detSize().min()), min(rzRange.max(), predictionRZ.detSize().max()));
        }
        if (radius.empty())
          continue;

        // std::cout << "++R " << radius.min() << " " << radius.max()  << std::endl;

        auto rPhi1 = predictionRPhitmp(radius.max());
        bool ok1 = !rPhi1.empty();
        if (ok1) {
          correction.correctRPhiRange(rPhi1);
          rPhi1.first /= radius.max();
          rPhi1.second /= radius.max();
        }
        auto rPhi2 = predictionRPhitmp(radius.min());
        bool ok2 = !rPhi2.empty();
        if (ok2) {
          correction.correctRPhiRange(rPhi2);
          rPhi2.first /= radius.min();
          rPhi2.second /= radius.min();
        }

        if (ok1) {
          rPhi1.first = normalizedPhi(rPhi1.first);
          rPhi1.second = proxim(rPhi1.second, rPhi1.first);
          if (ok2) {
            rPhi2.first = proxim(rPhi2.first, rPhi1.first);
            rPhi2.second = proxim(rPhi2.second, rPhi1.first);
            phiRange = rPhi1.sum(rPhi2);
          } else
            phiRange = rPhi1;
        } else if (ok2) {
          rPhi2.first = normalizedPhi(rPhi2.first);
          rPhi2.second = proxim(rPhi2.second, rPhi2.first);
          phiRange = rPhi2;
        } else
          continue;
      }

      constexpr float nSigmaRZ = 3.46410161514f;  // std::sqrt(12.f); // ...and continue as before
      constexpr float nSigmaPhi = 3.f;

      foundNodes.clear();  // Now recover hits in bounding box...
      float prmin = phiRange.min(), prmax = phiRange.max();

      if (prmax - prmin > maxDelphi) {
        auto prm = phiRange.mean();
        prmin = prm - 0.5f * maxDelphi;
        prmax = prm + 0.5f * maxDelphi;
      }

      if (barrelLayer) {
        Range regMax = predictionRZ.detRange();
        Range regMin = predictionRZ(regMax.min() - regOffset);
        regMax = predictionRZ(regMax.max() + regOffset);
        correction.correctRZRange(regMin);
        correction.correctRZRange(regMax);
        if (regMax.min() < regMin.min()) {
          swap(regMax, regMin);
        }
        KDTreeBox phiZ(prmin, prmax, regMin.min() - nSigmaRZ * rzError[il], regMax.max() + nSigmaRZ * rzError[il]);
        hitTree[il].search(phiZ, foundNodes);
      } else {
        KDTreeBox phiZ(prmin,
                       prmax,
                       rzRange.min() - regOffset - nSigmaRZ * rzError[il],
                       rzRange.max() + regOffset + nSigmaRZ * rzError[il]);
        hitTree[il].search(phiZ, foundNodes);
      }

      // std::cout << ip << ": " << thirdLayerDetLayer[il]->seqNum() << " " << foundNodes.size() << " " << prmin << " " << prmax << std::endl;

      // int kk=0;
      for (auto KDdata : foundNodes) {
        if (theMaxElement != 0 && result.size() >= theMaxElement) {
          result.clear();
          if (tripletLastLayerIndex)
            tripletLastLayerIndex->clear();
          edm::LogError("TooManyTriplets") << " number of triples exceeds maximum. no triplets produced.";
          return;
        }

        float p3_u = hits.u[KDdata];
        float p3_v = hits.v[KDdata];
        float p3_phi = hits.lphi[KDdata];

        //if ((kk++)%100==0)
        //std::cout << kk << ": " << p3_u << " " << p3_v << " " << p3_phi << std::endl;

        Range allowed = predictionRZ(p3_u);
        correction.correctRZRange(allowed);
        float vErr = nSigmaRZ * hits.dv[KDdata];
        Range hitRange(p3_v - vErr, p3_v + vErr);
        Range crossingRange = allowed.intersection(hitRange);
        if (crossingRange.empty())
          continue;

        float ir = 1.f / hits.rv(KDdata);
        // limit error to 90 degree
        constexpr float maxPhiErr = 0.5 * M_PI;
        float phiErr = nSigmaPhi * hits.drphi[KDdata] * ir;
        phiErr = std::min(maxPhiErr, phiErr);
        bool nook = true;
        for (int icharge = -1; icharge <= 1; icharge += 2) {
          Range rangeRPhi = predictionRPhi(hits.rv(KDdata), icharge);
          if (rangeRPhi.first > rangeRPhi.second)
            continue;  // range is empty
          correction.correctRPhiRange(rangeRPhi);
          if (checkPhiInRange(p3_phi, rangeRPhi.first * ir - phiErr, rangeRPhi.second * ir + phiErr, maxPhiErr)) {
            // insert here check with comparitor
            OrderedHitTriplet hittriplet(
                doublets.hit(ip, HitDoublets::inner), doublets.hit(ip, HitDoublets::outer), hits.theHits[KDdata].hit());
            if (!theComparitor || theComparitor->compatible(hittriplet)) {
              result.push_back(hittriplet);
              // to bookkeep the triplets and 3rd layers in triplet EDProducer
              if (tripletLastLayerIndex)
                tripletLastLayerIndex->push_back(il);
            } else {
              LogDebug("RejectedTriplet") << "rejected triplet from comparitor ";
            }
            nook = false;
            break;
          }
        }
        if (nook)
          LogDebug("RejectedTriplet") << "rejected triplet from second phicheck " << p3_phi;
      }
    }
  }
  // std::cout << "triplets " << result.size() << std::endl;
}
