#include <functional>

#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/PixelSeeding/interface/CAHitQuadrupletGenerator.h"
#include "RecoTracker/PixelSeeding/interface/ThirdHitPredictionFromCircle.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "RecoTracker/PixelSeeding/interface/CAGraph.h"
#include "CellularAutomaton.h"

namespace {
  template <typename T>
  T sqr(T x) {
    return x * x;
  }
}  // namespace

using namespace std;

constexpr unsigned int CAHitQuadrupletGenerator::minLayers;

CAHitQuadrupletGenerator::CAHitQuadrupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : theFieldToken(iC.esConsumes()),
      extraHitRPhitolerance(cfg.getParameter<double>(
          "extraHitRPhitolerance")),  //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
      maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
      fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
      fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
      useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
      caThetaCut(cfg.getParameter<double>("CAThetaCut"),
                 cfg.getParameter<std::vector<edm::ParameterSet>>("CAThetaCut_byTriplets")),
      caPhiCut(cfg.getParameter<double>("CAPhiCut"),
               cfg.getParameter<std::vector<edm::ParameterSet>>("CAPhiCut_byTriplets")),
      caHardPtCut(cfg.getParameter<double>("CAHardPtCut")) {
  edm::ParameterSet comparitorPSet = cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  if (comparitorName != "none") {
    theComparitor = SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC);
  }
}

void CAHitQuadrupletGenerator::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<double>("extraHitRPhitolerance", 0.1);
  desc.add<bool>("fitFastCircle", false);
  desc.add<bool>("fitFastCircleChi2Cut", false);
  desc.add<bool>("useBendingCorrection", false);
  desc.add<double>("CAThetaCut", 0.00125);
  desc.add<double>("CAPhiCut", 10);

  edm::ParameterSetDescription validatorCACut;
  validatorCACut.add<string>("seedingLayers", "BPix1+BPix2+BPix3");
  validatorCACut.add<double>("cut", 0.00125);
  std::vector<edm::ParameterSet> defaultCACutVector;
  edm::ParameterSet defaultCACut;
  defaultCACut.addParameter<string>("seedingLayers", "");
  defaultCACut.addParameter<double>("cut", -1.);
  defaultCACutVector.push_back(defaultCACut);
  desc.addVPSet("CAThetaCut_byTriplets", validatorCACut, defaultCACutVector);
  desc.addVPSet("CAPhiCut_byTriplets", validatorCACut, defaultCACutVector);

  desc.add<double>("CAHardPtCut", 0);
  desc.addOptional<bool>("CAOnlyOneLastHitPerLayerFilter")
      ->setComment(
          "Deprecated and has no effect. To be fully removed later when the parameter is no longer used in HLT "
          "configurations.");
  edm::ParameterSetDescription descMaxChi2;
  descMaxChi2.add<double>("pt1", 0.2);
  descMaxChi2.add<double>("pt2", 1.5);
  descMaxChi2.add<double>("value1", 500);
  descMaxChi2.add<double>("value2", 50);
  descMaxChi2.add<bool>("enabled", true);
  desc.add<edm::ParameterSetDescription>("maxChi2", descMaxChi2);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything();  // until we have moved SeedComparitor too to EDProducers
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
}

void CAHitQuadrupletGenerator::initEvent(const edm::Event& ev, const edm::EventSetup& es) {
  if (theComparitor)
    theComparitor->init(ev, es);
  theField = &es.getData(theFieldToken);
}
namespace {
  void createGraphStructure(const SeedingLayerSetsHits& layers, CAGraph& g) {
    for (unsigned int i = 0; i < layers.size(); i++) {
      for (unsigned int j = 0; j < 4; ++j) {
        auto vertexIndex = 0;
        auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(), layers[i][j].name());
        if (foundVertex == g.theLayers.end()) {
          g.theLayers.emplace_back(layers[i][j].name(), layers[i][j].detLayer()->seqNum(), layers[i][j].hits().size());
          vertexIndex = g.theLayers.size() - 1;
        } else {
          vertexIndex = foundVertex - g.theLayers.begin();
        }
        if (j == 0) {
          if (std::find(g.theRootLayers.begin(), g.theRootLayers.end(), vertexIndex) == g.theRootLayers.end()) {
            g.theRootLayers.emplace_back(vertexIndex);
          }
        }
      }
    }
  }
  void clearGraphStructure(const SeedingLayerSetsHits& layers, CAGraph& g) {
    g.theLayerPairs.clear();
    for (unsigned int i = 0; i < g.theLayers.size(); i++) {
      g.theLayers[i].theInnerLayers.clear();
      g.theLayers[i].theInnerLayerPairs.clear();
      g.theLayers[i].theOuterLayers.clear();
      g.theLayers[i].theOuterLayerPairs.clear();
      for (auto& v : g.theLayers[i].isOuterHitOfCell)
        v.clear();
    }
  }
  void fillGraph(const SeedingLayerSetsHits& layers,
                 const IntermediateHitDoublets::RegionLayerSets& regionLayerPairs,
                 CAGraph& g,
                 std::vector<const HitDoublets*>& hitDoublets) {
    for (unsigned int i = 0; i < layers.size(); i++) {
      for (unsigned int j = 0; j < 4; ++j) {
        auto vertexIndex = 0;
        auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(), layers[i][j].name());
        if (foundVertex == g.theLayers.end()) {
          vertexIndex = g.theLayers.size() - 1;
        } else {
          vertexIndex = foundVertex - g.theLayers.begin();
        }

        if (j > 0) {
          auto innerVertex = std::find(g.theLayers.begin(), g.theLayers.end(), layers[i][j - 1].name());

          CALayerPair tmpInnerLayerPair(innerVertex - g.theLayers.begin(), vertexIndex);

          if (std::find(g.theLayerPairs.begin(), g.theLayerPairs.end(), tmpInnerLayerPair) == g.theLayerPairs.end()) {
            auto found = std::find_if(regionLayerPairs.begin(),
                                      regionLayerPairs.end(),
                                      [&](const IntermediateHitDoublets::LayerPairHitDoublets& pair) {
                                        return pair.innerLayerIndex() == layers[i][j - 1].index() &&
                                               pair.outerLayerIndex() == layers[i][j].index();
                                      });
            if (found != regionLayerPairs.end()) {
              hitDoublets.emplace_back(&(found->doublets()));
              g.theLayerPairs.push_back(tmpInnerLayerPair);
              g.theLayers[vertexIndex].theInnerLayers.push_back(innerVertex - g.theLayers.begin());
              innerVertex->theOuterLayers.push_back(vertexIndex);
              g.theLayers[vertexIndex].theInnerLayerPairs.push_back(g.theLayerPairs.size() - 1);
              innerVertex->theOuterLayerPairs.push_back(g.theLayerPairs.size() - 1);
            }
          }
        }
      }
    }
  }
}  // namespace

void CAHitQuadrupletGenerator::hitNtuplets(const IntermediateHitDoublets& regionDoublets,
                                           std::vector<OrderedHitSeeds>& result,
                                           const SeedingLayerSetsHits& layers) {
  CAGraph g;

  std::vector<const HitDoublets*> hitDoublets;

  const int numberOfHitsInNtuplet = 4;
  std::vector<CACell::CAntuplet> foundQuadruplets;

  int index = 0;
  for (const auto& regionLayerPairs : regionDoublets) {
    const TrackingRegion& region = regionLayerPairs.region();
    hitDoublets.clear();
    foundQuadruplets.clear();
    if (index == 0) {
      createGraphStructure(layers, g);
      caThetaCut.setCutValuesByLayerIds(g);
      caPhiCut.setCutValuesByLayerIds(g);
    } else {
      clearGraphStructure(layers, g);
    }

    fillGraph(layers, regionLayerPairs, g, hitDoublets);

    CellularAutomaton ca(g);

    ca.createAndConnectCells(hitDoublets, region, caThetaCut, caPhiCut, caHardPtCut);

    ca.evolve(numberOfHitsInNtuplet);

    ca.findNtuplets(foundQuadruplets, numberOfHitsInNtuplet);

    auto& allCells = ca.getAllCells();

    const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(*theField);

    // re-used thoughout
    std::array<float, 4> bc_r;
    std::array<float, 4> bc_z;
    std::array<float, 4> bc_errZ2;
    std::array<GlobalPoint, 4> gps;
    std::array<GlobalError, 4> ges;
    std::array<bool, 4> barrels;

    unsigned int numberOfFoundQuadruplets = foundQuadruplets.size();

    // Loop over quadruplets
    for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId) {
      auto isBarrel = [](const unsigned id) -> bool { return id == PixelSubdetector::PixelBarrel; };
      for (unsigned int i = 0; i < 3; ++i) {
        auto const& ahit = allCells[foundQuadruplets[quadId][i]].getInnerHit();
        gps[i] = ahit->globalPosition();
        ges[i] = ahit->globalPositionError();
        barrels[i] = isBarrel(ahit->geographicalId().subdetId());
      }

      auto const& ahit = allCells[foundQuadruplets[quadId][2]].getOuterHit();
      gps[3] = ahit->globalPosition();
      ges[3] = ahit->globalPositionError();
      barrels[3] = isBarrel(ahit->geographicalId().subdetId());
      // TODO:
      // - if we decide to always do the circle fit for 4 hits, we don't
      //   need ThirdHitPredictionFromCircle for the curvature; then we
      //   could remove extraHitRPhitolerance configuration parameter
      ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2], extraHitRPhitolerance);
      const float curvature = predictionRPhi.curvature(ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
      const float abscurv = std::abs(curvature);
      const float thisMaxChi2 = maxChi2Eval.value(abscurv);
      if (theComparitor) {
        SeedingHitSet tmpTriplet(allCells[foundQuadruplets[quadId][0]].getInnerHit(),
                                 allCells[foundQuadruplets[quadId][2]].getInnerHit(),
                                 allCells[foundQuadruplets[quadId][2]].getOuterHit());

        if (!theComparitor->compatible(tmpTriplet)) {
          continue;
        }
      }

      float chi2 = std::numeric_limits<float>::quiet_NaN();
      // TODO: Do we have any use case to not use bending correction?
      if (useBendingCorrection) {
        // Following PixelFitterByConformalMappingAndLine
        const float simpleCot = (gps.back().z() - gps.front().z()) / (gps.back().perp() - gps.front().perp());
        const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, *theField);
        for (int i = 0; i < 4; ++i) {
          const GlobalPoint& point = gps[i];
          const GlobalError& error = ges[i];
          bc_r[i] = sqrt(sqr(point.x() - region.origin().x()) + sqr(point.y() - region.origin().y()));
          bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt, *theField)(bc_r[i]);
          bc_z[i] = point.z() - region.origin().z();
          bc_errZ2[i] = (barrels[i]) ? error.czz() : error.rerr(point) * sqr(simpleCot);
        }
        RZLine rzLine(bc_r, bc_z, bc_errZ2, RZLine::ErrZ2_tag());
        chi2 = rzLine.chi2();
      } else {
        RZLine rzLine(gps, ges, barrels);
        chi2 = rzLine.chi2();
      }
      if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2) {
        continue;
      }
      // TODO: Do we have any use case to not use circle fit? Maybe
      // HLT where low-pT inefficiency is not a problem?
      if (fitFastCircle) {
        FastCircleFit c(gps, ges);
        chi2 += c.chi2();
        if (edm::isNotFinite(chi2))
          continue;
        if (fitFastCircleChi2Cut && chi2 > thisMaxChi2)
          continue;
      }
      result[index].emplace_back(allCells[foundQuadruplets[quadId][0]].getInnerHit(),
                                 allCells[foundQuadruplets[quadId][1]].getInnerHit(),
                                 allCells[foundQuadruplets[quadId][2]].getInnerHit(),
                                 allCells[foundQuadruplets[quadId][2]].getOuterHit());
    }
    index++;
  }
}
