// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "CAHitNtupletGenerator.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAPixelDoublets.h"
#include "CAPixelDoubletsAlgos.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {

    using namespace caHitNtupletGenerator;
    using namespace caPixelDoublets;
    using namespace caStructures;
    using namespace pixelTopology;
    using namespace pixelTrack;

    template <typename T>
    T sqr(T x) {
      return x * x;
    }

    // Common Params
    template <typename TrackerTraits>
    void fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
      desc.add<double>("cellZ0Cut", TrackerTraits::cellZ0Cut)->setComment("Z0 cut for cells");

      //// Pixel Cluster Cuts (@cell level)
      desc.add<double>("dzdrFact", TrackerTraits::dzdrFact);
      desc.add<int>("minYsizeB1", TrackerTraits::minYsizeB1)
          ->setComment("Cut on inner hit cluster size (in global z / local y) for barrel-forward cells. Barrel 1 cut.");
      desc.add<int>("minYsizeB2", TrackerTraits::minYsizeB2)
          ->setComment(
              "Cut on inner hit cluster size (in global z / local y) for barrel-forward cells. Anything but Barrel 1 "
              "cut.");
      desc.add<int>("maxDYsize12", TrackerTraits::maxDYsize12)
          ->setComment(
              "Cut on cluster size differences (in global z / local y) for barrel-forward cells. Barrel 1-2 cells.");
      desc.add<int>("maxDYsize", TrackerTraits::maxDYsize)
          ->setComment(
              "Cut on cluster size differences (in global z / local y) for barrel-forward cells. Other barrel cells.");
      desc.add<int>("maxDYPred", TrackerTraits::maxDYPred)
          ->setComment(
              "Maximum difference between actual and expected cluster size of inner RecHit. Barrel-forward cells.");

      edm::ParameterSetDescription geometryParams;
      // layers params
      geometryParams
          .add<std::vector<double>>(
              "caDCACuts",
              std::vector<double>(TrackerTraits::dcaCuts, TrackerTraits::dcaCuts + TrackerTraits::numberOfLayers))
          ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
      geometryParams
          .add<std::vector<double>>(
              "caThetaCuts",
              std::vector<double>(TrackerTraits::thetaCuts, TrackerTraits::thetaCuts + TrackerTraits::numberOfLayers))
          ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
      geometryParams
          .add<std::vector<unsigned int>>(
              "startingPairs",
              std::vector<unsigned int>(TrackerTraits::startingPairs,
                                        TrackerTraits::startingPairs + TrackerTraits::nStartingPairs))
          ->setComment("The list of the ids of pairs from which the CA ntuplets building may start.");
      // cells params
      geometryParams
          .add<std::vector<unsigned int>>(
              "pairGraph",
              std::vector<unsigned int>(TrackerTraits::layerPairs,
                                        TrackerTraits::layerPairs + (TrackerTraits::nPairsForQuadruplets * 2)))
          ->setComment("CA graph (layer pairs used for building doublets/cells)");
      geometryParams
          .add<std::vector<int>>(
              "phiCuts",
              std::vector<int>(TrackerTraits::phicuts, TrackerTraits::phicuts + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in dphi for cells");
      geometryParams
          .add<std::vector<double>>(
              "ptCuts",
              std::vector<double>(TrackerTraits::ptCuts, TrackerTraits::ptCuts + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in pt for cells");
      geometryParams
          .add<std::vector<double>>("minInner",
                                    std::vector<double>(TrackerTraits::minInner,
                                                        TrackerTraits::minInner + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (min value)");
      geometryParams
          .add<std::vector<double>>("maxInner",
                                    std::vector<double>(TrackerTraits::maxInner,
                                                        TrackerTraits::maxInner + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (max value)");
      geometryParams
          .add<std::vector<double>>("minOuter",
                                    std::vector<double>(TrackerTraits::minOuter,
                                                        TrackerTraits::minOuter + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (min value)");
      geometryParams
          .add<std::vector<double>>("maxOuter",
                                    std::vector<double>(TrackerTraits::maxOuter,
                                                        TrackerTraits::maxOuter + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (max value)");
      geometryParams
          .add<std::vector<double>>(
              "maxDR",
              std::vector<double>(TrackerTraits::maxDR, TrackerTraits::maxDR + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in max dr for cells");
      geometryParams
          .add<std::vector<double>>(
              "minDZ",
              std::vector<double>(TrackerTraits::minDZ, TrackerTraits::minDZ + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in minimum dz between hits for cells");
      geometryParams
          .add<std::vector<double>>(
              "maxDZ",
              std::vector<double>(TrackerTraits::maxDZ, TrackerTraits::maxDZ + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in maximum dz between hits for cells");

      desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
          ->setComment("Layer-dependent cuts and settings of the CA");

      // Container sizes
      //
      // maxNumberOfDoublets and maxNumberOfTuples may be defined at runtime depending on the number of hits.
      // This is done via a FormulaEvaluator expecting 'x' as nHits.
      // e.g. : maxNumberOfDoublets = cms.string( '0.00022*pow(x,2)  + 0.53*x + 10000' )
      // will compute maxNumberOfDoublets for each event as
      //
      //  	maxNumberOfDoublets = 2.2e-4 * nHits^2 + 0.53 * nHits + 10000
      //
      // this may also be simply a constant (as for the default parameters)
      //
      // 	 maxNumberOfDoublets = cms.string(str(512*1024))
      //

      desc.add<std::string>("maxNumberOfDoublets", std::to_string(TrackerTraits::maxNumberOfDoublets))
          ->setComment(
              "Max nummber of doublets (cells) as a string. The string will be parsed to a TFormula, depending on "
              "nHits (labeled 'x'), \n and evaluated for each event. May also be a constant.");
      desc.add<std::string>("maxNumberOfTuples", std::to_string(TrackerTraits::maxNumberOfTuples))
          ->setComment("Max nummber of tuples as a string. Same behavior as maxNumberOfDoublets.");
      desc.add<double>("avgHitsPerTrack", double(TrackerTraits::avgHitsPerTrack))
          ->setComment("Number of hits per track. Average per track.");
      desc.add<double>("avgCellsPerHit", TrackerTraits::avgCellsPerHit)
          ->setComment("Number of cells for which an hit is the outer hit. Average per hit.");
      desc.add<double>("avgCellsPerCell", TrackerTraits::avgCellsPerCell)
          ->setComment("Number of cells connected to another cell. Average per cell.");
      desc.add<double>("avgTracksPerCell", TrackerTraits::avgTracksPerCell)
          ->setComment("Number of tracks to which a cell belongs. Average per cell.");

      // nTuplet Cuts and Params
      desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
      //// p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
      //// 87 cm/GeV = 1/(3.8T * 0.3)
      //// take less than radius given by the hardPtCut and reject everything below
      desc.add<double>("hardCurvCut", TrackerTraits::hardCurvCut)
          ->setComment("Cut on minimum curvature, used in DCA ntuplet selection");

      desc.add<bool>("earlyFishbone", true);
      desc.add<bool>("lateFishbone", false);
      desc.add<bool>("fillStatistics", false);
      desc.add<unsigned int>("minHitsPerNtuplet", 4);
      desc.add<unsigned int>("minHitsForSharingCut", 10)
          ->setComment("Maximum number of hits in a tuple to clean also if the shared hit is on bpx1");

      desc.add<bool>("fitNas4", false)->setComment("fit only 4 hits out of N");

      desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
      desc.add<bool>("doSharedHitCut", true)->setComment("Sharing hit nTuples cleaning");
      desc.add<bool>("dupPassThrough", false)->setComment("Do not reject duplicate");
      desc.add<bool>("useSimpleTripletCleaner", true)->setComment("use alternate implementation");
    }

    AlgoParams makeCommonParams(edm::ParameterSet const& cfg) {
      return AlgoParams({

          // Container sizes
          (float)cfg.getParameter<double>("avgHitsPerTrack"),
          (float)cfg.getParameter<double>("avgCellsPerHit"),
          (float)cfg.getParameter<double>("avgCellsPerCell"),
          (float)cfg.getParameter<double>("avgTracksPerCell"),

          // Algo params
          (uint16_t)cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
          (uint16_t)cfg.getParameter<unsigned int>("minHitsForSharingCut"),
          (float)cfg.getParameter<double>("ptmin"),
          (float)cfg.getParameter<double>("hardCurvCut"),
          (float)cfg.getParameter<double>("cellZ0Cut"),

          // Pixel Cluster Cut Params
          (float)cfg.getParameter<double>("dzdrFact"),
          (int16_t)cfg.getParameter<int>("minYsizeB1"),
          (int16_t)cfg.getParameter<int>("minYsizeB2"),
          (int16_t)cfg.getParameter<int>("maxDYsize12"),
          (int16_t)cfg.getParameter<int>("maxDYsize"),
          (int16_t)cfg.getParameter<int>("maxDYPred"),

          // Flags
          cfg.getParameter<bool>("useRiemannFit"),
          cfg.getParameter<bool>("fitNas4"),
          cfg.getParameter<bool>("earlyFishbone"),
          cfg.getParameter<bool>("lateFishbone"),
          cfg.getParameter<bool>("fillStatistics"),
          cfg.getParameter<bool>("doSharedHitCut"),
          cfg.getParameter<bool>("dupPassThrough"),
          cfg.getParameter<bool>("useSimpleTripletCleaner")});
    }

    //This is needed to have the partial specialization for isPhase1Topology/isPhase2Topology
    template <typename TrackerTraits, typename Enable = void>
    struct TopologyCuts {};

    template <typename TrackerTraits>
    struct TopologyCuts<TrackerTraits, isPhase1Topology<TrackerTraits>> {
      static constexpr ::pixelTrack::QualityCutsT<TrackerTraits> makeQualityCuts(edm::ParameterSet const& pset) {
        auto coeff = pset.getParameter<std::array<double, 2>>("chi2Coeff");
        auto ptMax = pset.getParameter<double>("chi2MaxPt");

        coeff[1] = (coeff[1] - coeff[0]) / log2(ptMax);
        return ::pixelTrack::QualityCutsT<TrackerTraits>{// polynomial coefficients for the pT-dependent chi2 cut
                                                         {(float)coeff[0], (float)coeff[1], 0.f, 0.f},
                                                         // max pT used to determine the chi2 cut
                                                         (float)ptMax,
                                                         // chi2 scale factor: 8 for broken line fit, ?? for Riemann fit
                                                         (float)pset.getParameter<double>("chi2Scale"),
                                                         // regional cuts for triplets
                                                         {(float)pset.getParameter<double>("tripletMaxTip"),
                                                          (float)pset.getParameter<double>("tripletMinPt"),
                                                          (float)pset.getParameter<double>("tripletMaxZip")},
                                                         // regional cuts for quadruplets
                                                         {(float)pset.getParameter<double>("quadrupletMaxTip"),
                                                          (float)pset.getParameter<double>("quadrupletMinPt"),
                                                          (float)pset.getParameter<double>("quadrupletMaxZip")}};
      }
    };

    template <typename TrackerTraits>
    struct TopologyCuts<TrackerTraits, isPhase2Topology<TrackerTraits>> {
      static constexpr ::pixelTrack::QualityCutsT<TrackerTraits> makeQualityCuts(edm::ParameterSet const& pset) {
        return ::pixelTrack::QualityCutsT<TrackerTraits>{
            static_cast<float>(pset.getParameter<double>("maxChi2")),
            static_cast<float>(pset.getParameter<double>("maxChi2TripletsOrQuadruplets")),
            static_cast<float>(pset.getParameter<double>("maxChi2Quintuplets")),
            static_cast<float>(pset.getParameter<double>("minPt")),
            static_cast<float>(pset.getParameter<double>("maxTip")),
            static_cast<float>(pset.getParameter<double>("maxZip")),
        };
      }
    };

  }  // namespace

  using namespace std;

  template <typename TrackerTraits>
  CAHitNtupletGenerator<TrackerTraits>::CAHitNtupletGenerator(const edm::ParameterSet& cfg)
      : m_params(makeCommonParams(cfg),
                 TopologyCuts<TrackerTraits>::makeQualityCuts(cfg.getParameterSet("trackQualityCuts"))) {
#ifdef DUMP_GPU_TK_TUPLES
    printf("TK: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
           "tid",
           "qual",
           "nh",
           "nl",
           "charge",
           "pt",
           "eta",
           "phi",
           "tip",
           "zip",
           "chi2",
           "h1",
           "h2",
           "h3",
           "h4",
           "h5",
           "hn");
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGenerator<TrackerTraits>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    static_assert(sizeof(TrackerTraits) == 0,
                  "Note: this fillPSetDescription is a dummy one. Please specialise it for the correct version of "
                  "CAHitNtupletGenerator<TrackerTraits>.");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase1>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase1>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut");
    trackQualityCuts.add<std::vector<double>>("chi2Coeff", {0.9, 1.8})->setComment("chi2 at 1GeV and at ptMax above");
    trackQualityCuts.add<double>("chi2Scale", 8.)
        ->setComment(
            "Factor to multiply the pT-dependent chi2 cut (currently: 8 for the broken line fit, ?? for the Riemann "
            "fit)");
    trackQualityCuts.add<double>("tripletMinPt", 0.5)->setComment("Min pT for triplets, in GeV");
    trackQualityCuts.add<double>("tripletMaxTip", 0.3)->setComment("Max |Tip| for triplets, in cm");
    trackQualityCuts.add<double>("tripletMaxZip", 12.)->setComment("Max |Zip| for triplets, in cm");
    trackQualityCuts.add<double>("quadrupletMinPt", 0.3)->setComment("Min pT for quadruplets, in GeV");
    trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
    trackQualityCuts.add<double>("quadrupletMaxZip", 12.)->setComment("Max |Zip| for quadruplets, in cm");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply "
            "\"region cuts\" based on the fit results (pT, Tip, Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::HIonPhase1>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::HIonPhase1>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut");
    trackQualityCuts.add<std::vector<double>>("chi2Coeff", {0.9, 1.8})->setComment("chi2 at 1GeV and at ptMax above");
    trackQualityCuts.add<double>("chi2Scale", 8.)
        ->setComment(
            "Factor to multiply the pT-dependent chi2 cut (currently: 8 for the broken line fit, ?? for the Riemann "
            "fit)");
    trackQualityCuts.add<double>("tripletMinPt", 0.0)->setComment("Min pT for triplets, in GeV");
    trackQualityCuts.add<double>("tripletMaxTip", 0.1)->setComment("Max |Tip| for triplets, in cm");
    trackQualityCuts.add<double>("tripletMaxZip", 6.)->setComment("Max |Zip| for triplets, in cm");
    trackQualityCuts.add<double>("quadrupletMinPt", 0.0)->setComment("Min pT for quadruplets, in GeV");
    trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
    trackQualityCuts.add<double>("quadrupletMaxZip", 6.)->setComment("Max |Zip| for quadruplets, in cm");

    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply "
            "\"region cuts\" based on the fit results (pT, Tip, Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase2>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 5.)->setComment("Max normalized chi2 for tracks with 6 or more hits");
    trackQualityCuts.add<double>("maxChi2TripletsOrQuadruplets", 5.)
        ->setComment("Max normalized chi2 for tracks with 4 or less hits");
    trackQualityCuts.add<double>("maxChi2Quintuplets", 5.)->setComment("Max normalized chi2 for tracks with 5 hits");
    trackQualityCuts.add<double>("minPt", 0.5)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2OT>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase2OT>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 5.)->setComment("Max normalized chi2 for tracks with 6 or more hits");
    trackQualityCuts.add<double>("maxChi2TripletsOrQuadruplets", 1.)
        ->setComment("Max normalized chi2 for tracks with 4 or less hits");
    trackQualityCuts.add<double>("maxChi2Quintuplets", 3.)->setComment("Max normalized chi2 for tracks with 5 hits");
    trackQualityCuts.add<double>("minPt", 0.9)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <typename TrackerTraits>
  reco::TracksSoACollection CAHitNtupletGenerator<TrackerTraits>::makeTuplesAsync(HitsOnDevice const& hits_d,
                                                                                  CAGeometryOnDevice const& geometry_d,
                                                                                  float bfield,
                                                                                  uint32_t nDoublets,
                                                                                  uint32_t nTracks,
                                                                                  Queue& queue) const {
    using HelixFit = HelixFit<TrackerTraits>;
    using GPUKernels = CAHitNtupletGeneratorKernels<TrackerTraits>;
    using TrackHitSoA = ::reco::TrackHitSoA;
    using HitContainer = caStructures::HitContainerT<TrackerTraits>;

    const int32_t H = m_params.algoParams_.avgHitsPerTrack_;

    reco::TracksSoACollection trackCollection(queue, static_cast<int32_t>(nTracks), static_cast<int32_t>(nTracks * H));

    auto tracks = trackCollection.view().tracks();

    auto trackingHits = hits_d.view().trackingHits();
    auto hitModules = hits_d.view().hitModules();

    auto layers = geometry_d.view().layers();
    auto graph = geometry_d.view().graph();
    auto modules = geometry_d.view().modules();

    // Don't bother if less than 2 this
    if (trackingHits.metadata().size() < 2) {
      const auto device = alpaka::getDev(queue);
      auto ntracks_d = cms::alpakatools::make_device_view(device, tracks.nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      return trackCollection;
    }
    GPUKernels kernels(
        m_params, hits_d.nHits(), hits_d.offsetBPIX2(), nDoublets, nTracks, layers.metadata().size(), queue);

    kernels.prepareHits(trackingHits, hitModules, layers, queue);
    kernels.buildDoublets(trackingHits, graph, layers, hits_d.offsetBPIX2(), queue);
    kernels.launchKernels(
        trackingHits, hits_d.offsetBPIX2(), layers.metadata().size(), trackCollection.view(), layers, graph, queue);

    HelixFit fitter(bfield, m_params.algoParams_.fitNas4_);
    fitter.allocate(kernels.tupleMultiplicity(), tracks, kernels.hitContainer());
    if (m_params.algoParams_.useRiemannFit_) {
      fitter.launchRiemannKernels(
          trackingHits, modules, trackingHits.metadata().size(), TrackerTraits::maxNumberOfQuadruplets, queue);
    } else {
      fitter.launchBrokenLineKernels(
          trackingHits, modules, trackingHits.metadata().size(), TrackerTraits::maxNumberOfQuadruplets, queue);
    }
    kernels.classifyTuples(trackingHits, tracks, queue);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished building pixel tracks on GPU" << std::endl;
#endif

    return trackCollection;
  }

  template class CAHitNtupletGenerator<pixelTopology::Phase1>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2OT>;
  template class CAHitNtupletGenerator<pixelTopology::HIonPhase1>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
