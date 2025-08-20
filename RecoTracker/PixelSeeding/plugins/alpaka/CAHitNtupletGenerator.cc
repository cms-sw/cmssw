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

    //Common Params
    void fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
      desc.add<double>("cellZ0Cut", 12.0f)->setComment("Z0 cut for cells");
      desc.add<double>("cellPtCut", 0.5f)->setComment("Preliminary pT cut at cell building level.");

      //// Pixel Cluster Cuts (@cell level)
      desc.add<double>("dzdrFact", 8.0f * 0.0285f / 0.015f);
      desc.add<int>("minYsizeB1", 1)
          ->setComment("Cut on inner hit cluster size (in Y) for barrel-forward cells. Barrel 1 cut.");
      desc.add<int>("minYsizeB2", 1)
          ->setComment("Cut on inner hit cluster size (in Y) for barrel-forward cells. Barrel 2 cut.");
      desc.add<int>("maxDYsize12", 28)
          ->setComment("Cut on cluster size differences (in Y) for barrel-forward cells. Barrel 1-2 cells.");
      desc.add<int>("maxDYsize", 20)
          ->setComment("Cut on cluster size differences (in Y) for barrel-forward cells. Other barrel cells.");
      desc.add<int>("maxDYPred", 20)
          ->setComment("Cut on cluster size differences (in Y) for barrel-forward cells. Barrel-forward cells.");

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

      desc.add<std::string>("maxNumberOfDoublets", std::to_string(pixelTopology::Phase1::maxNumberOfDoublets))
          ->setComment(
              "Max nummber of doublets (cells) as a string. The string will be parsed to a TFormula, depending on "
              "nHits (labeled 'x'), \n and evaluated for each event. May also be a constant.");
      desc.add<std::string>("maxNumberOfTuples", std::to_string(pixelTopology::Phase1::maxNumberOfTuples))
          ->setComment("Max nummber of tuples as a string. Same behavior as maxNumberOfDoublets.");
      desc.add<double>("avgHitsPerTrack", 5.0f)->setComment("Number of hits per track. Average per track.");
      desc.add<double>("avgCellsPerHit", 25.0f)
          ->setComment("Number of cells for which an hit is the outer hit. Average per hit.");
      desc.add<double>("avgCellsPerCell", 2.0f)
          ->setComment("Number of cells connected to another cell. Average per cell.");
      desc.add<double>("avgTracksPerCell", 1.0f)
          ->setComment("Number of tracks to which a cell belongs. Average per cell.");

      // nTuplet Cuts and Params
      desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
      //// p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
      //// 87 cm/GeV = 1/(3.8T * 0.3)
      //// take less than radius given by the hardPtCut and reject everything below
      desc.add<double>("hardCurvCut", 1.f / (0.35 * 87.f))
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
          (float)cfg.getParameter<double>("cellPtCut"),

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
    fillDescriptionsCommon(desc);

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
            "\"region "
            "cuts\" based on the fit results (pT, Tip, Zip).");

    edm::ParameterSetDescription geometryParams;
    using namespace phase1PixelTopology;
    // layers params
    geometryParams
        .add<std::vector<double>>("caDCACuts",
                                  std::vector<double>(std::begin(dcaCuts), std::begin(dcaCuts) + numberOfLayers))
        ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
    geometryParams
        .add<std::vector<double>>("caThetaCuts",
                                  std::vector<double>(std::begin(thetaCuts), std::begin(thetaCuts) + numberOfLayers))
        ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
    geometryParams.add<std::vector<unsigned int>>("startingPairs", {0u, 1u, 2u})
        ->setComment(
            "The list of the ids of pairs from which the CA ntuplets building may start.");  //TODO could be parsed via an expression
    // cells params
    geometryParams
        .add<std::vector<unsigned int>>(
            "pairGraph",
            std::vector<unsigned int>(std::begin(layerPairs),
                                      std::begin(layerPairs) + (pixelTopology::Phase1::nPairsForQuadruplets * 2)))
        ->setComment("CA graph");
    geometryParams
        .add<std::vector<int>>(
            "phiCuts",
            std::vector<int>(std::begin(phicuts), std::begin(phicuts) + pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in phi for cells");
    geometryParams
        .add<std::vector<double>>(
            "minZ",
            std::vector<double>(std::begin(minz), std::begin(minz) + pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in min z (on inner hit) for cells");
    geometryParams
        .add<std::vector<double>>(
            "maxZ",
            std::vector<double>(std::begin(maxz), std::begin(maxz) + pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in max z (on inner hit) for cells");
    geometryParams
        .add<std::vector<double>>(
            "maxR",
            std::vector<double>(std::begin(maxr), std::begin(maxr) + pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in max r for cells");

    desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::HIonPhase1>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon(desc);

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
            "\"region "
            "cuts\" based on the fit results (pT, Tip, Zip).");

    edm::ParameterSetDescription geometryParams;
    using namespace phase1PixelTopology;
    // layers params
    geometryParams
        .add<std::vector<double>>("caDCACuts",
                                  std::vector<double>(std::begin(phase1HIonPixelTopology::dcaCuts),
                                                      std::begin(phase1HIonPixelTopology::dcaCuts) + numberOfLayers))
        ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
    geometryParams
        .add<std::vector<double>>("caThetaCuts",
                                  std::vector<double>(std::begin(phase1HIonPixelTopology::thetaCuts),
                                                      std::begin(phase1HIonPixelTopology::thetaCuts) + numberOfLayers))
        ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
    geometryParams.add<std::vector<unsigned int>>("startingPairs", {0u, 1u, 2u})
        ->setComment(
            "The list of the ids of pairs from which the CA ntuplets building may start.");  //TODO could be parsed via an expression
    // cells params
    geometryParams
        .add<std::vector<unsigned int>>(
            "pairGraph",
            std::vector<unsigned int>(std::begin(layerPairs),
                                      std::begin(layerPairs) + (pixelTopology::Phase1::nPairsForQuadruplets * 2)))
        ->setComment("CA graph");
    geometryParams
        .add<std::vector<int>>("phiCuts",
                               std::vector<int>(std::begin(phase1HIonPixelTopology::phicuts),
                                                std::begin(phase1HIonPixelTopology::phicuts) +
                                                    pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in phi for cells");
    geometryParams
        .add<std::vector<double>>(
            "minZ",
            std::vector<double>(std::begin(minz), std::begin(minz) + pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in min z (on inner hit) for cells");
    geometryParams
        .add<std::vector<double>>(
            "maxZ",
            std::vector<double>(std::begin(maxz), std::begin(maxz) + pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in max z (on inner hit) for cells");
    geometryParams
        .add<std::vector<double>>(
            "maxR",
            std::vector<double>(std::begin(maxr), std::begin(maxr) + pixelTopology::Phase1::nPairsForQuadruplets))
        ->setComment("Cuts in max r for cells");

    desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 5.)->setComment("Max normalized chi2");
    trackQualityCuts.add<double>("minPt", 0.5)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");

    edm::ParameterSetDescription geometryParams;
    using namespace phase2PixelTopology;
    // layers params
    geometryParams
        .add<std::vector<double>>("caDCACuts",
                                  std::vector<double>(std::begin(dcaCuts), std::begin(dcaCuts) + numberOfLayers))
        ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
    geometryParams
        .add<std::vector<double>>("caThetaCuts",
                                  std::vector<double>(std::begin(thetaCuts), std::begin(thetaCuts) + numberOfLayers))
        ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
    geometryParams
        .add<std::vector<unsigned int>>("startingPairs",
                                        {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32})
        ->setComment(
            "The list of the ids of pairs from which the CA ntuplets building may start.");  //TODO could be parsed via an expression
    // cells params
    geometryParams
        .add<std::vector<unsigned int>>(
            "pairGraph", std::vector<unsigned int>(std::begin(layerPairs), std::begin(layerPairs) + (nPairs * 2)))
        ->setComment("CA graph");
    geometryParams
        .add<std::vector<int>>("phiCuts", std::vector<int>(std::begin(phicuts), std::begin(phicuts) + nPairs))
        ->setComment("Cuts in phi for cells");
    geometryParams.add<std::vector<double>>("minZ", std::vector<double>(std::begin(minz), std::begin(minz) + nPairs))
        ->setComment("Cuts in min z (on inner hit) for cells");
    geometryParams.add<std::vector<double>>("maxZ", std::vector<double>(std::begin(maxz), std::begin(maxz) + nPairs))
        ->setComment("Cuts in max z (on inner hit) for cells");
    geometryParams.add<std::vector<double>>("maxR", std::vector<double>(std::begin(maxr), std::begin(maxr) + nPairs))
        ->setComment("Cuts in max r for cells");

    desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
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

    reco::TracksSoACollection tracks({{int(nTracks), int(nTracks * H)}}, queue);

    // Don't bother if less than 2 this
    if (hits_d.view().metadata().size() < 2) {
      const auto device = alpaka::getDev(queue);
      auto ntracks_d = cms::alpakatools::make_device_view(device, tracks.view().nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      return tracks;
    }
    GPUKernels kernels(
        m_params, hits_d.nHits(), hits_d.offsetBPIX2(), nDoublets, nTracks, geometry_d.view().metadata().size(), queue);

    kernels.prepareHits(hits_d.view(), hits_d.view<::reco::HitModuleSoA>(), geometry_d.view(), queue);
    kernels.buildDoublets(hits_d.view(),
                          geometry_d.view<::reco::CAGraphSoA>(),
                          geometry_d.view<::reco::CALayersSoA>(),
                          hits_d.offsetBPIX2(),
                          queue);
    kernels.launchKernels(hits_d.view(),
                          hits_d.offsetBPIX2(),
                          geometry_d.view().metadata().size(),
                          tracks.view(),
                          tracks.view<TrackHitSoA>(),
                          geometry_d.view<::reco::CALayersSoA>(),
                          geometry_d.view<::reco::CAGraphSoA>(),
                          queue);

    HelixFit fitter(bfield, m_params.algoParams_.fitNas4_);
    fitter.allocate(kernels.tupleMultiplicity(), tracks.view(), kernels.hitContainer());
    if (m_params.algoParams_.useRiemannFit_) {
      fitter.launchRiemannKernels(hits_d.view(),
                                  geometry_d.view<::reco::CAModulesSoA>(),
                                  hits_d.view().metadata().size(),
                                  TrackerTraits::maxNumberOfQuadruplets,
                                  queue);
    } else {
      fitter.launchBrokenLineKernels(hits_d.view(),
                                     geometry_d.view<::reco::CAModulesSoA>(),
                                     hits_d.view().metadata().size(),
                                     TrackerTraits::maxNumberOfQuadruplets,
                                     queue);
    }
    kernels.classifyTuples(hits_d.view(), tracks.view(), queue);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished building pixel tracks on GPU" << std::endl;
#endif

    return tracks;
  }

  template class CAHitNtupletGenerator<pixelTopology::Phase1>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2>;
  template class CAHitNtupletGenerator<pixelTopology::HIonPhase1>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
