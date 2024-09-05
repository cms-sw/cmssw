//#define GPU_DEBUG
//#define DUMP_GPU_TK_TUPLES

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
    using namespace pixelTopology;
    using namespace pixelTrack;

    template <typename T>
    T sqr(T x) {
      return x * x;
    }

    //Common Params
    void fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
      // 87 cm/GeV = 1/(3.8T * 0.3)
      // take less than radius given by the hardPtCut and reject everything below
      // auto hardCurvCut = 1.f/(0.35 * 87.f);
      desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
      desc.add<double>("CAThetaCutBarrel", 0.002f)->setComment("Cut on RZ alignement for Barrel");
      desc.add<double>("CAThetaCutForward", 0.003f)->setComment("Cut on RZ alignment for Forward");
      desc.add<double>("hardCurvCut", 1.f / (0.35 * 87.f))
          ->setComment("Cut on minimum curvature, used in DCA ntuplet selection");
      desc.add<double>("dcaCutInnerTriplet", 0.15f)->setComment("Cut on origin radius when the inner hit is on BPix1");
      desc.add<double>("dcaCutOuterTriplet", 0.25f)->setComment("Cut on origin radius when the outer hit is on BPix1");
      desc.add<bool>("earlyFishbone", true);
      desc.add<bool>("lateFishbone", false);
      desc.add<bool>("fillStatistics", false);
      desc.add<unsigned int>("minHitsPerNtuplet", 4);
      desc.add<unsigned int>("minHitsForSharingCut", 10)
          ->setComment("Maximum number of hits in a tuple to clean also if the shared hit is on bpx1");

      desc.add<bool>("fitNas4", false)->setComment("fit only 4 hits out of N");
      desc.add<bool>("doClusterCut", true);
      desc.add<bool>("doZ0Cut", true);
      desc.add<bool>("doPtCut", true);
      desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
      desc.add<bool>("doSharedHitCut", true)->setComment("Sharing hit nTuples cleaning");
      desc.add<bool>("dupPassThrough", false)->setComment("Do not reject duplicate");
      desc.add<bool>("useSimpleTripletCleaner", true)->setComment("use alternate implementation");
    }

    AlgoParams makeCommonParams(edm::ParameterSet const& cfg) {
      return AlgoParams({cfg.getParameter<unsigned int>("minHitsForSharingCut"),
                         cfg.getParameter<bool>("useRiemannFit"),
                         cfg.getParameter<bool>("fitNas4"),
                         cfg.getParameter<bool>("includeJumpingForwardDoublets"),
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
      static constexpr CAParamsT<TrackerTraits> makeCACuts(edm::ParameterSet const& cfg) {
        return CAParamsT<TrackerTraits>{{cfg.getParameter<unsigned int>("maxNumberOfDoublets"),
                                         cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
                                         (float)cfg.getParameter<double>("ptmin"),
                                         (float)cfg.getParameter<double>("CAThetaCutBarrel"),
                                         (float)cfg.getParameter<double>("CAThetaCutForward"),
                                         (float)cfg.getParameter<double>("hardCurvCut"),
                                         (float)cfg.getParameter<double>("dcaCutInnerTriplet"),
                                         (float)cfg.getParameter<double>("dcaCutOuterTriplet")}};
      };

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
      static constexpr CAParamsT<TrackerTraits> makeCACuts(edm::ParameterSet const& cfg) {
        return CAParamsT<TrackerTraits>{{cfg.getParameter<unsigned int>("maxNumberOfDoublets"),
                                         cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
                                         (float)cfg.getParameter<double>("ptmin"),
                                         (float)cfg.getParameter<double>("CAThetaCutBarrel"),
                                         (float)cfg.getParameter<double>("CAThetaCutForward"),
                                         (float)cfg.getParameter<double>("hardCurvCut"),
                                         (float)cfg.getParameter<double>("dcaCutInnerTriplet"),
                                         (float)cfg.getParameter<double>("dcaCutOuterTriplet")},
                                        {(bool)cfg.getParameter<bool>("includeFarForwards")}};
      }

      static constexpr ::pixelTrack::QualityCutsT<TrackerTraits> makeQualityCuts(edm::ParameterSet const& pset) {
        return ::pixelTrack::QualityCutsT<TrackerTraits>{
            static_cast<float>(pset.getParameter<double>("maxChi2")),
            static_cast<float>(pset.getParameter<double>("minPt")),
            static_cast<float>(pset.getParameter<double>("maxTip")),
            static_cast<float>(pset.getParameter<double>("maxZip")),
        };
      }
    };

    //Cell Cuts, as they are the cuts have the same logic for Phase2 and Phase1
    //keeping them separate would allow further differentiation in the future
    //moving them to TopologyCuts and using the same syntax
    template <typename TrackerTraits>
    CellCutsT<TrackerTraits> makeCellCuts(edm::ParameterSet const& cfg) {
      return CellCutsT<TrackerTraits>{cfg.getParameter<bool>("doClusterCut"),
                                      cfg.getParameter<bool>("doZ0Cut"),
                                      cfg.getParameter<bool>("doPtCut"),
                                      cfg.getParameter<bool>("idealConditions"),
                                      (float)cfg.getParameter<double>("cellZ0Cut"),
                                      (float)cfg.getParameter<double>("cellPtCut"),
                                      cfg.getParameter<std::vector<int>>("phiCuts")};
    }

  }  // namespace

  using namespace std;

  template <typename TrackerTraits>
  CAHitNtupletGenerator<TrackerTraits>::CAHitNtupletGenerator(const edm::ParameterSet& cfg)
      : m_params(makeCommonParams(cfg),
                 makeCellCuts<TrackerTraits>(cfg),
                 TopologyCuts<TrackerTraits>::makeQualityCuts(cfg.getParameterSet("trackQualityCuts")),
                 TopologyCuts<TrackerTraits>::makeCACuts(cfg)) {
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

    desc.add<unsigned int>("maxNumberOfDoublets", pixelTopology::Phase1::maxNumberOfDoublets);
    desc.add<bool>("idealConditions", true);
    desc.add<bool>("includeJumpingForwardDoublets", false);
    desc.add<double>("cellZ0Cut", 12.0);
    desc.add<double>("cellPtCut", 0.5);

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

    desc.add<std::vector<int>>(
            "phiCuts",
            std::vector<int>(std::begin(phase1PixelTopology::phicuts), std::end(phase1PixelTopology::phicuts)))
        ->setComment("Cuts in phi for cells");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::HIonPhase1>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon(desc);

    desc.add<unsigned int>("maxNumberOfDoublets", pixelTopology::HIonPhase1::maxNumberOfDoublets);
    desc.add<bool>("idealConditions", false);
    desc.add<bool>("includeJumpingForwardDoublets", false);
    desc.add<double>("cellZ0Cut", 10.0);
    desc.add<double>("cellPtCut", 0.0);

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

    desc.add<std::vector<int>>(
            "phiCuts",
            std::vector<int>(std::begin(phase1PixelTopology::phicuts), std::end(phase1PixelTopology::phicuts)))
        ->setComment("Cuts in phi for cells");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon(desc);

    desc.add<unsigned int>("maxNumberOfDoublets", pixelTopology::Phase2::maxNumberOfDoublets);
    desc.add<bool>("idealConditions", false);
    desc.add<bool>("includeFarForwards", true);
    desc.add<bool>("includeJumpingForwardDoublets", true);
    desc.add<double>("cellZ0Cut", 7.5);
    desc.add<double>("cellPtCut", 0.85);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 5.)->setComment("Max normalized chi2");
    trackQualityCuts.add<double>("minPt", 0.5)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");

    desc.add<std::vector<int>>(
            "phiCuts",
            std::vector<int>(std::begin(phase2PixelTopology::phicuts), std::end(phase2PixelTopology::phicuts)))
        ->setComment("Cuts in phi for cells");
  }

  template <typename TrackerTraits>
  TracksSoACollection<TrackerTraits> CAHitNtupletGenerator<TrackerTraits>::makeTuplesAsync(
      HitsOnDevice const& hits_d, ParamsOnDevice const* cpeParams, float bfield, Queue& queue) const {
    using HelixFit = HelixFit<TrackerTraits>;
    using TrackSoA = TracksSoACollection<TrackerTraits>;
    using GPUKernels = CAHitNtupletGeneratorKernels<TrackerTraits>;

    TrackSoA tracks(queue);

    // Don't bother if less than 2 this
    if (hits_d.view().metadata().size() < 2) {
      const auto device = alpaka::getDev(queue);
      auto ntracks_d = cms::alpakatools::make_device_view(device, tracks.view().nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      return tracks;
    }
    GPUKernels kernels(m_params, hits_d.view().metadata().size(), hits_d.offsetBPIX2(), queue);

    kernels.buildDoublets(hits_d.view(), hits_d.offsetBPIX2(), queue);
    kernels.launchKernels(hits_d.view(), hits_d.offsetBPIX2(), tracks.view(), queue);

    HelixFit fitter(bfield, m_params.fitNas4_);
    fitter.allocate(kernels.tupleMultiplicity(), tracks.view());
    if (m_params.useRiemannFit_) {
      fitter.launchRiemannKernels(
          hits_d.view(), cpeParams, hits_d.view().metadata().size(), TrackerTraits::maxNumberOfQuadruplets, queue);
    } else {
      fitter.launchBrokenLineKernels(
          hits_d.view(), cpeParams, hits_d.view().metadata().size(), TrackerTraits::maxNumberOfQuadruplets, queue);
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
