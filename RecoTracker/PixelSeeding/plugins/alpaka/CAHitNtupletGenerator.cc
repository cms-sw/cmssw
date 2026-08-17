// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

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
    // Per-topology defaults. Phase2OTStubs -- the only topology this fork deploys -- carries the DEPLOYED
    // working point, so its HLT cfi need only state what differs from it. Every other topology keeps the
    // upstream defaults with this fork's added features OFF, so a menu that configures them gets upstream
    // behaviour.
    //
    // Generic (topology-independent) cut parameters keep the upstream names and semantics at the top
    // level of the module PSet -- every deployed Run-3 / HIon / Phase-2 menu sets them there. They are
    // consumed by filling the corresponding CAGeometry SoA scalars/columns in CAHitNtuplet.cc:
    //   ptmin       -> tripletCuts.ptmin           (scalar)
    //   hardCurvCut -> tripletCuts.maxCurv         (scalar)
    //   cellZ0Cut   -> doubletCuts.maxZ0           (broadcast to every layer pair)
    //   dzdrFact    -> doubletCuts.dzdrFact        (scalar)
    //   minYsizeB1  -> doubletCuts.minInnerSizeB1  (scalar)
    //   minYsizeB2  -> doubletCuts.minInnerSizeB2  (scalar)
    //   maxDYsize12 -> doubletCuts.maxDSizeB1      (scalar)
    //   maxDYsize   -> doubletCuts.maxDSize        (scalar)
    //   maxDYPred   -> doubletCuts.maxDSizePred    (scalar)
    // The optional per-layer-pair entries of the `geometry` PSet override the same values: when present
    // they win, when absent the generic scalar above is used. Only the stub-seeded configurations set them.
    template <typename TrackerTraits>
    void fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
      constexpr bool kStubs = std::is_same_v<TrackerTraits, pixelTopology::Phase2OTStubs>;
      constexpr bool kPhase1 = std::is_same_v<TrackerTraits, pixelTopology::Phase1>;

      // ---------------------------------------------
      // Generic cut parameters
      // ---------------------------------------------
      desc.add<double>("cellZ0Cut", TrackerTraits::cellZ0Cut)->setComment("Z0 cut for cells");

      //// Pixel Cluster Cuts (@cell level)
      desc.add<double>("dzdrFact", TrackerTraits::dzdrFact);
      desc.add<int>("minYsizeB1", TrackerTraits::minInnerSizeB1)
          ->setComment("Cut on inner hit cluster size (in global z / local y) for barrel-forward cells. Barrel 1 cut.");
      desc.add<int>("minYsizeB2", TrackerTraits::minInnerSizeB2)
          ->setComment(
              "Cut on inner hit cluster size (in global z / local y) for barrel-forward cells. Anything but Barrel 1 "
              "cut.");
      desc.add<int>("maxDYsize12", TrackerTraits::maxDSizeB1)
          ->setComment(
              "Cut on cluster size differences (in global z / local y) for barrel-forward cells. Barrel 1-2 cells.");
      desc.add<int>("maxDYsize", TrackerTraits::maxDSize)
          ->setComment(
              "Cut on cluster size differences (in global z / local y) for barrel-forward cells. Other barrel cells.");
      desc.add<int>("maxDYPred", TrackerTraits::maxDSizePred)
          ->setComment(
              "Maximum difference between actual and expected cluster size of inner RecHit. Barrel-forward cells.");

      // nTuplet Cuts and Params
      desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
      //// p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
      //// 87 cm/GeV = 1/(3.8T * 0.3)
      //// take less than radius given by the hardPtCut and reject everything below
      desc.add<double>("hardCurvCut", TrackerTraits::maxCurv)
          ->setComment("Cut on minimum curvature, used in DCA ntuplet selection");

      // ---- OT-stub extension: triplet scalars with no upstream counterpart ----
      desc.add<double>("maxPhiResid", -1.0)
          ->setComment(
              "OT-stub extension. Max |phi residual| per connection during chain extension [rad]. "
              "Negative = disabled. Typical value when enabled: 0.01.");
      desc.add<bool>("sameDPhiSign", kStubs)
          ->setComment(
              "OT-stub extension. If true, require dPhi12 * dPhi23 > 0 in a triplet. Off on every "
              "topology but Phase2OTStubs.");

      // -------------------------------------------------------------------------------------------
      // Layer- and layer-pair-dependent configuration: the `geometry` PSet.
      //
      // The CA SoA is organised by purpose (layers / graph / doubletCuts / tripletCuts / ntupletCuts)
      // and indexes the two triplet cuts by LAYER PAIR rather than by layer; CAHitNtuplet.cc performs
      // the conversion when it builds the geometry product, broadcasting each per-layer value onto the
      // pairs whose own INNER layer it belongs to. That broadcast is exactly the indexing the kernels
      // read back (caDCACuts at the triplet's innermost layer, caThetaCuts at its middle layer).
      //
      // The OT-stub extension adds a small set of OPTIONAL parameters at the end of this block; every
      // one of them is absent from a non-stub configuration and inert when absent.
      // -------------------------------------------------------------------------------------------
      edm::ParameterSetDescription geometryParams;

      // ---- layers params ----
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
      // The list of pair ids the ntuplet building may start from. The topology tables carry the same
      // information as a per-pair flag array sized to the topology's own pair bound
      // (nPairsForQuadruplets), so the default id list is obtained by scanning that array.
      std::vector<unsigned int> startingPairsDefault;
      for (int pairId = 0; pairId < TrackerTraits::nPairsForQuadruplets; ++pairId)
        if (TrackerTraits::startingPairs[pairId])
          startingPairsDefault.push_back(static_cast<unsigned int>(pairId));
      geometryParams.add<std::vector<unsigned int>>("startingPairs", startingPairsDefault)
          ->setComment("The list of the ids of pairs from which the CA ntuplets building may start.");
      geometryParams
          .add<std::vector<double>>("startMaxInnerR", std::vector<double>(TrackerTraits::numberOfLayers, 99.0))
          ->setComment(
              "The maximum allowed r coordinate of the inner hit of a doublet to use it as a starting point for "
              "ntuplet building.");
      /*
      Cut on quadruplets (two triplets sharing a doublet) using the curvatures Ci, Co of the triplets:
      |Co - Ci| < (|Co| + |Ci|)/2 * maxDCurv + floorDCurv
      */
      geometryParams.add<std::vector<double>>("maxDCurv", std::vector<double>(TrackerTraits::numberOfLayers, 99.))
          ->setComment("Cut on curvature difference between two consecutive triplets.");
      geometryParams.add<std::vector<double>>("floorDCurv", std::vector<double>(TrackerTraits::numberOfLayers, 99.))
          ->setComment("Offset for the cut on curvature difference between two consecutive triplets.");
      geometryParams
          .add<std::vector<double>>("fishboneCuts", std::vector<double>(TrackerTraits::numberOfLayers, 0.99999f))
          ->setComment(
              "Threshold for merging aligned doublets in fishbone cleaning. Depends on the layer of the outer RecHit. "
              "Warning: this will be a float in the final algorithm, therefore 0.9999999 will become 1 == no merging!");

      // ---- cells params ----
      geometryParams
          .add<std::vector<unsigned int>>(
              "pairGraph",
              std::vector<unsigned int>(TrackerTraits::layerPairs,
                                        TrackerTraits::layerPairs + (TrackerTraits::nPairsForQuadruplets * 2)))
          ->setComment("CA graph (layer pairs used for building doublets/cells)");
      geometryParams
          .add<std::vector<unsigned int>>("skipsLayers",
                                          std::vector<unsigned int>(TrackerTraits::nPairsForQuadruplets, 0U))
          ->setComment(
              "List of bools idicating whether layer pairs are skipping layers or not (0 means non-skipping, 1 means "
              "skipping). This is relevant for the N-tuplet building as non-skipping ones are prioritized.");
      geometryParams
          .add<std::vector<int>>(
              "phiCuts",
              std::vector<int>(TrackerTraits::maxDPhi, TrackerTraits::maxDPhi + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in dphi for cells");
      geometryParams
          .add<std::vector<double>>(
              "ptCuts",
              std::vector<double>(TrackerTraits::minPt, TrackerTraits::minPt + TrackerTraits::nPairsForQuadruplets))
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

      // ---------------------------------------------------------------------------------------
      // OT-stub extension. All entries below are OPTIONAL: the three "...PerPair" vectors fall back
      // to the broadcast of their per-layer / scalar source, the stub-only cuts fall back to the
      // disabled sentinel -1.
      // ---------------------------------------------------------------------------------------
      geometryParams.addOptional<std::vector<double>>("caDCACutsPerPair")
          ->setComment(
              "OT-stub extension. Per-layer-pair override of caDCACuts: one entry per layer pair, replacing the "
              "broadcast of the per-layer caDCACuts. Absent = use caDCACuts.");
      geometryParams.addOptional<std::vector<double>>("caThetaCutsPerPair")
          ->setComment(
              "OT-stub extension. Per-layer-pair override of caThetaCuts: one entry per layer pair, replacing the "
              "broadcast of the per-layer caThetaCuts. Absent = use caThetaCuts.");
      geometryParams.addOptional<std::vector<double>>("cellZ0CutPerPair")
          ->setComment(
              "OT-stub extension. Per-layer-pair override of the scalar cellZ0Cut: one entry per layer pair. "
              "Absent = use cellZ0Cut for every pair.");
      geometryParams.addOptional<std::vector<double>>("floorDCACuts")
          ->setComment(
              "OT-stub extension. Additive DCA floor for high-pT tracks, one entry per layer pair "
              "(dcaThreshold = caDCACut * curvature + floorDCACut), which prevents the threshold from "
              "collapsing at high pT. Negative = disabled. Absent = disabled on every pair.");
      geometryParams.addOptional<std::vector<double>>("maxStubCurvSigma")
          ->setComment(
              "OT-stub extension. Stub-stub pairwise curvature-significance cut, one entry per layer pair. "
              "Negative = disabled. Barrel flat-flat: kappa-corrected significance; forward: dPhiDr significance.");
      geometryParams.addOptional<std::vector<double>>("maxStubGeomCurvSigma")
          ->setComment(
              "OT-stub extension. Geometric-vs-stub kappa significance cut, one entry per layer pair. "
              "Negative = disabled (no stubs on that pair).");
      geometryParams.addOptional<std::vector<double>>("maxStubInnerDoubletDCurv")
          ->setComment(
              "OT-stub extension. Cut on the phi residual at the middle hit [rad], one entry per layer pair. "
              "Negative = disabled. Requires nStubs >= 2 for a reliable stub-kappa prediction.");

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
      desc.add<uint32_t>("minNumberOfDoublets", kStubs ? 23552 : 0)
          ->setComment(
              "Per-event floor on the doublet capacity: the effective capacity is max(formula, floor). Sized "
              "from measured quiet-event demand (the hit-dependent formulas undershoot at low "
              "occupancy); demand beyond it is truncated and counted by the overflow sentinel. 0 removes the "
              "floor, leaving the formula alone to size the container as upstream does.");
      desc.add<uint32_t>("minNumberOfTuples", kStubs ? 13312 : 0)
          ->setComment("Per-event floor on the tuple capacity. Same behavior as minNumberOfDoublets.");
      desc.add<double>("avgHitsPerTrack", double(TrackerTraits::avgHitsPerTrack))
          ->setComment("Number of hits per track. Average per track.");
      desc.add<double>("avgCellsPerHit", TrackerTraits::avgCellsPerHit)
          ->setComment(
              "Number of cells for which an hit is the outer hit. Average per hit. NOTE: this no longer sizes any "
              "buffer. The hit->cell association holds exactly one entry per cell, so it is sized from the cell "
              "bound itself (the exact count when useExactAllocations is on, maxNumberOfDoublets otherwise). The "
              "parameter is kept because it is set by existing configurations and read by the sizing reporter.");
      desc.add<double>("avgCellsPerCell", TrackerTraits::avgCellsPerCell)
          ->setComment("Number of cells connected to another cell. Average per cell.");
      desc.add<double>("avgTracksPerCell", TrackerTraits::avgTracksPerCell)
          ->setComment("Number of tracks to which a cell belongs. Average per cell.");

      desc.add<bool>("earlyFishbone", true);
      desc.add<bool>("lateFishbone", false);
      desc.add<bool>("onlySameLayersFishbone", kStubs);
      desc.add<bool>("fillStatistics", false);
      desc.add<unsigned int>("minHitsPerNtuplet", 4)
          ->setComment("Minimum number of layers an N-tuplet must span to be kept.");
      desc.add<unsigned int>("minHitsForSharingCut", kStubs ? 1 : 10)
          ->setComment("Maximum number of hits in a tuple to clean also if the shared hit is on bpx1");

      desc.add<bool>("fitNas4", false)->setComment("fit only 4 hits out of N");
      desc.add<bool>("verboseBLFit", false)
          ->setComment("one-shot device dump of the first fitted tracks at the end of each BL-fit launch (debug)");

      desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
      desc.add<bool>("useFitCorrections", kStubs)
          ->setComment(
              "Correctness package of the factorized fast BrokenLine fit. true applies it as ONE unit: "
              "one thin scatterer per gap charged at that gap's arrival node (with the rigid-node guard for "
              "a material-free gap), the Karimaki-consistent Fisher basis in the covariance blend, the pion "
              "1/beta factor and no pt cap in the scattering variance, trapezoid quadrature in the material "
              "line integral, and the covariance blend completed to the full 3x3; false is the plain "
              "upstream algebra. It is the CA main fit's own material model, so the merger's General Broken "
              "Lines refit and the extender walk are unaffected. It moves the whole "
              "emitted covariance and the chi2, so every covariance-derived input of the high-purity "
              "selector shifts and covPhiDxy changes sign on part of the population: the selector's working "
              "point is NOT comparable across this switch, and flipping it requires retraining the selector. "
              "MUST agree with the merger's extPredFitCorrections. Defaults to true on Phase2OTStubs and "
              "false on every other topology.");
      desc.add<bool>("doSharedHitCut", true)->setComment("Sharing hit nTuples cleaning");
      desc.add<bool>("dupPassThrough", false)->setComment("Do not reject duplicate");
      desc.add<bool>("useSimpleTripletCleaner", true)->setComment("use alternate implementation");
      desc.add<bool>("doTripletCleaner", true)
          ->setComment(
              "Disable the triplet cleaner entirely.");  // FIXME this should be implemented as an automatic check (simple if) that disables if minHitsPerNtuplet > 3
      desc.add<bool>("doFastDuplicateRemover", true)->setComment("Disable the fastDuplicateRemover");
      desc.add<bool>("doEarlyDuplicateRemover", true)->setComment("Disable the earlyDuplicateRemover");

      // Device-memory allocation strategy
      desc.add<bool>("useExactAllocations", false)
          ->setComment(
              "Size the event-dependent device buffers from the measured per-event occupancy instead of "
              "from the configured caps: a count-only doublet pass plus one device readback per event "
              "determines the exact allocation sizes for the doublet store, the hit->cell association and "
              "the cell-derived containers, and the hit->track storage is sized from the actual "
              "hits-in-tracks count delivered through the acquire/produce boundary. Costs one host "
              "synchronization per event plus the count pass; adapts automatically to occupancy beyond "
              "the calibrated caps. false allocates everything up-front from the configured caps with "
              "zero synchronizations. On the CPU backend the count pass is skipped and the caps size the "
              "doublet store (identical results, no serial-time cost).");

      desc.add<double>("fastDupNSigma2", kStubs ? 2.5 : (kPhase1 ? 25.0 : 5.0))
          ->setComment(
              "CA duplicate-removal parameter-covariance gate width nSigma^2: two tracks are duplicates "
              "when every compared fitted parameter satisfies dp^2 <= fastDupNSigma2*(cov_i+cov_j) "
              "(Kernel_fastDuplicateRemover + Kernel_rejectDuplicate). It is a width in units of the "
              "fitted covariance, so it tracks the fit's covariance convention. Lower tightens (fewer "
              "merges); higher merges more aggressively. The default reproduces, topology by topology, "
              "the constant those kernels hard-wire upstream: 5 for the five-parameter check used by "
              "every Phase-2 topology, 25 for the two-parameter check of the Phase-1 specializations "
              "(which read their own nSigma2Phase1 constant and ignore this parameter, so the value is "
              "only informative there). Phase2OTStubs uses 2.5, matching the tighter covariance its "
              "fit emits.");
    }

    AlgoParams makeCommonParams(edm::ParameterSet const& cfg) {
      // Designated initializers: the field set of AlgoParams is shared with upstream and grows there,
      // so a positional aggregate initialization would silently shift every value the next time a
      // member is inserted. Keep this list keyed by name.
      return AlgoParams{
          // Container sizes
          .avgHitsPerTrack_ = (float)cfg.getParameter<double>("avgHitsPerTrack"),
          .avgCellsPerHit_ = (float)cfg.getParameter<double>("avgCellsPerHit"),
          .avgCellsPerCell_ = (float)cfg.getParameter<double>("avgCellsPerCell"),
          .avgTracksPerCell_ = (float)cfg.getParameter<double>("avgTracksPerCell"),

          // Algo params
          .minHitsPerNtuplet_ = (uint16_t)cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
          .minHitsForSharingCut_ = (uint16_t)cfg.getParameter<unsigned int>("minHitsForSharingCut"),

          // Flags
          .useRiemannFit_ = cfg.getParameter<bool>("useRiemannFit"),
          .useFitCorrections_ = cfg.getParameter<bool>("useFitCorrections"),
          .fitNas4_ = cfg.getParameter<bool>("fitNas4"),
          .earlyFishbone_ = cfg.getParameter<bool>("earlyFishbone"),
          .lateFishbone_ = cfg.getParameter<bool>("lateFishbone"),
          .onlySameLayersFishbone_ = cfg.getParameter<bool>("onlySameLayersFishbone"),
          .doStats_ = cfg.getParameter<bool>("fillStatistics"),
          .doSharedHitCut_ = cfg.getParameter<bool>("doSharedHitCut"),
          .dupPassThrough_ = cfg.getParameter<bool>("dupPassThrough"),
          .useSimpleTripletCleaner_ = cfg.getParameter<bool>("useSimpleTripletCleaner"),
          .doTripletCleaner_ = cfg.getParameter<bool>("doTripletCleaner"),
          .doFastDuplicateRemover_ = cfg.getParameter<bool>("doFastDuplicateRemover"),
          .doEarlyDuplicateRemover_ = cfg.getParameter<bool>("doEarlyDuplicateRemover"),

          // Allocation strategy: the single useExactAllocations switch drives both internal modes --
          // the deferred demand-based allocations (delayAllocations_) and the count-only doublet pass
          // (countDoubletsFirst_). The count pass is forced off on the CPU backend: it costs real
          // serial time, while its only benefit is trimming a device allocation to the exact size --
          // on the host the generous cap is cheap. The accepted doublet set is identical either way
          // (the capacity guard in the doublet writer never binds in either regime), so the physics
          // output does not depend on this choice and one configuration runs unchanged on every
          // backend.
          .delayAllocations_ = cfg.getParameter<bool>("useExactAllocations"),
          .countDoubletsFirst_ =
              cfg.getParameter<bool>("useExactAllocations") && !std::is_same_v<Device, alpaka::DevCpu>,

          // CA fast-duplicate / shared-hit covariance gate width, in units of the fitted covariance
          .fastDupNSigma2_ = (float)cfg.getParameter<double>("fastDupNSigma2"),
      };
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
            static_cast<float>(pset.getParameter<double>("maxNtupletStubChi2")),
        };
      }
    };

  }  // namespace

  using namespace std;

  template <typename TrackerTraits>
  CAHitNtupletGenerator<TrackerTraits>::CAHitNtupletGenerator(const edm::ParameterSet& cfg)
      : m_params(makeCommonParams(cfg),
                 TopologyCuts<TrackerTraits>::makeQualityCuts(cfg.getParameterSet("trackQualityCuts"))),
        m_verboseBLDump(cfg.getParameter<bool>("verboseBLFit")) {
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
    trackQualityCuts.add<double>("maxNtupletStubChi2", -1.)
        ->setComment(
            "Ntuplet-wide stub-curvature consistency: max reduced-chi2 of the per-stub curvatures of all stubs on a "
            "track around their inverse-variance-weighted mean. Demotes inconsistent (combinatorial) tracks below "
            "tight. Negative = disabled.");
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
    trackQualityCuts.add<double>("maxNtupletStubChi2", -1.)
        ->setComment(
            "Ntuplet-wide stub-curvature consistency: max reduced-chi2 of the per-stub curvatures of all stubs on a "
            "track around their inverse-variance-weighted mean. Demotes inconsistent (combinatorial) tracks below "
            "tight. Negative = disabled.");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2OTStubs>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase2OTStubs>(desc);

    // Defaults follow the HLT configuration (both HLT cfis set the same three chi2 values).
    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 7.)->setComment("Max normalized chi2 for tracks with 6 or more hits");
    trackQualityCuts.add<double>("maxChi2TripletsOrQuadruplets", 7.)
        ->setComment("Max normalized chi2 for tracks with 4 or less hits");
    trackQualityCuts.add<double>("maxChi2Quintuplets", 7.)->setComment("Max normalized chi2 for tracks with 5 hits");
    trackQualityCuts.add<double>("minPt", 0.9)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    trackQualityCuts.add<double>("maxNtupletStubChi2", -1.)
        ->setComment(
            "Ntuplet-wide stub-curvature consistency: max reduced-chi2 of the per-stub curvatures of all stubs on a "
            "track around their inverse-variance-weighted mean. Demotes inconsistent (combinatorial) tracks below "
            "tight. Negative = disabled.");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <typename TrackerTraits>
  typename CAHitNtupletGenerator<TrackerTraits>::PendingTuples CAHitNtupletGenerator<TrackerTraits>::beginTuplesAsync(
      HitsOnDeviceRefProdVector const& hitsRefProdVector,
      CAGeometryOnDevice const& geometry_d,
      float bfield,
      uint32_t nDoublets,
      uint32_t nTracks,
      Queue& queue,
      const float* rhoMapDevice) const {
    using GPUKernels = CAHitNtupletGeneratorKernels<TrackerTraits>;

    PendingTuples pending;
    pending.bfield = bfield;
    pending.rhoMapDevice = rhoMapDevice;
    pending.maxTuples = nTracks;
    pending.maxDoublets = nDoublets;

    // Output trackHits rows: the same expression the kernels use for the internal per-track hit
    // container, so the SoA the hits are copied into is never smaller than the container that
    // feeds it (see caHitNtupletGenerator::nHitRowsForTuples).
    const uint32_t nHitRows = caHitNtupletGenerator::nHitRowsForTuples(nTracks, m_params.algoParams_.avgHitsPerTrack_);
    pending.tracks.emplace(queue, nTracks, nHitRows);

    auto tracks = pending.tracks->view().tracks();

    HitsMultiView trackingHits(
        hitsRefProdVector, [](edm::RefProd<HitsOnDevice> hits) -> auto { return hits->const_view().trackingHits(); });

    std::vector<int> hitModulesSizes;
    for (const auto& hit : hitsRefProdVector) {
      hitModulesSizes.push_back(static_cast<int>(hit->nModules()));
    }

    // We need to encounter for the last hidden module, so we add 1 to the last element of hitModulesSizes
    if (!hitModulesSizes.empty()) {
      ++hitModulesSizes.back();
    }

    ModulesMultiView hitModules(
        hitsRefProdVector,
        [](edm::RefProd<HitsOnDevice> hits) -> auto { return hits->const_view().hitModules(); },
        hitModulesSizes);

    auto layers = geometry_d.view().layers();
    auto graph = geometry_d.view().graph();
    auto doubletCuts = geometry_d.view().doubletCuts();
    auto tripletCuts = geometry_d.view().tripletCuts();
    auto ntupletCuts = geometry_d.view().ntupletCuts();

    const uint32_t nHits = static_cast<uint32_t>(trackingHits.size());
    const uint32_t offsetBPIX2 = static_cast<uint32_t>(hitsRefProdVector[0]->offsetBPIX2());

    // Don't bother if less than 2 hits: return the empty (built=false) pending state; finish
    // hands the zeroed collection through.
    if (trackingHits.size() < 2) {
      const auto device = alpaka::getDev(queue);
      auto ntracks_d = cms::alpakatools::make_device_view(device, tracks.nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      return pending;
    }
    pending.kernels =
        std::make_unique<GPUKernels>(m_params, nHits, offsetBPIX2, nDoublets, nTracks, layers.metadata().size(), queue);
    auto& kernels = *pending.kernels;

    // Lazy per-stream init of the always-on overflow accumulator (kOvfWords device words) and of
    // its pinned mirror (two kOvfWords slots, see the member declaration), then arm the kernels
    // object so classifyTuples launches the sentinel into it.
    if (!ovfAccum_) {
      ovfAccum_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kOvfWords));
      ovfHost_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2u * kOvfWords));
      alpaka::memset(queue, *ovfAccum_, 0);
      std::fill_n(ovfHost_->data(), 2u * kOvfWords, 0u);
    }
    kernels.ovfAccum_ = ovfAccum_->data();

    // Both internal switches are driven by the single useExactAllocations configuration flag (the
    // count pass is additionally excluded on the CPU backend -- see the parameter construction).
    // With the flag off, all buffers are allocated up-front in the kernels constructor and there is
    // no device->host synchronization here.
    const bool delay = m_params.algoParams_.delayAllocations_;
    // The count pass already reads the doublet count back to the host inside buildDoublets, to size
    // the cells buffer and the hit->cell storage. That value is handed back here and reused as the
    // allocateAfterDoublets basis, so the scalar crosses the bus (and blocks the host) exactly once
    // per event.
    const bool countFirst = m_params.algoParams_.countDoubletsFirst_;

    kernels.prepareHits(trackingHits, hitModules, layers, queue);

    const uint32_t nCellsCounted = kernels.buildDoublets(trackingHits, graph, layers, doubletCuts, offsetBPIX2, queue);

    if (delay) {
      // Size the cell-derived buffers from the actual number of doublets. With countDoubletsFirst on
      // the count pass already brought that number across (nCellsCounted): reuse it, no second sync.
      // Without it, the number only exists on the device and one D2H sync is unavoidable here.
      const uint32_t nCells = countFirst ? nCellsCounted : kernels.readbackNCells(queue);
      kernels.allocateAfterDoublets(nCells, queue);
    }

    kernels.launchKernels(trackingHits,
                          offsetBPIX2,
                          layers.metadata().size(),
                          pending.tracks->view(),
                          layers,
                          graph,
                          tripletCuts,
                          ntupletCuts,
                          queue);

    // Size the hit->track storage from the actual hits-in-tracks count. Under delayAllocations
    // the count is not read back here: the counter block is enqueued as an async D2H into the
    // pending buffer, the framework seam guarantees it has landed before finishTuplesAsync runs,
    // and the allocation happens there -- still ahead of its first consumer (classifyTuples).
    // The demand-exact sizing is kept at zero host-blocking cost. Without delay the storage is
    // allocated up-front and the call here only keeps the GPU_DEBUG allocation report complete.
    if (delay) {
      pending.countsHost.emplace(
          cms::alpakatools::make_host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 5u));
      kernels.enqueueCountsReadback(*pending.countsHost, queue);
    } else {
      kernels.allocateAfterNtuplets(0u, queue);
    }

    // Seam readback: one async D2H of the tuple-multiplicity per-N-bin offsets into the pinned
    // pending buffer. No wait here -- the framework's acquire->produce boundary guarantees the
    // copy has landed before finishTuplesAsync consumes it (both fit passes then run with the
    // launch-elision information for free; see HelixFit::setHostTupleMultiplicityOffsets).
    if (const uint32_t* offDev = kernels.tupleMultiplicityOffsets(); offDev != nullptr) {
      constexpr uint32_t kNOff = TrackerTraits::maxHitsOnTrack + 2u;
      pending.offsetsHost.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, std::size_t(kNOff)));
      auto offSrc =
          cms::alpakatools::make_device_view(alpaka::getDev(queue), const_cast<uint32_t*>(offDev), std::size_t(kNOff));
      alpaka::memcpy(queue, *pending.offsetsHost, offSrc);
    }
    pending.built = true;
    return pending;
  }

  template <typename TrackerTraits>
  reco::TracksSoACollection CAHitNtupletGenerator<TrackerTraits>::finishTuplesAsync(
      PendingTuples&& pending,
      HitsOnDeviceRefProdVector const& hitsRefProdVector,
      CAGeometryOnDevice const& geometry_d,
      Queue& queue) const {
    using HelixFit = HelixFit<TrackerTraits>;

    if (!pending.built)
      return std::move(*pending.tracks);

    auto& kernels = *pending.kernels;

    // Deferred hit->track storage sizing (delayAllocations): the counter block enqueued in
    // beginTuplesAsync has landed (framework seam guarantee). Word [0] is the tuple
    // AtomicPairCounter -- low half tuple count, high half hits-in-tracks total; every track has
    // >= 1 hit, so the larger half is the hits-in-tracks total. Allocate here, ahead of
    // classifyTuples, its first consumer.
    if (pending.countsHost) {
      const uint64_t apcRaw = static_cast<uint64_t>(pending.countsHost->data()[0]);
      const uint32_t apcLo = static_cast<uint32_t>(apcRaw & 0xFFFFFFFFull);
      const uint32_t apcHi = static_cast<uint32_t>(apcRaw >> 32);
      kernels.allocateAfterNtuplets(std::max(apcLo, apcHi), queue);
    }

    auto tracks = pending.tracks->view().tracks();
    HitsMultiView trackingHits(
        hitsRefProdVector, [](edm::RefProd<HitsOnDevice> hits) -> auto { return hits->const_view().trackingHits(); });
    auto modules = geometry_d.view().modules();
    const uint32_t nTracks = pending.maxTuples;
    const float bfield = pending.bfield;

    HelixFit fitter(bfield, m_params.algoParams_.fitNas4_);
    fitter.setVerboseDump(m_verboseBLDump);
    fitter.setMaterialMap(pending.rhoMapDevice);  // device BLMaterialMap (EventSetup condition)
    fitter.allocate(kernels.tupleMultiplicity(), tracks, kernels.hitContainer());
    // The per-N-bin offsets were read back once in beginTuplesAsync and have landed (framework
    // seam); the fit consumes the host values -- zero fit-side readbacks or waits. If the
    // container was absent, fall back to the cap-bounded fit (no elision).
    if (pending.offsetsHost)
      fitter.setHostTupleMultiplicityOffsets(pending.offsetsHost->data());
    if (m_params.algoParams_.useRiemannFit_) {
      fitter.launchRiemannKernels(
          trackingHits, modules, trackingHits.size(), TrackerTraits::maxNumberOfQuadruplets, queue);
    } else {
      // The CA main fit is the factorized fast BrokenLine fit (circle+line, 5 params + factorized cov +
      // chi2), one launchBrokenLineKernels call for every CA iteration and every topology. The fit's
      // per-tuple work bound is the runtime tuple capacity nTracks (the
      // per-event cap the tuple containers are sized to), not the compile-time maxNumberOfQuadruplets:
      // tuple ids are always < nTracks, so the extra chunks the launcher would iterate over with the
      // compile-time cap could only launch empty kernels.
      fitter.setFitCorrections(m_params.algoParams_.useFitCorrections_);
      fitter.launchBrokenLineKernels(trackingHits, modules, trackingHits.size(), nTracks, queue);
    }
    kernels.classifyTuples(trackingHits, tracks, queue);

    // Refresh the pinned overflow mirror with the running totals. Ordered after classifyTuples,
    // which is where the always-on overflow sentinel is enqueued on this same queue, so the
    // snapshot includes this event. Async, no wait; consumed at endStream via reportOverflows
    // once every event queue has drained. The destination alternates between the mirror's two
    // slots so that the copies of two consecutive events -- which run on different queues and can
    // therefore overlap -- never write the same bytes.
    const uint32_t ovfSlot = ovfSlot_;
    ovfSlot_ ^= 1u;
    auto ovfHostSlot = cms::alpakatools::make_host_view(ovfHost_->data() + ovfSlot * kOvfWords, kOvfWords);
    alpaka::memcpy(queue, ovfHostSlot, *ovfAccum_);
#ifdef CA_SIZING_DUMP
    // Per-event per-container demand dump, for re-fitting the cap curves. One line per event with
    // the actual per-container demand and the caps they were sized against. The readback is a
    // single D2H sync of the 5-word extraStorage (APC + nCells + nTriplets + nCellTracks).
    {
      uint32_t dumpNTracks, dumpNHitsInTracks, dumpNCells, dumpNTriplets, dumpNCellTracks;
      kernels.readbackAllCounts(queue, dumpNTracks, dumpNHitsInTracks, dumpNCells, dumpNTriplets, dumpNCellTracks);
      // The cell->cell edge count is the triplet count (Kernel_connect writes triplets into
      // deviceTriplets_ which is sized from nCells * avgCellsPerCell). The cell->track edge
      // count is nCellTracks (Kernel_connect writes cell->track pairs into deviceTracksCells_
      // which is sized from nCells * avgTracksPerCell).
      printf(
          "[CA Sizing] nHits=%u nCells=%u capCells=%u nTriplets=%u capTrips=%u "
          "nTracks=%u capTuples=%u nHitsInTracks=%u capHitCont=%u nCellTracks=%u capCellTrk=%u\n",
          static_cast<uint32_t>(trackingHits.size()),
          dumpNCells,
          pending.maxDoublets,
          dumpNTriplets,
          kernels.tripletsN(),
          dumpNTracks,
          nTracks,
          dumpNHitsInTracks,
          // The binding per-track-hit bound: the hitContainer content capacity, which is also the
          // output trackHits row count (both come from nHitRowsForTuples).
          caHitNtupletGenerator::nHitRowsForTuples(nTracks, m_params.algoParams_.avgHitsPerTrack_),
          dumpNCellTracks,
          kernels.tracksCellsN());
    }
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished building pixel tracks on GPU" << std::endl;
#endif

    return std::move(*pending.tracks);
  }

  template <typename TrackerTraits>
  void CAHitNtupletGenerator<TrackerTraits>::reportOverflows(std::string const& moduleLabel) const {
    if (!ovfHost_)
      return;  // stream never processed an event
    // Element-wise maximum of the mirror's two slots (the device counters only grow, so the
    // larger of the two snapshots is the later one). Only the six written words are examined;
    // kOvfWords - 6 trailing words are reserved and never touched by the sentinel.
    auto const* v = ovfHost_->data();
    constexpr uint32_t kUsed = 6u;
    uint32_t tot[kUsed];
    bool any = false;
    for (uint32_t i = 0; i < kUsed; ++i) {
      tot[i] = std::max(v[i], v[i + kOvfWords]);
      any = any || (tot[i] != 0u);
    }
    if (any) {
      edm::LogWarning("CAHitNtupletGenerator")
          << moduleLabel << ": observed container overflow in this stream: output was truncated"
          << " (tracks/doublets/hits were dropped): tuples=" << tot[0] << " cells=" << tot[1]
          << " cellToCell=" << tot[2] << " cellToTrack=" << tot[3] << " hitContent=" << tot[4]
          << " hitToTuple=" << tot[5]
          << ". The container capacities (maxNumberOfDoublets/maxNumberOfTuples/avg* ratios) need review.";
    }
  }

  template class CAHitNtupletGenerator<pixelTopology::Phase1>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2OT>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2OTStubs>;
  template class CAHitNtupletGenerator<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
