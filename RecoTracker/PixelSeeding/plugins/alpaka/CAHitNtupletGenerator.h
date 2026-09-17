#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h

#include <alpaka/alpaka.hpp>

#include <optional>
#include <memory>

#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"

#include "CACell.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CATripletDumpMacro.h"  // CA_TRIPLET_DUMP toggle for the #ifdef'd dump member/accessor below
#include "HelixFit.h"

namespace edm {
  class ParameterSetDescription;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletGenerator {
  public:
    using HitsView = ::reco::TrackingRecHitView;
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using HitsOnHost = ::reco::TrackingRecHitHost;

    using HitsOnDeviceRefProdVector = edm::RefProdVector<HitsOnDevice>;

    // Everything the CA build needs from the event's hit collections, assembled by the producer: the
    // topology's hit view and module-start view (see CAStructures.h) plus the two scalars.
    using HitsInput = caStructures::HitsInputT<TrackerTraits>;

    using TkSoADevice = reco::TracksSoACollection;
    using Quality = ::pixelTrack::Quality;

    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

    using CAGeometryOnDevice = reco::CAGeometrySoACollection;

  public:
    CAHitNtupletGenerator(const edm::ParameterSet& cfg);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

    // NOTE: beginJob and endJob were meant to be used
    // to fill the statistics. This is still not implemented in Alpaka
    // since we are missing the begin/endJob functionality for the Alpaka
    // producers.
    //
    // void beginJob();
    // void endJob();

    // The producer builds the CA ntuplets, runs the broken-line fit and selects high-purity tracks.
    // It does not consume the outer-tracker or stacked-sensor products.
    //
    // Acquire/produce split (stream::SynchronizingEDProducer): the producer calls beginTuplesAsync
    // in acquire() -- hit prep, doublets, CA kernels, and one async D2H of the tuple-multiplicity
    // per-N-bin offsets -- and finishTuplesAsync in produce() -- both fit passes (consuming the
    // landed offsets host-side, see HelixFit::setHostTupleMultiplicityOffsets), tuple
    // classification and the product. The framework's acquire->produce boundary waits for the copy
    // in its async callback instead of on a TBB thread. State crossing the boundary lives in
    // PendingTuples, a member of the producer.
    struct PendingTuples {
      // unique_ptr, not optional: the kernels object is not move-constructible (reference +
      // const members), and PendingTuples must move across the acquire->produce seam.
      std::unique_ptr<CAHitNtupletGeneratorKernels<TrackerTraits>> kernels;
      std::optional<TkSoADevice> tracks;
      // Pinned host mirror of the tuple-multiplicity per-N-bin offsets; filled by one async D2H
      // enqueued at the end of beginTuplesAsync. The framework seam guarantees it has landed
      // before finishTuplesAsync runs.
      std::optional<cms::alpakatools::host_buffer<uint32_t[]>> offsetsHost;
      // Pinned host mirror of the 5-word extraStorage counter block, enqueued async in
      // beginTuplesAsync when delayAllocations is on and landed by the same seam guarantee.
      // Word [0] is the tuple AtomicPairCounter; finishTuplesAsync decodes it to size the
      // hit->track storage without any host wait.
      std::optional<cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>> countsHost;
      uint32_t maxTuples = 0;
      uint32_t maxDoublets = 0;
      float bfield = 0.f;
      const float* rhoMapDevice = nullptr;
      // Normalized (Bz,Br) r-z field map (BLBFieldMap EventSetup condition, device-resident). Carried
      // across the acquire->produce seam beside the material map because the fit consumes them together
      // (see HelixFit::setBFieldMap); null is the scalar-field fallback.
      const float* bMapDevice = nullptr;
      bool built = false;  // false = early-out (too few hits): tracks holds the empty collection
    };

    PendingTuples beginTuplesAsync(HitsInput const& hits,
                                   CAGeometryOnDevice const& params_d,
                                   float bfield,
                                   uint32_t maxDoublets,
                                   uint32_t maxTuples,
                                   Queue& queue,
                                   const float* rhoMapDevice,
                                   const float* bMapDevice) const;

    TkSoADevice finishTuplesAsync(PendingTuples&& pending,
                                  HitsInput const& hits,
                                  CAGeometryOnDevice const& params_d,
                                  Queue& queue) const;

    // Always-on overflow surfacing, independent of doStats_: the capacity guards in the build
    // kernels truncate silently, so without this a shortfall leaves no trace in production.
    // Per-stream persistent device accumulator (kOvfWords words, layout in
    // Kernel_overflowSentinel; slots 0-5 in use, 6-7 reserved), armed into each event's kernels
    // object, plus a pinned host mirror refreshed by an async D2H enqueued after classifyTuples
    // (where the sentinel is launched). No wait: by endStream every event queue has drained, so
    // the mirror holds the final totals.
    //
    // The mirror carries two slots and each event writes the one its parity selects. Consecutive
    // events of a stream run on different queues, so their copies can be in flight at the same
    // time; alternating slots keeps two overlapping copies from writing the same bytes.
    // reportOverflows() takes the element-wise maximum of the two slots, which is exact because
    // the device counters only ever grow.
    // Mutable for the same reason as phiBinnerOut_: beginTuplesAsync/finishTuplesAsync are const.
    static constexpr uint32_t kOvfWords = 8u;
    mutable std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> ovfAccum_;
    mutable std::optional<cms::alpakatools::host_buffer<uint32_t[]>> ovfHost_;
    mutable uint32_t ovfSlot_ = 0;
    void reportOverflows(std::string const& moduleLabel) const;

#ifdef CA_TRIPLET_DUMP
    // Per-built-triplet training-dataset capture surfaced out of finishTuplesAsync: the kernels
    // object that owns the buffer lives in PendingTuples and is destroyed with it, so
    // finishTuplesAsync moves the kernels' device_tripletDump_ into here before returning; the
    // producer then moves it out and emplaces it as the 'Triplet' nano product. Mutable because
    // finishTuplesAsync is const. Empty/zero footprint when CA_TRIPLET_DUMP is off (the whole
    // member is compiled out).
    mutable std::optional<TripletDumpSoACollection> device_tripletDump_;
    std::optional<TripletDumpSoACollection>& tripletDumpBuffer() const { return device_tripletDump_; }
#endif

  private:
    Params m_params;
    // One-shot post-fit device dump of the first fitted tracks (HelixFit::setVerboseDump). Config: verboseBLFit.
    bool m_verboseBLDump;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
