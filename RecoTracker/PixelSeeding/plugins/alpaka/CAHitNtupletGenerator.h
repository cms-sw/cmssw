#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

#include "CACell.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "HelixFit.h"

namespace edm {
  class ParameterSetDescription;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletGenerator {
  public:
    using HitsView = TrackingRecHitSoAView<TrackerTraits>;
    using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackingRecHitsSoACollection<TrackerTraits>;
    using HitsOnHost = TrackingRecHitHost<TrackerTraits>;
    using hindex_type = typename TrackingRecHitSoA<TrackerTraits>::hindex_type;

    using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
    using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;
    using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

    using CACell = CACellT<TrackerTraits>;
    using TkSoAHost = TracksHost<TrackerTraits>;
    using TkSoADevice = TracksSoACollection<TrackerTraits>;
    using HitContainer = typename reco::TrackSoA<TrackerTraits>::HitContainer;
    using Tuple = HitContainer;

    using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
    using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

    using Quality = ::pixelTrack::Quality;

    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

    //using ParamsOnDevice = pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::base_traits_t<TrackerTraits>>;
    using FrameOnDevice = FrameSoACollection;
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

    TkSoADevice makeTuplesAsync(HitsOnDevice const& hits_d,
                                FrameOnDevice const& frame_d,
                                //ParamsOnDevice const* cpeParams,
                                float bfield,
                                Queue& queue) const;

  private:
    void buildDoublets(const HitsConstView& hh, Queue& queue) const;

    void hitNtuplets(const HitsConstView& hh, const edm::EventSetup& es, bool useRiemannFit, Queue& queue);

    void launchKernels(const HitsConstView& hh, bool useRiemannFit, Queue& queue) const;

    Params m_params;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
