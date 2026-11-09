#ifndef RecoTracker_PixelSeeding_plugins_alpaka_Phase2OTVectorHitsFormationKernelsWrapper_h
#define RecoTracker_PixelSeeding_plugins_alpaka_Phase2OTVectorHitsFormationKernelsWrapper_h

#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Launches the vector-hit-formation kernels.
  class Phase2OTVectorHitsFormationKernelsWrapper {
  public:
    void countVectorHits(Queue& queue,
                         ::reco::OTRecHitsConstView const& hits,
                         ::reco::OTHitModuleConstView const& moduleView,
                         ::reco::StackedModuleGeometryConstView const& geometry,
                         float const* maxWidthBarrelFlat,
                         float const* maxWidthBarrelTilted,
                         float const* maxWidthEndcap,
                         uint32_t* vectorHitCounts,
                         uint32_t nModules);

    // Finalize vector hit offsets via device-side prefix scan
    void finalizeOffsets(Queue& queue, uint32_t* vectorHitOffsets, uint32_t nModules);

    void formVectorHits(Queue& queue,
                        ::reco::OTRecHitsConstView const& hits,
                        ::reco::OTHitModuleConstView const& moduleView,
                        ::reco::StackedModuleGeometryConstView const& geometry,
                        float const* maxWidthBarrelFlat,
                        float const* maxWidthBarrelTilted,
                        float const* maxWidthEndcap,
                        uint32_t const* vectorHitOffsets,
                        ::reco::VectorHitsView vectorHits,
                        uint32_t nModules);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_Phase2OTVectorHitsFormationKernelsWrapper_h
