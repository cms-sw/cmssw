#ifndef RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernelsWrapper_h
#define RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernelsWrapper_h

#include <cstdint>
#include <optional>

#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationVectorHitStyleKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Manages device buffers and launches stub-formation kernels.
  class OTStubFormationVectorHitStyleKernelsWrapper {
  public:
    explicit OTStubFormationVectorHitStyleKernelsWrapper(Queue& queue);

    ~OTStubFormationVectorHitStyleKernelsWrapper() = default;

    void countStubs(Queue& queue,
                    ::reco::OTRecHitsConstView const& hits,
                    ::reco::OTHitModuleConstView const& moduleView,
                    ::reco::StackedModuleGeometryConstView const& geometry,
                    float const* maxWidthBarrelFlat,
                    float const* maxWidthBarrelTilted,
                    float const* maxWidthEndcap,
                    int32_t const* barrelFlatMaxCSDiff,
                    int32_t const* barrelTiltedMaxCSDiff,
                    int32_t const* endcapMaxCSDiff,
                    int32_t const* barrelFlatMaxCS,
                    int32_t const* barrelTiltedMaxCS,
                    int32_t const* endcapMaxCS,
                    int32_t const* barrelFlatMaxCSSum,
                    int32_t const* barrelTiltedMaxCSSum,
                    int32_t const* endcapMaxCSSum,
                    uint32_t* stubCounts,
                    uint32_t nModules);

    // Finalize stub offsets via device-side prefix scan
    void finalizeOffsets(Queue& queue, uint32_t* stubOffsets, uint32_t nModules);

    void formStubs(Queue& queue,
                   ::reco::OTRecHitsConstView const& hits,
                   ::reco::OTHitModuleConstView const& moduleView,
                   ::reco::StackedModuleGeometryConstView const& geometry,
                   float const* maxWidthBarrelFlat,
                   float const* maxWidthBarrelTilted,
                   float const* maxWidthEndcap,
                   int32_t const* barrelFlatMaxCSDiff,
                   int32_t const* barrelTiltedMaxCSDiff,
                   int32_t const* endcapMaxCSDiff,
                   int32_t const* barrelFlatMaxCS,
                   int32_t const* barrelTiltedMaxCS,
                   int32_t const* endcapMaxCS,
                   int32_t const* barrelFlatMaxCSSum,
                   int32_t const* barrelTiltedMaxCSSum,
                   int32_t const* endcapMaxCSSum,
                   uint32_t const* stubOffsets,
                   ::reco::StubsView stubs,
                   uint32_t nModules);

  private:
    // Prefix-scan atomic-counter workspace
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> prefixScanWorkspace_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernelsWrapper_h
