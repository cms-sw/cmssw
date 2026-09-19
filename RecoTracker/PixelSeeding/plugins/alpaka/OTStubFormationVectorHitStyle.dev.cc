#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationVectorHitStyleKernels.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationVectorHitStyleKernelsWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  OTStubFormationVectorHitStyleKernelsWrapper::OTStubFormationVectorHitStyleKernelsWrapper(Queue& queue)
      : prefixScanWorkspace_(cms::alpakatools::make_device_buffer<int32_t[]>(queue, 1)) {
    alpaka::memset(queue, *prefixScanWorkspace_, 0);
  }

  void OTStubFormationVectorHitStyleKernelsWrapper::countStubs(Queue& queue,
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
                                                               uint32_t nModules) {
    // Warp-cooperative: one module per Y-thread, one warp per module.
    // gridDim.y capped at 65535; uniform_elements_y strides remaining modules.
    const uint32_t stride = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
    const uint32_t modulesPerBlock = 8;  // 8 modules * 32 threads = 256 threads/block
    auto numBlocks = cms::alpakatools::divide_up_by(nModules, modulesPerBlock);
    numBlocks = std::min<uint32_t>(numBlocks, 65535u);
    const Vec2D blocks{numBlocks, 1u};
    const Vec2D threads{modulesPerBlock, stride};
    auto workDiv = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

    alpaka::exec<Acc2D>(queue,
                        workDiv,
                        otStubFormationVectorHitStyle::CountStubsVectorHitStyleKernel{},
                        hits,
                        moduleView,
                        geometry,
                        maxWidthBarrelFlat,
                        maxWidthBarrelTilted,
                        maxWidthEndcap,
                        barrelFlatMaxCSDiff,
                        barrelTiltedMaxCSDiff,
                        endcapMaxCSDiff,
                        barrelFlatMaxCS,
                        barrelTiltedMaxCS,
                        endcapMaxCS,
                        barrelFlatMaxCSSum,
                        barrelTiltedMaxCSSum,
                        endcapMaxCSSum,
                        stubCounts,
                        nModules);
  }

  void OTStubFormationVectorHitStyleKernelsWrapper::finalizeOffsets(Queue& queue,
                                                                    uint32_t* stubOffsets,
                                                                    uint32_t nModules) {
    alpaka::memset(queue, *prefixScanWorkspace_, 0);

    // In-place exclusive scan: counting kernel wrote to stubOffsets[iModule + 1].
    auto nthreads = 1024u;
    auto nblocks = (nModules + 1 + nthreads - 1) / nthreads;
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nblocks, nthreads);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>(),
                        stubOffsets,                   // input (counts at offset positions)
                        stubOffsets,                   // output (in-place exclusive scan)
                        nModules + 1,                  // size
                        nblocks,                       // number of blocks
                        prefixScanWorkspace_->data(),  // atomic counter workspace
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
  }

  void OTStubFormationVectorHitStyleKernelsWrapper::formStubs(Queue& queue,
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
                                                              uint32_t nModules) {
    // Same warp-cooperative work division as countStubs above.
    const uint32_t stride = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
    const uint32_t modulesPerBlock = 8;  // 8 modules * 32 threads = 256 threads/block
    auto numBlocks = cms::alpakatools::divide_up_by(nModules, modulesPerBlock);
    numBlocks = std::min<uint32_t>(numBlocks, 65535u);
    const Vec2D blocks{numBlocks, 1u};
    const Vec2D threads{modulesPerBlock, stride};
    auto workDiv = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

    alpaka::exec<Acc2D>(queue,
                        workDiv,
                        otStubFormationVectorHitStyle::FormStubsVectorHitStyleKernel{},
                        hits,
                        moduleView,
                        geometry,
                        maxWidthBarrelFlat,
                        maxWidthBarrelTilted,
                        maxWidthEndcap,
                        barrelFlatMaxCSDiff,
                        barrelTiltedMaxCSDiff,
                        endcapMaxCSDiff,
                        barrelFlatMaxCS,
                        barrelTiltedMaxCS,
                        endcapMaxCS,
                        barrelFlatMaxCSSum,
                        barrelTiltedMaxCSSum,
                        endcapMaxCSSum,
                        stubOffsets,
                        stubs,
                        nModules);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
