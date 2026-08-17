#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/Phase2OTVectorHitsFormationKernels.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/Phase2OTVectorHitsFormationKernelsWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void Phase2OTVectorHitsFormationKernelsWrapper::countVectorHits(Queue& queue,
                                                                  ::reco::OTRecHitsConstView const& hits,
                                                                  ::reco::OTHitModuleConstView const& moduleView,
                                                                  ::reco::StackedModuleGeometryConstView const& geometry,
                                                                  float const* maxWidthBarrelFlat,
                                                                  float const* maxWidthBarrelTilted,
                                                                  float const* maxWidthEndcap,
                                                                  uint32_t* vectorHitCounts,
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
                        phase2OTVectorHitsFormation::CountVectorHitsKernel{},
                        hits,
                        moduleView,
                        geometry,
                        maxWidthBarrelFlat,
                        maxWidthBarrelTilted,
                        maxWidthEndcap,
                        vectorHitCounts,
                        nModules);
  }

  void Phase2OTVectorHitsFormationKernelsWrapper::finalizeOffsets(Queue& queue,
                                                                  uint32_t* vectorHitOffsets,
                                                                  uint32_t nModules) {
    // Atomic counter of the multi-block prefix scan
    auto prefixScanWorkspace = cms::alpakatools::make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, prefixScanWorkspace, 0);

    // In-place exclusive scan: counting kernel wrote to vectorHitOffsets[iModule + 1].
    auto nthreads = 1024u;
    auto nblocks = (nModules + 1 + nthreads - 1) / nthreads;
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nblocks, nthreads);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>(),
                        vectorHitOffsets,            // input (counts at offset positions)
                        vectorHitOffsets,            // output (in-place exclusive scan)
                        nModules + 1,                // size
                        nblocks,                     // number of blocks
                        prefixScanWorkspace.data(),  // atomic counter workspace
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
  }

  void Phase2OTVectorHitsFormationKernelsWrapper::formVectorHits(Queue& queue,
                                                                 ::reco::OTRecHitsConstView const& hits,
                                                                 ::reco::OTHitModuleConstView const& moduleView,
                                                                 ::reco::StackedModuleGeometryConstView const& geometry,
                                                                 float const* maxWidthBarrelFlat,
                                                                 float const* maxWidthBarrelTilted,
                                                                 float const* maxWidthEndcap,
                                                                 uint32_t const* vectorHitOffsets,
                                                                 ::reco::VectorHitsView vectorHits,
                                                                 uint32_t nModules) {
    // Same warp-cooperative work division as countVectorHits above.
    const uint32_t stride = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
    const uint32_t modulesPerBlock = 8;  // 8 modules * 32 threads = 256 threads/block
    auto numBlocks = cms::alpakatools::divide_up_by(nModules, modulesPerBlock);
    numBlocks = std::min<uint32_t>(numBlocks, 65535u);
    const Vec2D blocks{numBlocks, 1u};
    const Vec2D threads{modulesPerBlock, stride};
    auto workDiv = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

    alpaka::exec<Acc2D>(queue,
                        workDiv,
                        phase2OTVectorHitsFormation::FormVectorHitsKernel{},
                        hits,
                        moduleView,
                        geometry,
                        maxWidthBarrelFlat,
                        maxWidthBarrelTilted,
                        maxWidthEndcap,
                        vectorHitOffsets,
                        vectorHits,
                        nModules);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
