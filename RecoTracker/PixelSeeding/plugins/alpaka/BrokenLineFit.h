#ifndef RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFit_h
#define RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFit_h

// HelixFit::launchBrokenLineKernels. The kernels are in BrokenLineFitKernels.h.

// #define GPU_DEBUG
#ifdef GPU_DEBUG
#include <iostream>
#endif

#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchBrokenLineKernels(const HitsMultiView& hv,
                                                        const ::reco::CAModulesConstView& cm,
                                                        Queue& queue) {
    ALPAKA_ASSERT_ACC(tuples_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting HelixFit<TrackerTraits>::launchBrokenLineKernels" << std::endl;
#endif

    //  Fit internals
    auto tkidDevice =
        cms::alpakatools::make_device_buffer<typename caStructures::tindex_type[]>(queue, maxNumberOfConcurrentFits_);
    // The buffers of the coarse-coordinate rule and of the second sensor of a vector hit, for the vector-hits topology
    // with fitCorrections (BLCoarseRuleBuffers).
    BLLaneBuffers<TrackerTraits> buffers(queue, BLCoarseRuleBuffers<TrackerTraits>::kActive && fitCfg_.fitCorrections);

    // The top bin [maxHitsOnTrackForFullFit, maxHitsOnTrack - 1] must not be empty.
    static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);
    const WorkDiv1D workDivFused = cms::alpakatools::make_workdiv<Acc1D>(kFitFusedBlocks<TrackerTraits>, kFitBlock);

    // Each round fits up to maxNumberOfConcurrentFits_ tuples.
    ALPAKA_ASSERT_ACC(hostTupleMultiplicityOffsets_);
    const uint32_t nRounds = cms::alpakatools::divide_up_by(
        fittedPopulation<TrackerTraits>(hostTupleMultiplicityOffsets_), maxNumberOfConcurrentFits_);
    for (uint32_t round = 0; round < nRounds; ++round) {
      alpaka::exec<Acc1D>(queue,
                          workDivFused,
                          Kernel_BLFastFitFused<TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          hv,
                          cm,
                          tkidDevice.data(),
                          buffers.hits.data(),
                          buffers.hitsGe.data(),
                          buffers.fastFit.data(),
                          round,
                          buffers.coarseRuleScratch(),
                          partnerRecHits_);
      alpaka::exec<Acc1D>(queue,
                          workDivFused,
                          Kernel_BLFitFused<TrackerTraits>{fitCfg_},
                          tupleMultiplicity_,
                          double(bField_),
                          outputSoa_,
                          tkidDevice.data(),
                          buffers.hits.data(),
                          buffers.hitsGe.data(),
                          buffers.fastFit.data(),
                          buffers.scratch.data(),
                          round);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFit_h
