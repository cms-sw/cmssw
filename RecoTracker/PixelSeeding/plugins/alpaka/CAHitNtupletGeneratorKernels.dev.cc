#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"
#ifdef DUMP_GPU_TK_TUPLES
#include <mutex>
#endif

// #define NTUPLE_DEBUG
// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::launchKernels(const HitsConstView &hh,
                                                                  TkSoAView &tracks_view,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    // zero tuples
    cms::alpakatools::launchZero<Acc1D>(&(tracks_view.hitIndices()), queue);

    int32_t nhits = hh.metadata().size();

#ifdef NTUPLE_DEBUG
    std::cout << "start tuple building. N hits " << nhits << std::endl;
    if (nhits < 2)
      std::cout << "too few hits " << nhits << std::endl;
#endif

    //
    // applying conbinatoric cleaning such as fishbone at this stage is too expensive
    //

    const auto nthTot = 64;
    const auto stride = 4;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    const auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    ALPAKA_ASSERT_OFFLOAD(numberOfBlocks < 65536);
    ALPAKA_ASSERT_OFFLOAD(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_,
                        this->device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                        hh,
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        this->device_theCellNeighbors_.data(),
                        this->isOuterHitOfCell_.data(),
                        this->m_params.caParams_);

    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.earlyFishbone_) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto fishboneWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
      alpaka::exec<Acc2D>(queue,
                          fishboneWorkDiv,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          this->device_theCells_.data(),
                          this->device_nCells_.data(),
                          this->isOuterHitOfCell_.data(),
                          nhits,
                          false);
    }
    blockSize = 64;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        kernel_find_ntuplets<TrackerTraits>{},
                        hh,
                        tracks_view,
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        this->device_theCellTracks_.data(),
                        this->device_hitTuple_apc_,
                        this->m_params.caParams_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (this->m_params.doStats_)
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_mark_used<TrackerTraits>{},
                          this->device_theCells_.data(),
                          this->device_nCells_.data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(HitContainer::totbins(), blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        cms::alpakatools::finalizeBulk{},
                        this->device_hitTuple_apc_,
                        &tracks_view.hitIndices());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    alpaka::exec<Acc1D>(queue, workDiv1D, kernel_fillHitDetIndices<TrackerTraits>{}, tracks_view, hh);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
    alpaka::exec<Acc1D>(queue, workDiv1D, kernel_fillNLayers<TrackerTraits>{}, tracks_view, this->device_hitTuple_apc_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        kernel_earlyDuplicateRemover<TrackerTraits>{},
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        tracks_view,
                        this->m_params.dupPassThrough_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfTuples / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        kernel_countMultiplicity<TrackerTraits>{},
                        tracks_view,
                        this->device_tupleMultiplicity_.data());
    cms::alpakatools::launchFinalize<Acc1D>(this->device_tupleMultiplicity_.data(), queue);

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, kernel_fillMultiplicity<TrackerTraits>{}, tracks_view, this->device_tupleMultiplicity_.data());
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.lateFishbone_) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          this->device_theCells_.data(),
                          this->device_nCells_.data(),
                          this->isOuterHitOfCell_.data(),
                          nhits,
                          true);
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(const HitsConstView &hh, Queue &queue) {
    auto nhits = hh.metadata().size();

    using namespace caPixelDoublets;

    using CACell = CACellT<TrackerTraits>;
    using OuterHitOfCell = typename CACell::OuterHitOfCell;
    using CellNeighbors = typename CACell::CellNeighbors;
    using CellTracks = typename CACell::CellTracks;
    using OuterHitOfCellContainer = typename CACell::OuterHitOfCellContainer;

#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    // in principle we can use "nhits" to heuristically dimension the workspace...
    ALPAKA_ASSERT_OFFLOAD(this->device_isOuterHitOfCell_.data());

    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
        [] ALPAKA_FN_ACC(Acc1D const &acc,
                         OuterHitOfCell *isOuterHitOfCell,
                         OuterHitOfCellContainer *container,
                         int32_t const *offset) {
          // this code runs on the device
          isOuterHitOfCell->container = container;
          isOuterHitOfCell->offset = *offset;
        },
        this->isOuterHitOfCell_.data(),
        this->device_isOuterHitOfCell_.data(),
        &hh.offsetBPIX2());

    {
      int threadsPerBlock = 128;
      // at least one block!
      int blocks = std::max(1u, cms::alpakatools::divide_up_by(nhits, threadsPerBlock));
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
      
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          initDoublets<TrackerTraits>{},
                          this->isOuterHitOfCell_.data(),
                          nhits,
                          this->device_theCellNeighbors_.data(),
                          this->device_theCellNeighborsContainer_,
                          this->device_theCellTracks_.data(),
                          this->device_theCellTracksContainer_);
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (0 == nhits)
      return;  // protect against empty events

    // take all layer pairs into account
    auto nActualPairs = this->m_params.nPairs();

    const int stride = 4;
    const int threadsPerBlock = TrackerTraits::getDoubletsFromHistoMaxBlockSize / stride;
    int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
    const Vec2D blks{blocks, 1u};
    const Vec2D thrs{threadsPerBlock, stride};
    const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        getDoubletsFromHisto<TrackerTraits>{},
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        this->device_theCellNeighbors_.data(),
                        this->device_theCellTracks_.data(),
                        hh,
                        this->isOuterHitOfCell_.data(),
                        nActualPairs,
                        this->m_params.caParams_.maxNumberOfDoublets_,
                        this->m_params.cellCuts_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples(const HitsConstView &hh,
                                                                   TkSoAView &tracks_view,
                                                                   Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

    uint32_t nhits = hh.metadata().size();

    auto blockSize = 64;

    // classify tracks based on kinematics
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfQuadruplets / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, kernel_classifyTracks<TrackerTraits>{}, tracks_view, this->m_params.qualityCuts_);

    if (this->m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_fishboneCleaner<TrackerTraits>{},
                          this->device_theCells_.data(),
                          this->device_nCells_.data(),
                          tracks_view);
    }

    // mark duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        kernel_fastDuplicateRemover<TrackerTraits>{},
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        tracks_view,
                        this->m_params.dupPassThrough_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (this->m_params.doSharedHitCut_ || this->m_params.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfQuadruplets / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitToTuple_.data());  //CHECK

      cms::alpakatools::launchFinalize<Acc1D>(this->device_hitToTuple_.data(), queue);
      alpaka::exec<Acc1D>(
          queue, workDiv1D, kernel_fillHitInTracks<TrackerTraits>{}, tracks_view, this->device_hitToTuple_.data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif
    }

    if (this->m_params.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfQuadruplets / 4,
                                                      blockSize);  // TODO: Check if correct
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.minHitsForSharingCut_,
                          this->m_params.dupPassThrough_,
                          this->device_hitToTuple_.data());

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_sharedHitCleaner<TrackerTraits>{},
                          hh,
                          tracks_view,
                          this->m_params.minHitsForSharingCut_,
                          this->m_params.dupPassThrough_,
                          this->device_hitToTuple_.data());

      if (this->m_params.useSimpleTripletCleaner_) {
        numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple::capacity(), blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            kernel_simpleTripletCleaner<TrackerTraits>{},
                            tracks_view,
                            this->m_params.minHitsForSharingCut_,
                            this->m_params.dupPassThrough_,
                            this->device_hitToTuple_.data());
      } else {
        numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple::capacity(), blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            kernel_tripletCleaner<TrackerTraits>{},
                            tracks_view,
                            this->m_params.minHitsForSharingCut_,
                            this->m_params.dupPassThrough_,
                            this->device_hitToTuple_.data());
      }
#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif
    }

    if (this->m_params.doStats_) {
      numberOfBlocks =
          cms::alpakatools::divide_up_by(std::max(nhits, m_params.caParams_.maxNumberOfDoublets_), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_checkOverflows<TrackerTraits>{},
                          tracks_view,
                          this->device_tupleMultiplicity_.data(),
                          this->device_hitToTuple_.data(),
                          this->device_hitTuple_apc_,
                          this->device_theCells_.data(),
                          this->device_nCells_.data(),
                          this->device_theCellNeighbors_.data(),
                          this->device_theCellTracks_.data(),
                          this->isOuterHitOfCell_.data(),
                          nhits,
                          this->m_params.caParams_.maxNumberOfDoublets_,
                          this->counters_.data());
    }

    if (this->m_params.doStats_) {
      // counters (add flag???)
   
      numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple::capacity(), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_doStatsForHitInTracks<TrackerTraits>{},
                          this->device_hitToTuple_.data(),
                          this->counters_.data());

      numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfQuadruplets / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(
          queue, workDiv1D, kernel_doStatsForTracks<TrackerTraits>{}, tracks_view, this->counters_.data());
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    static std::mutex lock;
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1u, 32u);
    {
      std::lock_guard<std::mutex> guard(lock);
      ++iev;
      for (int k = 0; k < 20000; k += 500) {
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            kernel_print_found_ntuplets<TrackerTraits>{},
                            hh,
                            tracks_view,
                            this->device_hitToTuple_.data(),
                            k,
                            k + 500,
                            iev);
        alpaka::wait(queue);
      }
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          kernel_print_found_ntuplets<TrackerTraits>{},
                          hh,
                          tracks_view,
                          this->device_hitToTuple_.data(),
                          20000,
                          1000000,
                          iev);

      alpaka::wait(queue);
    }
#endif
  }
/*
template <typename TrackerTraits>
void CAHitNtupletGeneratorKernels<TrackerTraits>::printCounters() {
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1,1);
    alpaka::exec<Acc1D>(queue_,workDiv1D,kernel_printCounters{},this->counters_.data());
}
*/
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase1>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
