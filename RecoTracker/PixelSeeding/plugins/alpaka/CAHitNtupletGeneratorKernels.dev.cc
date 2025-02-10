// C++ headers
#ifdef DUMP_GPU_TK_TUPLES
#include <mutex>
#endif

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// local headers
#include "CAFishbone.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

//#define GPU_DEBUG
//#define NTUPLE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  CAHitNtupletGeneratorKernels<TrackerTraits>::CAHitNtupletGeneratorKernels(Params const &params,
                                                                            uint32_t nhits,
                                                                            uint32_t offsetBPIX2,
                                                                            Queue &queue)
      : m_params(params),
        //////////////////////////////////////////////////////////
        // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
        //////////////////////////////////////////////////////////
        counters_{cms::alpakatools::make_device_buffer<Counters>(queue)},

        // workspace
        device_hitToTuple_{cms::alpakatools::make_device_buffer<HitToTuple>(queue)},
        device_hitToTupleStorage_{
            cms::alpakatools::make_device_buffer<typename HitToTuple::Counter[]>(queue, nhits + 1)},
        device_tupleMultiplicity_{cms::alpakatools::make_device_buffer<TupleMultiplicity>(queue)},

        // NB: In legacy, device_theCells_ and device_isOuterHitOfCell_ were allocated inside buildDoublets
        device_theCells_{
            cms::alpakatools::make_device_buffer<CACell[]>(queue, m_params.caParams_.maxNumberOfDoublets_)},
        // in principle we can use "nhits" to heuristically dimension the workspace...
        device_isOuterHitOfCell_{
            cms::alpakatools::make_device_buffer<OuterHitOfCellContainer[]>(queue, std::max(1u, nhits - offsetBPIX2))},
        isOuterHitOfCell_{cms::alpakatools::make_device_buffer<OuterHitOfCell>(queue)},

        device_theCellNeighbors_{cms::alpakatools::make_device_buffer<CellNeighborsVector>(queue)},
        device_theCellTracks_{cms::alpakatools::make_device_buffer<CellTracksVector>(queue)},
        // NB: In legacy, cellStorage_ was allocated inside buildDoublets
        cellStorage_{cms::alpakatools::make_device_buffer<unsigned char[]>(
            queue,
            TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors) +
                TrackerTraits::maxNumOfActiveDoublets * sizeof(CellTracks))},
        device_cellCuts_{cms::alpakatools::make_device_buffer<CellCuts>(queue)},
        device_theCellNeighborsContainer_{reinterpret_cast<CellNeighbors *>(cellStorage_.data())},
        device_theCellTracksContainer_{reinterpret_cast<CellTracks *>(
            cellStorage_.data() + TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors))},

        // NB: In legacy, device_storage_ was allocated inside allocateOnGPU
        device_storage_{
            cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 3u)},
        device_hitTuple_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_storage_.data())},
        device_hitToTuple_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_storage_.data() + 1)},
        device_nCells_{
            cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_storage_.data() + 2))} {
#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building. N hits " << nhits << std::endl;
#endif

    alpaka::memset(queue, counters_, 0);
    alpaka::memset(queue, device_nCells_, 0);
    alpaka::memset(queue, cellStorage_, 0);

    auto cellCuts_h = cms::alpakatools::make_host_view(m_params.cellCuts_);
    alpaka::memcpy(queue, device_cellCuts_, cellCuts_h);

    [[maybe_unused]] TupleMultiplicity *tupleMultiplicityDeviceData = device_tupleMultiplicity_.data();
    using TM = cms::alpakatools::OneToManyAssocRandomAccess<typename TrackerTraits::tindex_type,
                                                            TrackerTraits::maxHitsOnTrack + 1,
                                                            TrackerTraits::maxNumberOfTuples>;
    TM *tm = device_tupleMultiplicity_.data();
    TM::template launchZero<Acc1D>(tm, queue);
    TupleMultiplicity::template launchZero<Acc1D>(tupleMultiplicityDeviceData, queue);

    device_hitToTupleView_.assoc = device_hitToTuple_.data();
    device_hitToTupleView_.offStorage = device_hitToTupleStorage_.data();
    device_hitToTupleView_.offSize = nhits + 1;

    HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);
#ifdef GPU_DEBUG
    std::cout << "Allocations for CAHitNtupletGeneratorKernels: done!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::launchKernels(const HitsConstView &hh,
                                                                  uint32_t offsetBPIX2,
                                                                  TkSoAView &tracks_view,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    // zero tuples
    HitContainer::template launchZero<Acc1D>(&(tracks_view.hitIndices()), queue);

    uint32_t nhits = hh.metadata().size();

#ifdef NTUPLE_DEBUG
    std::cout << "start tuple building. N hits " << nhits << std::endl;
    if (nhits < 2)
      std::cout << "too few hits " << nhits << std::endl;
#endif

    //
    // applying combinatoric cleaning such as fishbone at this stage is too expensive
    //

    const auto nthTot = 64;
    const auto stride = 4;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    const auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        Kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_,
                        this->device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                        hh,
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        this->device_theCellNeighbors_.data(),
                        this->isOuterHitOfCell_.data(),
                        this->m_params.caParams_);

    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.earlyFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
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
                        Kernel_find_ntuplets<TrackerTraits>{},
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
                          Kernel_mark_used<TrackerTraits>{},
                          this->device_theCells_.data(),
                          this->device_nCells_.data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(HitContainer{}.totOnes(), blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(
        queue, workDiv1D, typename HitContainer::finalizeBulk{}, this->device_hitTuple_apc_, &tracks_view.hitIndices());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_fillHitDetIndices<TrackerTraits>{}, tracks_view, hh);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_fillNLayers<TrackerTraits>{}, tracks_view, this->device_hitTuple_apc_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_earlyDuplicateRemover<TrackerTraits>{},
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
                        Kernel_countMultiplicity<TrackerTraits>{},
                        tracks_view,
                        this->device_tupleMultiplicity_.data());
    TupleMultiplicity::template launchFinalize<Acc1D>(this->device_tupleMultiplicity_.data(), queue);

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, Kernel_fillMultiplicity<TrackerTraits>{}, tracks_view, this->device_tupleMultiplicity_.data());
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.lateFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
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
  void CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(const HitsConstView &hh,
                                                                  uint32_t offsetBPIX2,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using CACell = CACellT<TrackerTraits>;
    using OuterHitOfCell = typename CACell::OuterHitOfCell;
    using CellNeighbors = typename CACell::CellNeighbors;
    using CellTracks = typename CACell::CellTracks;
    using OuterHitOfCellContainer = typename CACell::OuterHitOfCellContainer;

    auto nhits = hh.metadata().size();
#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    // in principle we can use "nhits" to heuristically dimension the workspace...
    ALPAKA_ASSERT_ACC(this->device_isOuterHitOfCell_.data());

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
      int blocks = std::max(1u, cms::alpakatools::divide_up_by(nhits - offsetBPIX2, threadsPerBlock));
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          InitDoublets<TrackerTraits>{},
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
                        GetDoubletsFromHisto<TrackerTraits>{},
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
        queue, workDiv1D, Kernel_classifyTracks<TrackerTraits>{}, tracks_view, this->m_params.qualityCuts_);

    if (this->m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fishboneCleaner<TrackerTraits>{},
                          this->device_theCells_.data(),
                          this->device_nCells_.data(),
                          tracks_view);
    }

    // mark duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.caParams_.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fastDuplicateRemover<TrackerTraits>{},
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
                          Kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitToTuple_.data());  //CHECK

      HitToTuple::template launchFinalize<Acc1D>(this->device_hitToTupleView_, queue);
      alpaka::exec<Acc1D>(
          queue, workDiv1D, Kernel_fillHitInTracks<TrackerTraits>{}, tracks_view, this->device_hitToTuple_.data());
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
                          Kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.minHitsForSharingCut_,
                          this->m_params.dupPassThrough_,
                          this->device_hitToTuple_.data());

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_sharedHitCleaner<TrackerTraits>{},
                          hh,
                          tracks_view,
                          this->m_params.minHitsForSharingCut_,
                          this->m_params.dupPassThrough_,
                          this->device_hitToTuple_.data());

      if (this->m_params.useSimpleTripletCleaner_) {
        // (typename HitToTuple{}::capacity(),
        numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple{}.capacity(), blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_simpleTripletCleaner<TrackerTraits>{},
                            tracks_view,
                            this->m_params.minHitsForSharingCut_,
                            this->m_params.dupPassThrough_,
                            this->device_hitToTuple_.data());
      } else {
        numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple{}.capacity(), blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_tripletCleaner<TrackerTraits>{},
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
                          Kernel_checkOverflows<TrackerTraits>{},
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

      numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple{}.capacity(), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForHitInTracks<TrackerTraits>{},
                          this->device_hitToTuple_.data(),
                          this->counters_.data());

      numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfQuadruplets / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(
          queue, workDiv1D, Kernel_doStatsForTracks<TrackerTraits>{}, tracks_view, this->counters_.data());
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
                            Kernel_print_found_ntuplets<TrackerTraits>{},
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
                          Kernel_print_found_ntuplets<TrackerTraits>{},
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

  /* This will make sense when we will be able to run this once per job in Alpaka

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::printCounters() {
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1,1);
    alpaka::exec<Acc1D>(queue_, workDiv1D, Kernel_printCounters{}, this->counters_.data());
  }
  */

  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase1>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
