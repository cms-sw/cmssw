#include "RecoTracker/PixelSeeding/plugins/CAHitNtupletGeneratorKernelsImpl.h"

#include <mutex>

namespace {
  // cuda atomics are NOT atomics on CPU so protect stat update with a mutex
  // waiting for a more general solution (incuding multiple devices) to be proposed and implemented
  std::mutex lock_stat;
}  // namespace

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsCPU<TrackerTraits>::printCounters(Counters const *counters) {
  caHitNtupletGeneratorKernels::kernel_printCounters(counters);
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsCPU<TrackerTraits>::buildDoublets(const HitsConstView &hh,
                                                                   int32_t offsetBPIX2,
                                                                   cudaStream_t stream) {
  using namespace gpuPixelDoublets;

  using GPUCACell = GPUCACellT<TrackerTraits>;
  using OuterHitOfCell = typename GPUCACell::OuterHitOfCell;
  using CellNeighbors = typename GPUCACell::CellNeighbors;
  using CellTracks = typename GPUCACell::CellTracks;
  using OuterHitOfCellContainer = typename GPUCACell::OuterHitOfCellContainer;

  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits. BPIX2 offset is " << offsetBPIX2 << std::endl;
#endif

  // use "nhits" to heuristically dimension the workspace

  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //this->device_isOuterHitOfCell_ = Traits::template make_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits), stream);
  this->device_isOuterHitOfCell_ = std::make_unique<OuterHitOfCellContainer[]>(std::max(1U, nhits));
  assert(this->device_isOuterHitOfCell_.get());
  this->isOuterHitOfCell_ = OuterHitOfCell{this->device_isOuterHitOfCell_.get(), offsetBPIX2};

  auto cellStorageSize = TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors) +
                         TrackerTraits::maxNumOfActiveDoublets * sizeof(CellTracks);
  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //cellStorage_ = Traits::template make_unique<unsigned char[]>(cellStorageSize, stream);
  this->cellStorage_ = std::make_unique<unsigned char[]>(cellStorageSize);
  this->device_theCellNeighborsContainer_ = (CellNeighbors *)this->cellStorage_.get();
  this->device_theCellTracksContainer_ =
      (CellTracks *)(this->cellStorage_.get() + TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors));

  initDoublets<TrackerTraits>(this->isOuterHitOfCell_,
                              nhits,
                              this->device_theCellNeighbors_.get(),
                              this->device_theCellNeighborsContainer_,
                              this->device_theCellTracks_.get(),
                              this->device_theCellTracksContainer_);

  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //this->device_theCells_ = Traits::template make_unique<GPUCACell[]>(this->params_.cellCuts_.maxNumberOfDoublets_, stream);
  this->device_theCells_ = std::make_unique<GPUCACell[]>(this->params_.cellCuts_.maxNumberOfDoublets_);
  if (0 == nhits)
    return;  // protect against empty events

  // take all layer pairs into account
  auto nActualPairs = this->params_.nPairs();

  assert(nActualPairs <= TrackerTraits::nPairs);

  getDoubletsFromHisto<TrackerTraits>(this->device_theCells_.get(),
                                      this->device_nCells_,
                                      this->device_theCellNeighbors_.get(),
                                      this->device_theCellTracks_.get(),
                                      hh,
                                      this->isOuterHitOfCell_,
                                      nActualPairs,
                                      this->params_.cellCuts_);
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsCPU<TrackerTraits>::launchKernels(const HitsConstView &hh,
                                                                   TkSoAView &tracks_view,
                                                                   cudaStream_t cudaStream) {
  using namespace caHitNtupletGeneratorKernels;

  // zero tuples
  cms::cuda::launchZero(&tracks_view.hitIndices(), cudaStream);

  uint32_t nhits = hh.metadata().size();

#ifdef NTUPLE_DEBUG
  std::cout << "start tuple building. N hits " << nhits << std::endl;
  if (nhits < 2)
    std::cout << "too few hits " << nhits << std::endl;
#endif

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  kernel_connect<TrackerTraits>(this->device_hitTuple_apc_,
                                this->device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                                hh,
                                this->device_theCells_.get(),
                                this->device_nCells_,
                                this->device_theCellNeighbors_.get(),
                                this->isOuterHitOfCell_,
                                this->params_.caParams_);

  if (nhits > 1 && this->params_.earlyFishbone_) {
    gpuPixelDoublets::fishbone<TrackerTraits>(
        hh, this->device_theCells_.get(), this->device_nCells_, this->isOuterHitOfCell_, nhits, false);
  }

  kernel_find_ntuplets<TrackerTraits>(hh,
                                      tracks_view,
                                      this->device_theCells_.get(),
                                      this->device_nCells_,
                                      this->device_theCellTracks_.get(),
                                      this->device_hitTuple_apc_,
                                      this->params_.caParams_);
  if (this->params_.doStats_)
    kernel_mark_used(this->device_theCells_.get(), this->device_nCells_);

  cms::cuda::finalizeBulk(this->device_hitTuple_apc_, &tracks_view.hitIndices());

  kernel_fillHitDetIndices<TrackerTraits>(tracks_view, hh);
  kernel_fillNLayers<TrackerTraits>(tracks_view, this->device_hitTuple_apc_);

  // remove duplicates (tracks that share a doublet)
  kernel_earlyDuplicateRemover<TrackerTraits>(
      this->device_theCells_.get(), this->device_nCells_, tracks_view, this->params_.dupPassThrough_);

  kernel_countMultiplicity<TrackerTraits>(tracks_view, this->device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(this->device_tupleMultiplicity_.get(), cudaStream);
  kernel_fillMultiplicity<TrackerTraits>(tracks_view, this->device_tupleMultiplicity_.get());

  if (nhits > 1 && this->params_.lateFishbone_) {
    gpuPixelDoublets::fishbone<TrackerTraits>(
        hh, this->device_theCells_.get(), this->device_nCells_, this->isOuterHitOfCell_, nhits, true);
  }
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsCPU<TrackerTraits>::classifyTuples(const HitsConstView &hh,
                                                                    TkSoAView &tracks_view,
                                                                    cudaStream_t cudaStream) {
  using namespace caHitNtupletGeneratorKernels;

  int32_t nhits = hh.metadata().size();

  // classify tracks based on kinematics
  kernel_classifyTracks<TrackerTraits>(tracks_view, this->params_.qualityCuts_);
  if (this->params_.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    kernel_fishboneCleaner<TrackerTraits>(this->device_theCells_.get(), this->device_nCells_, tracks_view);
  }

  // remove duplicates (tracks that share a doublet)
  kernel_fastDuplicateRemover<TrackerTraits>(
      this->device_theCells_.get(), this->device_nCells_, tracks_view, this->params_.dupPassThrough_);

  // fill hit->track "map"
  if (this->params_.doSharedHitCut_ || this->params_.doStats_) {
    kernel_countHitInTracks<TrackerTraits>(tracks_view, this->device_hitToTuple_.get());
    cms::cuda::launchFinalize(this->hitToTupleView_, cudaStream);
    kernel_fillHitInTracks<TrackerTraits>(tracks_view, this->device_hitToTuple_.get());
  }

  // remove duplicates (tracks that share at least one hit)
  if (this->params_.doSharedHitCut_) {
    kernel_rejectDuplicate<TrackerTraits>(tracks_view,
                                          this->params_.minHitsForSharingCut_,
                                          this->params_.dupPassThrough_,
                                          this->device_hitToTuple_.get());

    kernel_sharedHitCleaner<TrackerTraits>(hh,
                                           tracks_view,
                                           this->params_.minHitsForSharingCut_,
                                           this->params_.dupPassThrough_,
                                           this->device_hitToTuple_.get());
    if (this->params_.useSimpleTripletCleaner_) {
      kernel_simpleTripletCleaner<TrackerTraits>(tracks_view,
                                                 this->params_.minHitsForSharingCut_,
                                                 this->params_.dupPassThrough_,
                                                 this->device_hitToTuple_.get());
    } else {
      kernel_tripletCleaner<TrackerTraits>(tracks_view,
                                           this->params_.minHitsForSharingCut_,
                                           this->params_.dupPassThrough_,
                                           this->device_hitToTuple_.get());
    }
  }

  if (this->params_.doStats_) {
    std::lock_guard guard(lock_stat);
    kernel_checkOverflows<TrackerTraits>(tracks_view,
                                         this->device_tupleMultiplicity_.get(),
                                         this->device_hitToTuple_.get(),
                                         this->device_hitTuple_apc_,
                                         this->device_theCells_.get(),
                                         this->device_nCells_,
                                         this->device_theCellNeighbors_.get(),
                                         this->device_theCellTracks_.get(),
                                         this->isOuterHitOfCell_,
                                         nhits,
                                         this->params_.cellCuts_.maxNumberOfDoublets_,
                                         this->counters_);
  }

  if (this->params_.doStats_) {
    // counters (add flag???)
    std::lock_guard guard(lock_stat);
    kernel_doStatsForHitInTracks<TrackerTraits>(this->device_hitToTuple_.get(), this->counters_);
    kernel_doStatsForTracks<TrackerTraits>(tracks_view, this->counters_);
  }

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  static std::mutex lock;
  {
    std::lock_guard<std::mutex> guard(lock);
    ++iev;
    kernel_print_found_ntuplets<TrackerTraits>(hh, tracks_view, this->device_hitToTuple_.get(), 0, 1000000, iev);
  }
#endif
}

template class CAHitNtupletGeneratorKernelsCPU<pixelTopology::Phase1>;
template class CAHitNtupletGeneratorKernelsCPU<pixelTopology::Phase2>;
