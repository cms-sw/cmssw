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

#define GPU_DEBUG
#define NTUPLE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  CAHitNtupletGeneratorKernels<TrackerTraits>::CAHitNtupletGeneratorKernels(Params const &params,
                                                                            uint32_t nHits,
                                                                            uint32_t offsetBPIX2,
                                                                            uint16_t nLayers,
                                                                            Queue &queue)
      : m_params(params),
        //////////////////////////////////////////////////////////
        // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
        //////////////////////////////////////////////////////////
        counters_{cms::alpakatools::make_device_buffer<Counters>(queue)},

        // Hits -> Track
        device_hitToTuple_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_hitToTupleStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHits * m_params.algoParams_.avgHitsPerTrack_)},
        device_hitToTupleOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nHits + 1)},
        
        // (Outer) Hits-> Cells
        device_hitToCell_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_hitToCellStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHits * TrackerTraits::maxCellsPerHit)},
        device_hitToCellOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nHits + 1)},

        // Hits
        device_hitPhiHist_{cms::alpakatools::make_device_buffer<PhiBinner>(queue)},
        device_phiBinnerStorage_{cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nHits)},
        device_layerStarts_{cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nLayers + 1)}, 

        // Cell -> Neighbor Cells
        device_cellToNeighbors_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_cellToNeighborsStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, m_params.algoParams_.maxNumberOfDoublets_ * TrackerTraits::maxCellNeighbors)},
        device_cellToNeighborsOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, m_params.algoParams_.maxNumberOfDoublets_ + 1)},

        // Tracks -> Hits
        device_hitContainer_{cms::alpakatools::make_device_buffer<SequentialContainer>(queue)},
        device_hitContainerStorage_{cms::alpakatools::make_device_buffer<SequentialContainerStorage[]>(queue, m_params.algoParams_.avgHitsPerTrack_ * m_params.algoParams_.maxNumberOfTuples_)},
        device_hitContainerOffsets_{cms::alpakatools::make_device_buffer<SequentialContainerOffsets[]>(queue, m_params.algoParams_.maxNumberOfTuples_ + 1)},     
        
        // No.Hits -> Track (Multiplicity)
        device_tupleMultiplicity_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_tupleMultiplicityStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, m_params.algoParams_.maxNumberOfTuples_)},
        device_tupleMultiplicityOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, TrackerTraits::maxHitsOnTrack + 1)},
        
        // NB: In legacy, device_theCells_ and device_isOuterHitOfCell_ were allocated inside buildDoublets
        device_theCells_{
            cms::alpakatools::make_device_buffer<CACell[]>(queue, m_params.algoParams_.maxNumberOfDoublets_)},
        device_simpleCells_{
            cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, m_params.algoParams_.maxNumberOfDoublets_)},
        // in principle we can use "nhits" to heuristically dimension the workspace...
        device_isOuterHitOfCell_{
            cms::alpakatools::make_device_buffer<OuterHitOfCellContainer[]>(queue, std::max(1, int(nHits - offsetBPIX2)))},
        isOuterHitOfCell_{cms::alpakatools::make_device_buffer<OuterHitOfCell>(queue)},

        device_theCellNeighbors_{cms::alpakatools::make_device_buffer<CellNeighborsVector>(queue)},
        device_theCellTracks_{cms::alpakatools::make_device_buffer<CellTracksVector>(queue)},
        cellStorage_{cms::alpakatools::make_device_buffer<unsigned char[]>(
            queue,
            TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors) +
                TrackerTraits::maxNumOfActiveDoublets * sizeof(CellTracks))},
        device_theCellNeighborsContainer_{reinterpret_cast<CellNeighbors *>(cellStorage_.data())},
        device_theCellTracksContainer_{reinterpret_cast<CellTracks *>(
            cellStorage_.data() + TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors))},

        // NB: In legacy, device_storage_ was allocated inside allocateOnGPU
        device_storage_{
            cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 3u)},
        device_hitTuple_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_storage_.data())},
        // device_hitToTuple_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_storage_.data() + 1)},
        device_nCells_{
            cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_storage_.data() + 1))},
        device_nTriplets_{
            cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_storage_.data() + 2))}
        {
#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building. N hits " << nHits << std::endl;
#endif

    alpaka::memset(queue, counters_, 0);
    alpaka::memset(queue, device_nCells_, 0);
    alpaka::memset(queue, cellStorage_, 0);
    alpaka::memset(queue, device_nTriplets_,0);

    // Hits -> Track
    device_hitToTupleView_.assoc = device_hitToTuple_.data();
    device_hitToTupleView_.contentStorage = device_hitToTupleStorage_.data();
    device_hitToTupleView_.offStorage = device_hitToTupleOffsets_.data();
    device_hitToTupleView_.contentSize = alpaka::getExtentProduct(device_hitToTupleStorage_); 
    device_hitToTupleView_.offSize = alpaka::getExtentProduct(device_hitToTupleOffsets_);

    GenericContainer::template launchZero<Acc1D>(device_hitToTupleView_, queue);

    // (Outer) Hits-> Cells
    device_hitToCellView_.assoc = device_hitToCell_.data();
    device_hitToCellView_.contentStorage = device_hitToCellStorage_.data();
    device_hitToCellView_.offStorage = device_hitToCellOffsets_.data();
    device_hitToCellView_.contentSize = alpaka::getExtentProduct(device_hitToCellStorage_); 
    device_hitToCellView_.offSize = alpaka::getExtentProduct(device_hitToCellOffsets_);
    std::cout << "device_hitToCellView_" << device_hitToCellView_.contentSize << " - " << device_hitToCellView_.offSize << std::endl;

    GenericContainer::template launchZero<Acc1D>(device_hitToCellView_, queue);

    // Hits
    device_hitPhiView_.assoc = device_hitPhiHist_.data();
    device_hitPhiView_.offSize = -1;
    device_hitPhiView_.offStorage = nullptr;
    device_hitPhiView_.contentSize = nHits;
    device_hitPhiView_.contentStorage = device_phiBinnerStorage_.data();

    // Cells-> Neighbor Cells
    device_cellToNeighborsView_.assoc = device_cellToNeighbors_.data();
    device_cellToNeighborsView_.contentStorage = device_cellToNeighborsStorage_.data();
    device_cellToNeighborsView_.offStorage = device_cellToNeighborsOffsets_.data();
    device_cellToNeighborsView_.contentSize = alpaka::getExtentProduct(device_cellToNeighborsStorage_); 
    device_cellToNeighborsView_.offSize = alpaka::getExtentProduct(device_cellToNeighborsOffsets_);
    std::cout << "device_cellToNeighborsView_" << device_cellToNeighborsView_.contentSize << " - " << device_cellToNeighborsView_.offSize << std::endl;

    GenericContainer::template launchZero<Acc1D>(device_cellToNeighborsView_, queue);

    // Tracks -> Hits
    device_tupleMultiplicityView_.assoc = device_tupleMultiplicity_.data();
    device_tupleMultiplicityView_.offStorage = device_tupleMultiplicityOffsets_.data();
    device_tupleMultiplicityView_.contentStorage = device_tupleMultiplicityStorage_.data();
    device_tupleMultiplicityView_.contentSize = alpaka::getExtentProduct(device_tupleMultiplicityStorage_);
    device_tupleMultiplicityView_.offSize = alpaka::getExtentProduct(device_tupleMultiplicityOffsets_);
    
    GenericContainer::template launchZero<Acc1D>(device_tupleMultiplicityView_, queue);

    // No.Hits -> Track (Multiplicity)
    device_hitContainerView_.assoc = device_hitContainer_.data();
    device_hitContainerView_.offStorage = device_hitContainerOffsets_.data();
    device_hitContainerView_.contentStorage = device_hitContainerStorage_.data();
    device_hitContainerView_.contentSize = alpaka::getExtentProduct(device_hitContainerStorage_);
    device_hitContainerView_.offSize = alpaka::getExtentProduct(device_hitContainerOffsets_);
    
    SequentialContainer::template launchZero<Acc1D>(device_hitContainerView_, queue);

    
    // in OneToManyAssoc?
    // initGenericContainer(device_hitToTupleView_,
    // device_hitToTuple_,device_hitToTupleOffsets_.data(),
    // device_hitToTupleStorage_.data(),
    // alpaka::getExtentProduct(device_hitToTupleOffsets_),
    // alpaka::getExtentProduct(device_hitToTupleStorage_));

    deviceTriplets_ = CACoupleSoACollection(device_cellToNeighborsView_.contentSize,queue);
    
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Allocations for CAHitNtupletGeneratorKernels: done!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::prepareHits(const HitsConstView &hh,
                                                                const HitModulesConstView &mm,
                                                                const reco::CALayersSoAConstView& ll,
                                                                Queue &queue)
    {
        using namespace caHitNtupletGeneratorKernels;

        const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, ll.metadata().size());
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            setHitsLayerStart{},
                            mm,
                            ll,
                            this->device_layerStarts_.data());
        
        cms::alpakatools::fillManyFromVector<Acc1D>(device_hitPhiHist_.data(),
                                                    device_hitPhiView_,
                                                    TrackerTraits::numberOfLayers,
                                                    hh.iphi(),
                                                    this->device_layerStarts_.data(),
                                                    hh.metadata().size(),
                                                    (uint32_t)256,
                                                    queue);
        
// #ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "CAHitNtupletGeneratorKernels -> Hits prepared (layer starts and histo) -> DONE!" << std::endl;
// #endif

    }   
  
  
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::launchKernels(const HitsConstView &hh,
                                                                  uint32_t offsetBPIX2,
                                                                  uint16_t nLayers,
                                                                  TkSoAView &tracks_view,
                                                                  TkHitsSoAView &tracks_hits_view,
                                                                  const reco::CALayersSoAConstView &ll,
                                                                  const reco::CACellsSoAConstView &cc,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

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
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.algoParams_.maxNumberOfDoublets_ / 4, blockSize);
    const auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.algoParams_.maxNumberOfDoublets_ / 4, blockSize);
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
    
    // uint32_t nCells = 0;
    // auto nCells_h = cms::alpakatools::make_host_view(nCells);
    // alpaka::memcpy(queue, nCells_h, this->device_nCells_);

    
    // auto cellNeighborsHisto = cms::alpakatools::make_device_buffer<GenericContainer> (queue);
    // auto cellNeighborsStorage = cms::alpakatools::make_device_buffer<GenericContainer::Counter[]>(queue, nCells * 64);
    // auto cellNeighborsStorageOff = cms::alpakatools::make_device_buffer<GenericContainer::Counter[]>(queue, nCells);
    // GenericContainerView cellNeighborsView;
    
    // cellNeighborsView.assoc = cellNeighborsHisto.data();
    // cellNeighborsView.offSize = nCells + 1;
    // cellNeighborsView.offStorage = cellNeighborsStorageOff.data();
    // cellNeighborsView.contentSize = nCells * 64;
    // cellNeighborsView.contentStorage = cellNeighborsStorage.data();
    
    // GenericContainer::template launchZero<Acc1D>(cellNeighborsView, queue);

    // std::cout << "Found nCells: " << nCells << std::endl;
    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        Kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_, // needed only to be reset, ready for next kernel
                        hh,
                        ll,
                        this->deviceTriplets_.view(),
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        this->device_nTriplets_.data(),
                        this->device_theCellNeighbors_.data(),
                        this->isOuterHitOfCell_.data(),
                        this->device_hitToCell_.data(),
                        this->m_params.algoParams_);
    
    // alpaka::exec<Acc2D>(queue,
    //                 kernelConnectWorkDiv,
    //                 Kernel_connectFill<TrackerTraits>{},
    //                 hh,
    //                 this->device_theCells_.data(),
    //                 this->device_nCells_.data(),
    //                 this->isOuterHitOfCell_.data(),
    //                 cellNeighborsHisto.data());
                    
    // GenericContainer::template launchFinalize<Acc1D>(cellNeighborsHisto.data(), queue);

    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.earlyFishbone_ and nhits > offsetBPIX2) {
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
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.algoParams_.maxNumberOfDoublets_ / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_find_ntuplets<TrackerTraits>{},
                        hh,
                        cc,
                        tracks_view,
                        this->device_hitContainer_.data(),
                        // cellNeighborsHisto.data(),
                        this->device_theCells_.data(),
                        this->device_nTriplets_.data(),
                        this->device_nCells_.data(),
                        this->device_theCellTracks_.data(),
                        this->device_hitTuple_apc_,
                        this->m_params.algoParams_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (this->m_params.algoParams_.doStats_)
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_mark_used<TrackerTraits>{},
                          this->device_theCells_.data(),
                          this->device_nCells_.data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(m_params.algoParams_.maxNumberOfTuples_+1, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(
        queue, workDiv1D, typename HitContainer::finalizeBulk{}, this->device_hitTuple_apc_, this->device_hitContainer_.data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_fillHitDetIndices<TrackerTraits>{}, tracks_view, tracks_hits_view, this->device_hitContainer_.data(), hh);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_fillNLayers<TrackerTraits>{}, tracks_view, tracks_hits_view, this->device_layerStarts_.data(), nLayers, this->device_hitTuple_apc_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.algoParams_.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_earlyDuplicateRemover<TrackerTraits>{},
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        tracks_view,
                        this->m_params.algoParams_.dupPassThrough_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.algoParams_.maxNumberOfTuples_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_countMultiplicity<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_.data(),
                        this->device_tupleMultiplicity_.data());
    GenericContainer::template launchFinalize<Acc1D>(this->device_tupleMultiplicity_.data(), queue);

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, Kernel_fillMultiplicity<TrackerTraits>{}, tracks_view, this->device_hitContainer_.data(), this->device_tupleMultiplicity_.data());
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.lateFishbone_ and nhits > offsetBPIX2) {
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
                                                                  const ::reco::CACellsSoAConstView &cc,
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

    const int stride = 4;
    int threadsPerBlock = TrackerTraits::getDoubletsFromHistoMaxBlockSize / stride;
    int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
    const Vec2D blks{blocks, 1u};
    const Vec2D thrs{threadsPerBlock, stride};
    const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        GetDoubletsFromHisto<TrackerTraits>{},
                        this->device_theCells_.data(),
                        this->device_simpleCells_.data(),
                        this->device_nCells_.data(),
                        this->device_theCellNeighbors_.data(),
                        this->device_theCellTracks_.data(),
                        hh,
                        cc,
                        this->device_layerStarts_.data(),
                        this->device_hitPhiHist_.data(),
                        this->isOuterHitOfCell_.data(),
                        this->device_hitToCell_.data(),
                        this->m_params.algoParams_
                        );

    HitToCell::template launchFinalize<Acc1D>(this->device_hitToCellView_, queue);

    threadsPerBlock = 512;
    blocks = cms::alpakatools::divide_up_by( m_params.algoParams_.maxNumberOfDoublets_ , threadsPerBlock);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        FillDoubletsHisto<TrackerTraits>{},
                        this->device_simpleCells_.data(),
                        this->device_nCells_.data(),
                        this->device_hitToCell_.data());

    

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
        queue, workDiv1D, Kernel_classifyTracks<TrackerTraits>{}, tracks_view, this->device_hitContainer_.data(), this->m_params.qualityCuts_);

    if (this->m_params.algoParams_.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.algoParams_.maxNumberOfDoublets_ / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fishboneCleaner<TrackerTraits>{},
                          this->device_theCells_.data(),
                          this->device_nCells_.data(),
                          tracks_view);
    }

    // mark duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.algoParams_.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fastDuplicateRemover<TrackerTraits>{},
                        this->device_theCells_.data(),
                        this->device_nCells_.data(),
                        tracks_view,
                        this->m_params.algoParams_.dupPassThrough_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (this->m_params.algoParams_.doSharedHitCut_ || this->m_params.algoParams_.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfQuadruplets / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_.data(),
                          this->device_hitToTuple_.data()); 

      GenericContainer::template launchFinalize<Acc1D>(this->device_hitToTupleView_, queue);
      alpaka::exec<Acc1D>(
          queue, workDiv1D, Kernel_fillHitInTracks<TrackerTraits>{}, tracks_view, this->device_hitContainer_.data(), this->device_hitToTuple_.data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif
    }

    if (this->m_params.algoParams_.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * TrackerTraits::maxNumberOfQuadruplets / 4,
                                                      blockSize);  // TODO: Check if correct
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.algoParams_.minHitsForSharingCut_,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitToTuple_.data());

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_sharedHitCleaner<TrackerTraits>{},
                          hh,
                          this->device_layerStarts_.data(),
                          tracks_view,
                          this->m_params.algoParams_.minHitsForSharingCut_,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitToTuple_.data());

      if (this->m_params.algoParams_.useSimpleTripletCleaner_) {
        // (typename HitToTuple{}::capacity(),
        numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple{}.capacity(), blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_simpleTripletCleaner<TrackerTraits>{},
                            tracks_view,
                            this->m_params.algoParams_.minHitsForSharingCut_,
                            this->m_params.algoParams_.dupPassThrough_,
                            this->device_hitToTuple_.data());
      } else {
        numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple{}.capacity(), blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_tripletCleaner<TrackerTraits>{},
                            tracks_view,
                            this->m_params.algoParams_.minHitsForSharingCut_,
                            this->m_params.algoParams_.dupPassThrough_,
                            this->device_hitToTuple_.data());
      }
#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif
    }

    if (this->m_params.algoParams_.doStats_) {
      numberOfBlocks =
          cms::alpakatools::divide_up_by(std::max(nhits, m_params.algoParams_.maxNumberOfDoublets_), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_checkOverflows<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_.data(),
                          this->device_tupleMultiplicity_.data(),
                          this->device_hitToTuple_.data(),
                          this->device_hitTuple_apc_,
                          this->device_theCells_.data(),
                          this->device_nCells_.data(),
                          this->device_theCellNeighbors_.data(),
                          this->device_theCellTracks_.data(),
                          this->isOuterHitOfCell_.data(),
                          nhits,
                          this->m_params.algoParams_.maxNumberOfDoublets_,
                          this->counters_.data());
    }

    if (this->m_params.algoParams_.doStats_) {
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
          queue, workDiv1D, Kernel_doStatsForTracks<TrackerTraits>{}, tracks_view, this->device_hitContainer_.data(), this->counters_.data());
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
                            this->device_hitContainer_.data(),
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
