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

// #define GPU_DEBUG
// #define NTUPLE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  CAHitNtupletGeneratorKernels<TrackerTraits>::CAHitNtupletGeneratorKernels(Params const &params,
                                                                            uint32_t nHits,
                                                                            uint32_t offsetBPIX2,
                                                                            uint32_t maxDoublets,
                                                                            uint32_t maxTuples,
                                                                            uint16_t nLayers,
                                                                            Queue &queue)
      : m_params(params),
        //////////////////////////////////////////////////////////
        // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
        //////////////////////////////////////////////////////////
        counters_{cms::alpakatools::make_device_buffer<Counters>(queue)},

        // One to Many Maps
        // Hits -> Track
        device_hitToTuple_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_hitToTupleStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, int(nHits * m_params.algoParams_.avgHitsPerTrack_) + 1)},
        device_hitToTupleOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nHits + 1)},
        device_hitToTupleView_{device_hitToTuple_.data(), device_hitToTupleOffsets_.data(), device_hitToTupleStorage_.data(), int(nHits + 1), int(nHits * m_params.algoParams_.avgHitsPerTrack_) + 1},
        
        // (Outer) Hits-> Cells
        device_hitToCell_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_hitToCellStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, int((nHits - offsetBPIX2) * m_params.algoParams_.avgCellsPerHit_) + 1)},
        device_hitToCellOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nHits - offsetBPIX2 + 1)},
        device_hitToCellView_{device_hitToCell_.data(), device_hitToCellOffsets_.data(), device_hitToCellStorage_.data(), int(nHits - offsetBPIX2 + 1), int((nHits - offsetBPIX2) * m_params.algoParams_.avgCellsPerHit_) + 1},

        // Hits
        device_hitPhiHist_{cms::alpakatools::make_device_buffer<PhiBinner>(queue)},
        device_phiBinnerStorage_{cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nHits)},
        device_hitPhiView_{device_hitPhiHist_.data(), nullptr, device_phiBinnerStorage_.data(), -1, int(nHits)},
        device_layerStarts_{cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nLayers + 1)}, 

        // Cell -> Neighbor Cells
        device_cellToNeighbors_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_cellToNeighborsStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, int(maxDoublets * m_params.algoParams_.avgCellsPerCell_) + 1)},
        device_cellToNeighborsOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, maxDoublets + 1)},
        device_cellToNeighborsView_{device_cellToNeighbors_.data(), device_cellToNeighborsOffsets_.data(), device_cellToNeighborsStorage_.data(), int(maxDoublets + 1), int(maxDoublets * m_params.algoParams_.avgCellsPerCell_)},
        
        // Cell -> Tracks
        device_cellToTracks_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_cellToTracksStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, int(maxDoublets * m_params.algoParams_.avgTracksPerCell_) + 1)},
        device_cellToTracksOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, maxDoublets + 1)},
        device_cellToTracksView_{device_cellToTracks_.data(), device_cellToTracksOffsets_.data(), device_cellToTracksStorage_.data(), int(maxDoublets + 1), int(maxDoublets * m_params.algoParams_.avgTracksPerCell_) + 1},

        // Tracks -> Hits
        device_hitContainer_{cms::alpakatools::make_device_buffer<SequentialContainer>(queue)},
        device_hitContainerStorage_{cms::alpakatools::make_device_buffer<SequentialContainerStorage[]>(queue, int(m_params.algoParams_.avgHitsPerTrack_ * maxTuples) + 1)},
        device_hitContainerOffsets_{cms::alpakatools::make_device_buffer<SequentialContainerOffsets[]>(queue, maxTuples + 1)},     
        device_hitContainerView_{device_hitContainer_.data(), device_hitContainerOffsets_.data(), device_hitContainerStorage_.data(), int(maxTuples + 1), int(m_params.algoParams_.avgHitsPerTrack_ * maxTuples) + 1},

        // No.Hits -> Track (Multiplicity)
        device_tupleMultiplicity_{cms::alpakatools::make_device_buffer<GenericContainer>(queue)},
        device_tupleMultiplicityStorage_{cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, maxTuples)},
        device_tupleMultiplicityOffsets_{cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, TrackerTraits::maxHitsOnTrack + 1)},
        device_tupleMultiplicityView_{device_tupleMultiplicity_.data(), device_tupleMultiplicityOffsets_.data(), device_tupleMultiplicityStorage_.data(), int(TrackerTraits::maxHitsOnTrack + 1), int(maxTuples)},
        
        // Structures and Counters Storage
        device_simpleCells_{
            cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, maxDoublets)},
        device_extraStorage_{
            cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 3u)},
        device_hitTuple_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_extraStorage_.data())},
        // device_cellCell_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_extraStorage_.data() +1)},
        device_nCells_{
            cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_.data() + 2))},
        device_nTriplets_{
            cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_.data() + 3))},
        device_nCellTracks_{
            cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_.data() + 4))},
        deviceTriplets_{CACoupleSoACollection(maxDoublets * m_params.algoParams_.avgCellsPerCell_,queue)},
        deviceTracksCells_{CACoupleSoACollection(int(maxDoublets * m_params.algoParams_.avgTracksPerCell_) + 1,queue)}
        {
#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building. N hits " << nHits << std::endl;
    std::cout << "maxTrips     = " << int(maxDoublets * m_params.algoParams_.avgCellsPerCell_) + 1 << std::endl;
    std::cout << "maxDoublets  = " << maxDoublets << std::endl;
    std::cout << "maxTrackCell = " << int(maxDoublets * m_params.algoParams_.avgTracksPerCell_) + 1 << std::endl;
#endif

    //if doStats?
    alpaka::memset(queue, counters_, 0);
    
    alpaka::memset(queue, device_nCells_, 0);
    alpaka::memset(queue, device_nTriplets_,0);
    alpaka::memset(queue, device_nCellTracks_,0);

    // alpaka::memset(queue,device_hitToTupleOffsets_,0); 
    // alpaka::memset(queue,device_hitToCellOffsets_,0); 
    // alpaka::memset(queue,device_cellToNeighborsOffsets_,0); 
    // alpaka::memset(queue,device_cellToTracksOffsets_,0); 
    // alpaka::memset(queue,device_hitContainerOffsets_,0); 
    // alpaka::memset(queue,device_tupleMultiplicityOffsets_,0); 
    // Hits -> Track
    HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);

    // (Outer) Hits-> Cells
    HitToCell::template launchZero<Acc1D>(device_hitToCellView_, queue);

    // Cells-> Neighbor Cells
    CellToCell::template launchZero<Acc1D>(device_cellToNeighborsView_, queue);
    
    // Cells-> Neighbor Cells
    CellToTrack::template launchZero<Acc1D>(device_cellToTracksView_, queue);
    
    // Tracks -> Hits
    TupleMultiplicity::template launchZero<Acc1D>(device_tupleMultiplicityView_, queue);

    // No.Hits -> Track (Multiplicity)
    HitContainer::template launchZero<Acc1D>(device_hitContainerView_, queue);
    
    maxNumberOfDoublets_ = maxDoublets;

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
                            SetHitsLayerStart{},
                            mm,
                            ll,
                            this->device_layerStarts_.data());
        
        cms::alpakatools::fillManyFromVector<Acc1D>(device_hitPhiHist_.data(),
                                                    device_hitPhiView_,
                                                    TrackerTraits::numberOfLayers, // could be ll.metadata().size()
                                                    hh.iphi(),
                                                    this->device_layerStarts_.data(),
                                                    hh.metadata().size(),
                                                    (uint32_t)256,
                                                    queue);
        
#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "CAHitNtupletGeneratorKernels -> Hits prepared (layer starts and histo) -> DONE!" << std::endl;
#endif

    }   
  
  
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::launchKernels(const HitsConstView &hh,
                                                                  uint32_t offsetBPIX2,
                                                                  uint16_t nLayers,
                                                                  TkSoAView &tracks_view,
                                                                  TkHitsSoAView &tracks_hits_view,
                                                                  const reco::CALayersSoAConstView &ll,
                                                                  const reco::CAGraphSoAConstView &cc,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    uint32_t nhits = hh.metadata().size();
    auto const maxDoublets = this->maxNumberOfDoublets_;
    auto const maxTuples = tracks_view.metadata().size();
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
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    const auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
    
    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        Kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_, // needed only to be reset, ready for next kernel
                        // this->device_cellCell_apc_,
                        hh,
                        ll,
                        this->deviceTriplets_.view(),
                        // this->device_theCells_.data(),
                        this->device_simpleCells_.data(),
                        this->device_nCells_.data(),
                        this->device_nTriplets_.data(),
                        // this->device_theCellNeighbors_.data(),
                        // this->isOuterHitOfCell_.data(),
                        this->device_hitToCell_.data(),
                        this->device_cellToNeighbors_.data(),
                        this->m_params.algoParams_);

    CellToCell::template launchFinalize<Acc1D>(this->device_cellToNeighborsView_, queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_connect -> Done!" << std::endl;
#endif

    auto threadsPerBlock = 1024;
    auto blocks = cms::alpakatools::divide_up_by( maxDoublets * m_params.algoParams_.avgCellsPerCell_ , threadsPerBlock);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                    workDiv1D,
                    Kernel_fillGenericCouple<TrackerTraits>{},
                    this->deviceTriplets_.view(),
                    this->device_nTriplets_.data(),
                    this->device_cellToNeighbors_.data());
    
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "cellToNeighbors -> Filled!" << std::endl;
#endif

    // Cells-> Tracks
    // CellToTrack::template launchInit<Acc1D>(device_cellToTracksView_, queue);
          
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
                        //   this->device_theCells_.data(),
                          this->device_simpleCells_.data(),
                          this->device_nCells_.data(),
                        //   this->isOuterHitOfCell_.data(),
                          this->device_hitToCell_.data(),
                          this->device_cellToTracks_.data(),
                          nhits - offsetBPIX2,
                          false);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Early fishbone -> Done!" << std::endl;
#endif

    }
    blockSize = 64;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_find_ntuplets<TrackerTraits>{},
                        cc,
                        tracks_view,
                        this->device_hitContainer_.data(),
                        this->device_cellToNeighbors_.data(),
                        this->device_cellToTracks_.data(),
                        this->deviceTracksCells_.view(),
                        this->device_simpleCells_.data(),
                        this->device_nCellTracks_.data(),
                        this->device_nTriplets_.data(),
                        this->device_nCells_.data(),
                        this->device_hitTuple_apc_,
                        this->m_params.algoParams_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_find_ntuplets -> Done!" << std::endl;
#endif

    CellToTracks::template launchFinalize<Acc1D>(this->device_cellToTracksView_, queue);

    blocks = cms::alpakatools::divide_up_by( maxDoublets * m_params.algoParams_.avgCellsPerCell_ , threadsPerBlock);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                    workDiv1D,
                    Kernel_fillGenericCouple<TrackerTraits>{},
                    this->deviceTracksCells_.view(),
	                this->device_nCellTracks_.data(),
                    this->device_cellToTracks_.data());
                    
    if (this->m_params.algoParams_.doStats_)
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_mark_used<TrackerTraits>{},
                          this->device_simpleCells_.data(),
                          this->device_cellToTracks_.data(),
                          this->device_nCells_.data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples+1, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(
        queue, workDiv1D, typename HitContainer::finalizeBulk{}, this->device_hitTuple_apc_, this->device_hitContainer_.data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_fillHitDetIndices<TrackerTraits>{}, tracks_view, tracks_hits_view, this->device_hitContainer_.data(), hh);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillHitDetIndices   -> done!" << std::endl;
#endif
    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_fillNLayers<TrackerTraits>{}, tracks_view, tracks_hits_view, this->device_layerStarts_.data(), nLayers, this->device_hitTuple_apc_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillNLayers   -> done!" << std::endl;
#endif

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_earlyDuplicateRemover<TrackerTraits>{},
                        this->device_simpleCells_.data(),
                        this->device_nCells_.data(),
                        this->device_cellToTracks_.data(),
                        tracks_view,
                        this->m_params.algoParams_.dupPassThrough_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_earlyDuplicateRemover   -> done!" << std::endl;
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_countMultiplicity<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_.data(),
                        this->device_tupleMultiplicity_.data());
    GenericContainer::template launchFinalize<Acc1D>(this->device_tupleMultiplicityView_, queue);

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, Kernel_fillMultiplicity<TrackerTraits>{}, tracks_view, this->device_hitContainer_.data(), this->device_tupleMultiplicity_.data());
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_countMultiplicity -> done!" << std::endl;
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
                        //   this->device_theCells_.data(),
                          this->device_simpleCells_.data(),
                          this->device_nCells_.data(),
                        //   this->isOuterHitOfCell_.data(),
                          this->device_hitToCell_.data(),
                          this->device_cellToTracks_.data(),
                          nhits - offsetBPIX2,
                          true);
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(const HitsConstView &hh,
                                                                  const ::reco::CAGraphSoAConstView &cc,
                                                                  uint32_t offsetBPIX2,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;

    auto nhits = hh.metadata().size();
    const auto maxDoublets = this->maxNumberOfDoublets_;
#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

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
    
    // (Outer) Hits-> Cells
    // HitToCell::template launchInit<Acc1D>(this->device_hitToCellView_, queue);
#ifdef GPU_DEBUG
    std::cout << "nActualPairs = " << cc.metadata().size() << std::endl;
#endif
    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        GetDoubletsFromHisto<TrackerTraits>{},
                        // this->device_theCells_.data(),
                        maxDoublets,
                        this->device_simpleCells_.data(),
                        this->device_nCells_.data(),
                        // this->device_cellCell_apc_,
                        // this->device_theCellNeighbors_.data(),
                        // this->device_theCellTracks_.data(),
                        hh,
                        cc,
                        this->device_layerStarts_.data(),
                        this->device_hitPhiHist_.data(),
                        // this->isOuterHitOfCell_.data(),
                        this->device_hitToCell_.data(),
                        this->m_params.algoParams_
                        );

    HitToCell::template launchFinalize<Acc1D>(this->device_hitToCellView_, queue);

    threadsPerBlock = 512;
    blocks = cms::alpakatools::divide_up_by( maxDoublets , threadsPerBlock);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        FillDoubletsHisto<TrackerTraits>{},
                        this->device_simpleCells_.data(),
                        this->device_nCells_.data(),
                        offsetBPIX2,
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
    auto const maxDoublets = this->maxNumberOfDoublets_;
    auto const maxTuples = tracks_view.metadata().size();
    // classify tracks based on kinematics
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(
        queue, workDiv1D, Kernel_classifyTracks<TrackerTraits>{}, tracks_view, this->device_hitContainer_.data(), this->m_params.qualityCuts_);

    if (this->m_params.algoParams_.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fishboneCleaner<TrackerTraits>{},
                          this->device_simpleCells_.data(),
                          this->device_nCells_.data(),
                          this->device_cellToTracks_.data(),
                          tracks_view);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fishboneCleaner   -> done!" << std::endl;
#endif
    // mark duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fastDuplicateRemover<TrackerTraits>{},
                        this->device_simpleCells_.data(),
                        this->device_nCells_.data(),
                        this->device_cellToTracks_.data(),
                        tracks_view,
                        this->m_params.algoParams_.dupPassThrough_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fastDuplicateRemover   -> done!" << std::endl;
#endif

    if (this->m_params.algoParams_.doSharedHitCut_ || this->m_params.algoParams_.doStats_) {

    //   // Hits -> Track
    //   HitToTuple::template launchInit<Acc1D>(device_hitToTupleView_, queue);

      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
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
      std::cout << "Kernel_countHitInTracks   -> done!" << std::endl;
#endif
    }

    if (this->m_params.algoParams_.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4,
                                                      blockSize);  // TODO: Check if correct
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.algoParams_.minHitsForSharingCut_,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitToTuple_.data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_rejectDuplicate   -> done!" << std::endl;
#endif

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_sharedHitCleaner<TrackerTraits>{},
                          hh,
                          this->device_layerStarts_.data(),
                          tracks_view,
                          this->m_params.algoParams_.minHitsForSharingCut_,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitToTuple_.data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_sharedHitCleaner   -> done!" << std::endl;
#endif
      if (this->m_params.algoParams_.useSimpleTripletCleaner_) {
        numberOfBlocks = cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_simpleTripletCleaner<TrackerTraits>{},
                            tracks_view,
                            this->m_params.algoParams_.minHitsForSharingCut_,
                            this->m_params.algoParams_.dupPassThrough_,
                            this->device_hitToTuple_.data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_simpleTripletCleaner   -> done!" << std::endl;
#endif
      } else {
        numberOfBlocks = cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_tripletCleaner<TrackerTraits>{},
                            tracks_view,
                            this->m_params.algoParams_.minHitsForSharingCut_,
                            this->m_params.algoParams_.dupPassThrough_,
                            this->device_hitToTuple_.data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_tripletCleaner   -> done!" << std::endl;
#endif
      }

    }

    if (this->m_params.algoParams_.doStats_) {
      numberOfBlocks =
          cms::alpakatools::divide_up_by(std::max(nhits, maxDoublets), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_checkOverflows<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_.data(),
                          this->device_tupleMultiplicity_.data(),
                          this->device_hitToTuple_.data(),
                          this->device_hitTuple_apc_,
                        //   this->device_theCells_.data(),
                          this->device_simpleCells_.data(),
                          this->device_nCells_.data(),
                        //   this->device_theCellNeighbors_.data(),
                        //   this->device_theCellTracks_.data(),
                        //   this->isOuterHitOfCell_.data(),
                          nhits,
                          this->maxNumberOfDoublets_,
                          this->m_params.algoParams_,
                          this->counters_.data());
    }

    if (this->m_params.algoParams_.doStats_) {
      // counters (add flag???)

      numberOfBlocks = cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForHitInTracks<TrackerTraits>{},
                          this->device_hitToTuple_.data(),
                          this->counters_.data());

      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
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
