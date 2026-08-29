// Orchestration TU of the SPLIT BrokenLine fit build. The heavy per-N device kernels
// (Kernel_BLFit + fast-fit kernels) and the per-N launcher helpers
// live in BrokenLineFitKernels.h and are explicitly instantiated in the
// disjoint-N BrokenLineFit_*.dev.cc TUs; here they are extern-template calls only. This file
// keeps the light launcher orchestration (HelixFit members), the debug dump
// kernel, and the HelixFit explicit instantiations. The split is a build-time division only: the
// launches keep the same order, arguments and grids they would have in a single TU.
#include "BrokenLineFitKernels.h"
#include <optional>  // fused-ladder partition tables, allocated only on the fused arm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // kFitBlock, the block dimension of the fit work divisions, lives in BrokenLineFitKernels.h next
  // to the fused main-fit ladder, whose bin <-> lane partition is expressed in units of it.

  // -------------------------------------------------------------------------
  // Debug dump of the first N fitted tracks (BL output) -- gated by the
  // HelixFit::setVerboseDump switch.  Converts the internal 5-parameter state
  // (phi0, d0, kappa, cotTheta, z0) into more readable pT [GeV] and eta so
  // the singleMu sample's reconstructed muon is identifiable at a glance.
  // pT = 0.003 * B[T] / |kappa[1/cm]|     (charge sign carried in kappa)
  // eta = asinh(cotTheta)
  // -------------------------------------------------------------------------
  class Kernel_BLFitDump {
  public:
    // tagId is an integer rendered inline so different fit invocations can be told apart in the log. Set it via
    // HelixFit::setVerboseDump(true) and the per-call argument. Passing only integers/floats keeps the kernel
    // portable (no host-string pointers crossing the host/device boundary).
    Kernel_BLFitDump(uint32_t n, float bField, uint32_t tagId) : nToPrint_(n), bField_(bField), tagId_(tagId) {}
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::reco::TrackSoAView tracks) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      const uint32_t total = uint32_t(std::max(0, tracks.nTracks()));
      const uint32_t cap = (nToPrint_ < total) ? nToPrint_ : total;
      printf("[BL-fit tag=%u] post-fit dump: nTracks=%u, showing first %u\n", tagId_, total, cap);
      for (uint32_t i = 0; i < cap; ++i) {
        const float phi0 = tracks[i].state()(0);
        const float d0 = tracks[i].state()(1);
        // state(2) is 1/pT in 1/GeV (CMSSW BL convention; see TracksSoA.h).
        // pT[GeV] = 1 / |state(2)|.  Sign of state(2) carries the charge.
        const float invPt = tracks[i].state()(2);
        const float cotTheta = tracks[i].state()(3);
        const float z0 = tracks[i].state()(4);
        const float absInvPt = std::abs(invPt);
        const float pT = (absInvPt > 1.e-9f) ? (1.f / absInvPt) : 1.e9f;
        const float charge = (invPt >= 0.f) ? +1.f : -1.f;
        const float eta = std::asinh(cotTheta);
        const float chi2 = tracks[i].chi2();
        const int q = int(tracks[i].quality());
        // Reported track uncertainties: covariance() is the packed Vector15f; idx 0 = sigma^2(phi),
        // idx 5 = sigma^2(d0)=sigma^2(dxy) (see DataFormats/TrackSoA/interface/TracksSoA.h).
        const float sdxy = 1.e4f * std::sqrt(std::abs(float(tracks[i].covariance()(5))));  // um
        const float sphi = std::sqrt(std::abs(float(tracks[i].covariance()(0))));          // rad
        // longitudinal: cov(14)=var(Zip=z0)=var(dz), cov(12)=var(cotTheta) (see TracksSoA copyFromCircle).
        const float sdz = 1.e4f * std::sqrt(std::abs(float(tracks[i].covariance()(14))));  // um, sigma(dz)
        const float scot = std::sqrt(std::abs(float(tracks[i].covariance()(12))));         // sigma(cotTheta)
        printf(
            "[BL-fit tag=%u] i=%u q=%d chi2=%.3f phi0=%.4f d0=%.5f invPt=%.6f cotTheta=%.4f z0=%.4f "
            "-> pT=%.3f GeV q=%+0.0f eta=%.4f  sigma(dxy)=%.2f um sigma(phi)=%.5f rad  sigma(dz)=%.2f um "
            "sigma(cotTheta)=%.6f\n",
            tagId_,
            i,
            q,
            chi2,
            phi0,
            d0,
            invPt,
            cotTheta,
            z0,
            pT,
            charge,
            eta,
            sdxy,
            sphi,
            sdz,
            scot);
      }
    }

  private:
    uint32_t nToPrint_;
    float bField_;
    uint32_t tagId_;
  };

  // The CA main fit: one full sweep of the N-binned BLFastFit+BLFit kernels, i.e. the factorized fast
  // BrokenLine fit (circle + line), for every CA iteration and every topology. It has ONE linearization,
  // so it is a single sweep with no phase split and no re-linearization reference buffer.
  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchBrokenLineKernels(const caStructures::HitsMultiView& hv,
                                                        const ::reco::CAModulesConstView& cm,
                                                        uint32_t hitsInFit,
                                                        uint32_t maxNumberOfTuples,
                                                        Queue& queue) {
    ALPAKA_ASSERT_ACC(tuples_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting HelixFit<TrackerTraits>::launchBrokenLineKernels" << std::endl;
#endif

    // Main-fit block dimension: kFitBlock. Launch
    // dimension only -- the fast-fit and fit kernels are grid-stride loops with the invalid-tkid break,
    // so the fitted-lane set and every fitted value are independent of it.
    const uint32_t blockSize = kFitBlock;
    // The launch grid sizes off maxNumberOfTuples, the host-known per-event tuple capacity (the
    // hit-count-scaled cap the CA sizes its tuple/multiplicity containers to; not a device readback). The
    // exact per-N-bin population lives only device-side (tupleMultiplicity); a tighter trim would need a
    // further D2H sync, which this path does not take. Grid sizing cannot move a result: the grid-stride
    // loop plus the invalid-tkid break make the fitted-lane set independent of the grid size.
    //  Fit internals
    auto tkidDevice =
        cms::alpakatools::make_device_buffer<typename caStructures::tindex_type[]>(queue, maxNumberOfConcurrentFits_);
    constexpr auto maxN = TrackerTraits::maxHitsOnTrackForFullFit;
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));
    // Per-fit solver scratch of the factorized fit: prepared-data + shared band block + helper vectors
    // per lane, in an [element][lane] layout whose stride is the concurrent-fit count, so it is allocated
    // at maxNumberOfConcurrentFits_ lanes (like the input buffers).
    constexpr int kScratchPerFit = brokenline::kLegacyFitScratchDoubles<int(maxN)>;
    auto gblScratchDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, std::size_t(maxNumberOfConcurrentFits_) * std::size_t(kScratchPerFit));
    // The per-bin ladder is the fitNas4 arm: that mode launches the N=4 bin once per rolling index and
    // then the tail bin, which is not a partition, so it cannot be expressed as a fused launch. It is also
    // what keeps runMainBin<N, TrackerTraits> (and the per-N Kernel_BLFastFit / Kernel_BLFit
    // instantiations of the split TUs) referenced.
    const bool serialLadder = fitNas4_;

    // Dynamic-partition bookkeeping of the fused ladder (see the block comment at the head of
    // BrokenLineFitKernels.h). One uint32 allocation holding, in order: the per-bin round cursor
    // [0, kNBins), the per-bin {firstLane, nLanes, tupleBase} ranges, and the per-block
    // {bin, firstLane, endLane} dispatch table (a few kB, caching-allocator backed). No memset:
    // Kernel_BLMainLaneRanges writes every word it later reads (the cursor on the first round, the ranges
    // and the block map on every round). The per-bin ladder never touches it.
    constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
    constexpr uint32_t kFusedBlk = kMainFusedBlocks<TrackerTraits>;
    constexpr uint32_t kPartWords = kNBins + kMainRangeStride * kNBins + kMainBlockMapStride * kFusedBlk;
    // Allocated only on the fused arm, so the per-bin ladder keeps its own allocation sequence: the
    // caching allocator hands out blocks in call order, and an extra allocation would change which block
    // every later allocation of the event gets.
    std::optional<decltype(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kPartWords))> partDevice;
    uint32_t* pCursor = nullptr;
    uint32_t* pRange = nullptr;
    uint32_t* pBlockMap = nullptr;
    if (!serialLadder) {
      partDevice.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kPartWords));
      pCursor = partDevice->data();
      pRange = pCursor + kNBins;
      pBlockMap = pRange + kMainRangeStride * kNBins;
    }
    // Fused grid: kMainFusedBlocks blocks of kFitBlock, the width that covers EVERY partition the
    // per-event populations can produce. Fixed and host-known; the device table says which blocks are
    // live and on which bin.
    const WorkDiv1D workDivFused = cms::alpakatools::make_workdiv<Acc1D>(kFusedBlk, kFitBlock);

    // Per-N launch state for the extern-template runMainBin helpers (see BrokenLineFitKernels.h).
    // The per-N (fast-fit + fit) launches live in runMainBin<N,TrackerTraits>, explicitly instantiated in
    // the BrokenLineFit_*.dev.cc N-range TUs so nvcc compiles the heavy per-N kernels in parallel.
    const BLMainLaunchCtx<TrackerTraits> ctx{queue,
                                             tuples_,
                                             tupleMultiplicity_,
                                             hv,
                                             cm,
                                             outputSoa_,
                                             bField_,
                                             rhoMap_,
                                             bMap_,
                                             tkidDevice.data(),
                                             hitsDevice.data(),
                                             hits_geDevice.data(),
                                             fast_fit_resultsDevice.data(),
                                             gblScratchDevice.data(),
                                             fitCorrections_,
                                             pRange,
                                             pBlockMap,
                                             workDivFused};

    // Chunk+bin-aware main-fit launch elision, driven by the tuple-multiplicity per-N-bin offsets:
    // (a) stop the chunk loop at the runtime fitted population and (b) elide any N-bin whose population
    // the chunk base offset already covers. Both are exactly the on-device no-op the launched kernel
    // would have performed (each thread's tuple_idx >= totTK on the first lane -> writes the invalid
    // sentinel and breaks -> no SoA write), so the collapse cannot move a result. Without it the offset
    // loop chunks the hit-scaled tuple cap and launches every N-bin in every chunk.
    constexpr uint32_t kNOff = TrackerTraits::maxHitsOnTrack + 2u;  // == the offsets buffer length
    uint32_t offHost[kNOff] = {0};
    bool skipEmpty = false;
    uint32_t chunkBound = maxNumberOfTuples;  // no offsets available: iterate the full hit-scaled cap
    if (hostTupleMultiplicityOffsets_ != nullptr) {
      // The caller (the producer's acquire) already read the offsets back with one async D2H and the
      // framework's acquire->produce boundary guarantees it has landed -- consuming the host values
      // here costs no memcpy and no wait. off[b] is the same array Kernel_BLFastFit reads for
      // totTK = off[nHitsH+1]-off[nHitsL], so the host populations are exactly what the on-device
      // break condition sees; nothing is re-derived here.
      for (uint32_t b = 0; b < kNOff; ++b)
        offHost[b] = hostTupleMultiplicityOffsets_[b];
      skipEmpty = true;
      // Total fitted population off[maxHitsOnTrack]-off[3] (tuples with selected-hit count in
      // [3, maxHitsOnTrack-1]) is an upper bound on any single N-bin's population, so a loop bounded by
      // it runs at least as many stride chunks as the largest bin needs; the per-bin gate elides the
      // rest. In practice this population is well below the stride, so the loop collapses to one chunk.
      const uint32_t lo = offHost[3];
      const uint32_t hi = offHost[TrackerTraits::maxHitsOnTrack];
      chunkBound = (hi > lo) ? (hi - lo) : 0u;
    }
    // Per-(N-bin, chunk) launch gate. A bin holding off[nHitsH+1]-off[nHitsL] tuples has NO work once the
    // chunk base offset reaches that count (Kernel_BLFastFit breaks on the first lane, Kernel_BLFit breaks
    // on the invalid sentinel -> zero SoA writes), so eliding the launch changes nothing.
    auto runBinGated = [&](auto Ntag, WorkDiv1D const& wd, uint32_t nHitsL, uint32_t nHitsH, uint32_t baseOffset) {
      constexpr int Nv = decltype(Ntag)::value;
      if (skipEmpty && baseOffset >= (offHost[nHitsH + 1u] - offHost[nHitsL]))
        return;
      runMainBin<Nv, TrackerTraits>(ctx, wd, nHitsL, nHitsH, baseOffset);
    };

    if (!serialLadder) {
      // The fused ladder. Per round: 1 range kernel + 1 fused fast fit + 1 fused fit, instead of the
      // per-bin ladder's 2 * kMainNBins launches per chunk. One round seats maxNumberOfConcurrentFits_
      // tuples and the demand can never exceed the population chunkBound counts, so
      // ceil(chunkBound / maxNumberOfConcurrentFits_) rounds always drain it and no tuple is left unfit.
      // chunkBound == 0 (no fitted population) runs zero rounds.
      const uint32_t nRounds = cms::alpakatools::divide_up_by(chunkBound, maxNumberOfConcurrentFits_);
      for (uint32_t round = 0; round < nRounds; ++round) {
        // The DYNAMIC PARTITION: prefix-sum the per-bin populations (read straight off
        // tupleMultiplicity -- no count kernel, no readback, no sync) into lane ranges and the
        // block -> bin dispatch table.
        alpaka::exec<Acc1D>(queue,
                            cms::alpakatools::make_workdiv<Acc1D>(1u, 1u),
                            Kernel_BLMainLaneRanges<TrackerTraits>{},
                            tupleMultiplicity_,
                            pCursor,
                            pRange,
                            pBlockMap,
                            maxNumberOfConcurrentFits_,
                            /*firstRound=*/round == 0u);
        runMainFusedFast<TrackerTraits>(ctx);
        runMainFusedFit<TrackerTraits>(ctx);
      }
    }

    for (uint32_t offset = 0; serialLadder && offset < chunkBound; offset += maxNumberOfConcurrentFits_) {
      // Size THIS chunk's grid from the remaining host-known tuple capacity. A full chunk
      // (>= a lane count of tuples still to come) takes the full 128-block triplet / 32-block quad-penta
      // grid; the last chunk -- and, when maxNumberOfTuples < the lane count, the only chunk -- trims to
      // divide_up(remaining). Grid-stride (uniform_elements over the lane stride) + the invalid-tkid break
      // make the fitted-lane set, and its values, independent of the block count. numberOfBlocks is >= 1
      // (remaining >= 1 while the loop runs); the quad-penta count is floored at 1 so its /4 never
      // collapses to a 0-block launch.
      const uint32_t chunkFits = std::min<uint32_t>(maxNumberOfConcurrentFits_, maxNumberOfTuples - offset);
      const uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(chunkFits, blockSize);
      const WorkDiv1D workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      const WorkDiv1D workDivQuadsPenta =
          cms::alpakatools::make_workdiv<Acc1D>(std::max<uint32_t>(1u, numberOfBlocks / 4), blockSize);
      // fit triplets
      runBinGated(std::integral_constant<int, 3>{}, workDivTriplets, 3u, 3u, offset);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_BLFastFit(3) and Kernel_BLFit(3) -> done! " << std::endl;
#endif

      if (fitNas4_) {
        // fit all as 4
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrack, 1>(
            [&runBinGated, &offset, &workDivQuadsPenta](auto i) {
              (void)i;
              runBinGated(std::integral_constant<int, 4>{}, workDivQuadsPenta, 4u, 4u, offset);
            });

        static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);

        //Fit all the rest using the maximum from previous call
        runBinGated(std::integral_constant<int, int(TrackerTraits::maxHitsOnTrackForFullFit)>{},
                    workDivQuadsPenta,
                    TrackerTraits::maxHitsOnTrackForFullFit,
                    TrackerTraits::maxHitsOnTrack - 1,
                    offset);
      } else {
        // Rolling fits for multiplicities 4 to maxHitsOnTrackForFullFit
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrackForFullFit, 1>(
            [&runBinGated, &offset, &workDivQuadsPenta](auto i) {
              constexpr int Nv = decltype(i)::value;
              runBinGated(i, workDivQuadsPenta, uint32_t(Nv), uint32_t(Nv), offset);
            });

        static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);

        // Fit all the rest using maxHitsOnTrackForFullFit hits
        runBinGated(std::integral_constant<int, int(TrackerTraits::maxHitsOnTrackForFullFit)>{},
                    workDivQuadsPenta,
                    TrackerTraits::maxHitsOnTrackForFullFit,
                    TrackerTraits::maxHitsOnTrack - 1,
                    offset);
      }
    }  // loop on concurrent fits

    if (verboseDump_) {
      // tag=1 -> post-fit dump of the main-fit launch
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(1u, 1u),
                          Kernel_BLFitDump{verboseDumpN_, bField_, 1u},
                          outputSoa_);
    }
  }

  template class HelixFit<pixelTopology::Phase1>;
  template class HelixFit<pixelTopology::Phase2>;
  template class HelixFit<pixelTopology::Phase2OT>;
  template class HelixFit<pixelTopology::Phase2OTStubs>;
  template class HelixFit<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
