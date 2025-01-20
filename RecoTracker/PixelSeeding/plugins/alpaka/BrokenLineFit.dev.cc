//#define BROKENLINE_DEBUG
//#define BL_DUMP_HITS

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/FrameSoALayout.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

#include "Geometry/CommonTopologies/interface/SimpleSeedingLayersTopology.h"

#include "HelixFit.h"

template <typename TrackerTraits>
using Tuples = typename reco::TrackSoA<TrackerTraits>::HitContainer;
template <typename TrackerTraits>
using OutputSoAView = reco::TrackSoAView<TrackerTraits>;
template <typename TrackerTraits>
using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

// #define BL_DUMP_HITS

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <int N, typename TrackerTraits>
  class Kernel_BLFastFit {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  Tuples<TrackerTraits> const *__restrict__ foundNtuplets,
                                  TupleMultiplicity<TrackerTraits> const *__restrict__ tupleMultiplicity,
                                  TrackingRecHitSoAConstView<TrackerTraits> hh,
                                  // pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const *__restrict__ cpeParams,
                                  FrameSoAConstView fr,
                                  typename TrackerTraits::tindex_type *__restrict__ ptkids,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  int32_t offset) const {
      constexpr uint32_t hitsInFit = N;
      constexpr auto invalidTkId = std::numeric_limits<typename TrackerTraits::tindex_type>::max();

      ALPAKA_ASSERT_ACC(hitsInFit <= nHitsL);
      ALPAKA_ASSERT_ACC(nHitsL <= nHitsH);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);

      // look in bin for this hit multiplicity
      int totTK = tupleMultiplicity->end(nHitsH) - tupleMultiplicity->begin(nHitsL);
      ALPAKA_ASSERT_ACC(totTK <= int(tupleMultiplicity->size()));
      ALPAKA_ASSERT_ACC(totTK >= 0);

#ifdef BROKENLINE_DEBUG
      const uint32_t threadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("%d total Ntuple\n", tupleMultiplicity->size());
        printf("%d Ntuple of size %d/%d for %d hits to fit\n", totTK, nHitsL, nHitsH, hitsInFit);
      }
#endif
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if ((int)tuple_idx >= totTK) {
          ptkids[local_idx] = invalidTkId;
          break;
        }
        // get it from the ntuple container (one to one to helix)
        auto tkid = *(tupleMultiplicity->begin(nHitsL) + tuple_idx);
        ALPAKA_ASSERT_ACC(static_cast<int>(tkid) < foundNtuplets->nOnes());

        ptkids[local_idx] = tkid;

        auto nHits = foundNtuplets->size(tkid);

        ALPAKA_ASSERT_ACC(nHits >= nHitsL);
        ALPAKA_ASSERT_ACC(nHits <= nHitsH);

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

#ifdef BL_DUMP_HITS
        auto &&done = alpaka::declareSharedVar<int, __COUNTER__>(acc);
        done = 0;
        alpaka::syncBlockThreads(acc);
        bool dump =
            (foundNtuplets->size(tkid) == 5 && 0 == alpaka::atomicAdd(acc, &done, 1, alpaka::hierarchy::Blocks{}));
#endif

        // Prepare data structure
        auto const *hitId = foundNtuplets->begin(tkid);

        // #define YERR_FROM_DC
#ifdef YERR_FROM_DC
        // try to compute more precise error in y
        auto dx = hh[hitId[hitsInFit - 1]].xGlobal() - hh[hitId[0]].xGlobal();
        auto dy = hh[hitId[hitsInFit - 1]].yGlobal() - hh[hitId[0]].yGlobal();
        auto dz = hh[hitId[hitsInFit - 1]].zGlobal() - hh[hitId[0]].zGlobal();
        float ux, uy, uz;
#endif

        float incr = std::max(1.f, float(nHits) / float(hitsInFit));
        float n = 0;
        for (uint32_t i = 0; i < hitsInFit; ++i) {
          int j = int(n + 0.5f);  // round
          if (hitsInFit - 1 == i)
            j = nHits - 1;  // force last hit to ensure max lever arm.
          ALPAKA_ASSERT_ACC(j < int(nHits));
          n += incr;
          auto hit = hitId[j];
          float ge[6];
          auto const &frame = fr.detFrame(hh.detectorIndex(hit));
#ifdef YERR_FROM_DC
          auto status = hh[hit].chargeAndStatus().status;
          int qbin = CPEFastParametrisation::kGenErrorQBins - 1 - status.qBin;
          ALPAKA_ASSERT_ACC(qbin >= 0 && qbin < 5);
          bool nok = (status.isBigY | status.isOneY);
          // compute cotanbeta and use it to recompute error
          frame.rotation().multiply(dx, dy, dz, ux, uy, uz);
          auto cb = std::abs(uy / uz);
          int bin =
              int(cb * (float(phase1PixelTopology::pixelThickess) / float(phase1PixelTopology::pixelPitchY)) * 8.f) - 4;
          int low_value = 0;
          int high_value = CPEFastParametrisation::kNumErrorBins - 1;
          // return estimated bin value truncated to [0, 15]
          bin = std::clamp(bin, low_value, high_value);
          float yerr = dp.sigmay[bin] * 1.e-4f;  // toCM
          yerr *= dp.yfact[qbin];                // inflate
          yerr *= yerr;
          yerr += dp.apeYY;
          yerr = nok ? hh[hit].yerrLocal() : yerr;
          frame.toGlobal(hh[hit].xerrLocal(), 0, yerr, ge);
#else
          frame.toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge);
          // if (hh[hit].detectorIndex() <= TrackerTraits::numberOfPixelModules)
          //   cpeParams->detParams(hh[hit].detectorIndex()).frame.toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge);
          // else
          // {
          //   auto xe = hh[hit].xerrLocal();
          //   auto ye = hh[hit].yerrLocal();

          //   ge[0] = xe;
          //   ge[1] = sqrt(xe*xe + ye*ye);
          //   ge[2] = ye;
          //   ge[3] = sqrt(xe*xe + ye*ye);
          //   ge[4] = sqrt(xe*xe + ye*ye);
          //   ge[5] = sqrt(xe*xe + ye*ye);
          // }
#endif

#ifdef BL_DUMP_HITS
          bool dump = foundNtuplets->size(tkid) == 5;
          if (dump) {
            printf("Track id %d %d Hit %d on %d\nGlobal: hits.col(%d) << %f,%f,%f\n",
                   local_idx,
                   tkid,
                   hit,
                   hh[hit].detectorIndex(),
                   i,
                   hh[hit].xGlobal(),
                   hh[hit].yGlobal(),
                   hh[hit].zGlobal());
            printf("Error: hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n", i, ge[0], ge[1], ge[2], ge[3], ge[4], ge[5]);
          }
#endif

          hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        brokenline::fastFit(acc, hits, fast_fit);

#ifdef BROKENLINE_DEBUG
        // any NaN value should cause the track to be rejected at a later stage
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(0)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(1)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(2)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(3)));
#endif
      }
    }
  };

  template <int N, typename TrackerTraits>
  struct Kernel_BLFit {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  TupleMultiplicity<TrackerTraits> const *__restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView<TrackerTraits> results_view,
                                  typename TrackerTraits::tindex_type const *__restrict__ ptkids,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit) const {
      ALPAKA_ASSERT_ACC(results_view.pt());
      ALPAKA_ASSERT_ACC(results_view.eta());
      ALPAKA_ASSERT_ACC(results_view.chi2());
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr auto invalidTkId = std::numeric_limits<typename TrackerTraits::tindex_type>::max();

      // same as above...
      // look in bin for this hit multiplicity
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        if (invalidTkId == ptkids[local_idx])
          break;
        auto tkid = ptkids[local_idx];

        ALPAKA_ASSERT_ACC(tkid < TrackerTraits::maxNumberOfTuples);

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        brokenline::PreparedBrokenLineData<N> data;

        brokenline::karimaki_circle_fit circle;
        riemannFit::LineFit line;

        brokenline::prepareBrokenLineData(acc, hits, fast_fit, bField, data);
        brokenline::lineFit(acc, hits_ge, fast_fit, bField, data, line);
        brokenline::circleFit(acc, hits, hits_ge, fast_fit, bField, data, circle);

        TracksUtilities<TrackerTraits>::copyFromCircle(
            results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bField), tkid);
        results_view[tkid].pt() = float(bField) / float(std::abs(circle.par(2)));
        results_view[tkid].eta() = alpaka::math::asinh(acc, line.par(0));
        results_view[tkid].chi2() = (circle.chi2 + line.chi2) / (2 * N - 5);

#ifdef BROKENLINE_DEBUG
        if (!(circle.chi2 >= 0) || !(line.chi2 >= 0))
          printf("kernelBLFit failed! %f/%f\n", circle.chi2, line.chi2);
        printf("kernelBLFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
               N,
               N,
               tkid,
               circle.par(0),
               circle.par(1),
               circle.par(2));
        printf("kernelBLHits line.par(0,1): %d %f,%f\n", tkid, line.par(0), line.par(1));
        printf("kernelBLHits chi2 cov %f/%f  %e,%e,%e,%e,%e\n",
               circle.chi2,
               line.chi2,
               circle.cov(0, 0),
               circle.cov(1, 1),
               circle.cov(2, 2),
               line.cov(0, 0),
               line.cov(1, 1));
#endif
      }
    }
  };

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchBrokenLineKernels(
      const TrackingRecHitSoAConstView<TrackerTraits> &hv,
      // pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const *cpeParams,
      const FrameSoAConstView &fr,
      uint32_t hitsInFit,
      uint32_t maxNumberOfTuples,
      Queue &queue) {
    ALPAKA_ASSERT_ACC(tuples_);

    uint32_t blockSize = 64;
    uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(maxNumberOfConcurrentFits_, blockSize);
    const WorkDiv1D workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const WorkDiv1D workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    //  Fit internals
    auto tkidDevice =
        cms::alpakatools::make_device_buffer<typename TrackerTraits::tindex_type[]>(queue, maxNumberOfConcurrentFits_);
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<6>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<6>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // fit triplets

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_BLFastFit<3, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          hv,
                          fr,
                          tkidDevice.data(),
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          3,
                          3,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_BLFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          bField_,
                          outputSoa_,
                          tkidDevice.data(),
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data());

      if (fitNas4_) {
        // fit all as 4
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrack, 1>([this,
                                                                       &hv,
                                                                       &fr,
                                                                       &tkidDevice,
                                                                       &hitsDevice,
                                                                       &hits_geDevice,
                                                                       &fast_fit_resultsDevice,
                                                                       &offset,
                                                                       &queue,
                                                                       &workDivQuadsPenta](auto i) {
          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFastFit<4, TrackerTraits>{},
                              tuples_,
                              tupleMultiplicity_,
                              hv,
                              fr,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data(),
                              4,
                              4,
                              offset);

          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFit<4, TrackerTraits>{},
                              tupleMultiplicity_,
                              bField_,
                              outputSoa_,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data());
        });

      } else {
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrackForFullFit, 1>([this,
                                                                                 &hv,
                                                                                 &fr,
                                                                                 &tkidDevice,
                                                                                 &hitsDevice,
                                                                                 &hits_geDevice,
                                                                                 &fast_fit_resultsDevice,
                                                                                 &offset,
                                                                                 &queue,
                                                                                 &workDivQuadsPenta](auto i) {
          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFastFit<i, TrackerTraits>{},
                              tuples_,
                              tupleMultiplicity_,
                              hv,
                              fr,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data(),
                              i,
                              i,
                              offset);

          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFit<i, TrackerTraits>{},
                              tupleMultiplicity_,
                              bField_,
                              outputSoa_,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data());
        });

        static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);

        //Fit all the rest using the maximum from previous call
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFastFit<TrackerTraits::maxHitsOnTrackForFullFit, TrackerTraits>{},
                            tuples_,
                            tupleMultiplicity_,
                            hv,
                            fr,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            TrackerTraits::maxHitsOnTrackForFullFit,
                            TrackerTraits::maxHitsOnTrack - 1,
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFit<TrackerTraits::maxHitsOnTrackForFullFit, TrackerTraits>{},
                            tupleMultiplicity_,
                            bField_,
                            outputSoa_,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data());
      }

    }  // loop on concurrent fits
  }

  template class HelixFit<pixelTopology::Phase1>;
  template class HelixFit<pixelTopology::Phase2>;
  template class HelixFit<pixelTopology::HIonPhase1>;
  template class HelixFit<pixelTopology::Phase1Strip>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
