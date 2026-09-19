#ifndef RecoTracker_PixelSeeding_plugins_alpaka_RiemannFit_h
#define RecoTracker_PixelSeeding_plugins_alpaka_RiemannFit_h

// #define FIT_DEBUG
#include <cstdint>

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
#include <iostream>
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/RiemannFit.h"

#include "HelixFit.h"
#include "CAStructures.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace alpaka;
  using namespace cms::alpakatools;

  template <int N, typename TrackerTraits>
  class Kernel_FastFit {
  public:
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  Tuples const *__restrict__ foundNtuplets,
                                  TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  uint32_t nHits,
                                  HitsMultiView hh,
                                  ::reco::CAModulesConstView cm,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit,
                                  uint32_t offset) const {
      constexpr uint32_t hitsInFit = N;

      ALPAKA_ASSERT_ACC(hitsInFit <= nHits);

      ALPAKA_ASSERT_ACC(pfast_fit);
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);

      // look in bin for this hit multiplicity

#ifdef RIEMANN_DEBUG
      const uint32_t threadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      if (cms::alpakatools::once_per_grid(acc))
        printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
#endif

      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          break;

        // get it from the ntuple container (one to one to helix)
        auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
        ALPAKA_ASSERT_ACC(tkid < foundNtuplets->nOnes());

        ALPAKA_ASSERT_ACC(foundNtuplets->size(tkid) == nHits);

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        // Prepare data structure
        auto const *hitId = foundNtuplets->begin(tkid);
        for (unsigned int i = 0; i < hitsInFit; ++i) {
          auto hit = hitId[i];
          float ge[6];

          cm.innerSensorFrame(hh[hit].detectorIndex()).toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge);

          hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        riemannFit::fastFit(acc, hits, fast_fit);

#ifdef RIEMANN_DEBUG
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
  class Kernel_CircleFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  uint32_t nHits,
                                  double bField,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit_input,
                                  riemannFit::CircleFit *circle_fit,
                                  uint32_t offset) const {
      ALPAKA_ASSERT_ACC(circle_fit);
      ALPAKA_ASSERT_ACC(N <= nHits);

      // same as above...

      // look in bin for this hit multiplicity
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          break;

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit_input + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        riemannFit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

        riemannFit::Matrix2Nd<N> hits_cov = riemannFit::Matrix2Nd<N>::Zero();
        riemannFit::loadCovariance2D(acc, hits_ge, hits_cov);

        circle_fit[local_idx] =
            riemannFit::circleFit(acc, hits.block(0, 0, 2, N), hits_cov, fast_fit, rad, bField, true);

#ifdef RIEMANN_DEBUG
//    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
//  printf("kernelCircleFit circle.par(0,1,2): %d %f,%f,%f\n", tkid,
//         circle_fit[local_idx].par(0), circle_fit[local_idx].par(1), circle_fit[local_idx].par(2));
#endif
      }
    }
  };

  template <int N, typename TrackerTraits>
  class Kernel_LineFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  uint32_t nHits,
                                  double bField,
                                  OutputSoAView results_view,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit_input,
                                  riemannFit::CircleFit *__restrict__ circle_fit,
                                  uint32_t offset) const {
      ALPAKA_ASSERT_ACC(circle_fit);
      ALPAKA_ASSERT_ACC(N <= nHits);

      // same as above...

      // look in bin for this hit multiplicity
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          break;

        // get it for the ntuple container (one to one to helix)
        int32_t tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit_input + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        auto const &line_fit = riemannFit::lineFit(acc, hits, hits_ge, circle_fit[local_idx], fast_fit, bField, true);

        riemannFit::fromCircleToPerigee(acc, circle_fit[local_idx]);

        reco::copyFromCircle(results_view,
                             circle_fit[local_idx].par,
                             circle_fit[local_idx].cov,
                             line_fit.par,
                             line_fit.cov,
                             1.f / float(bField),
                             tkid);
        results_view[tkid].pt() = bField / std::abs(circle_fit[local_idx].par(2));
        results_view[tkid].eta() = asinhf(line_fit.par(0));
        results_view[tkid].chi2() = (circle_fit[local_idx].chi2 + line_fit.chi2) / (2 * N - 5);

#ifdef RIEMANN_DEBUG
        printf("kernelLineFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
               N,
               nHits,
               tkid,
               circle_fit[local_idx].par(0),
               circle_fit[local_idx].par(1),
               circle_fit[local_idx].par(2));
        printf("kernelLineFit line.par(0,1): %d %f,%f\n", tkid, line_fit.par(0), line_fit.par(1));
        printf("kernelLineFit chi2 cov %f/%f %e,%e,%e,%e,%e\n",
               circle_fit[local_idx].chi2,
               line_fit.chi2,
               circle_fit[local_idx].cov(0, 0),
               circle_fit[local_idx].cov(1, 1),
               circle_fit[local_idx].cov(2, 2),
               line_fit.cov(0, 0),
               line_fit.cov(1, 1));
#endif
      }
    }
  };

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchRiemannKernels(const HitsMultiView &hv,
                                                     const ::reco::CAModulesConstView &cm,
                                                     uint32_t nhits,
                                                     uint32_t maxNumberOfTuples,
                                                     Queue &queue) {
    assert(tuples_);

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
    alpaka::wait(queue);
    std::cout << "FIT_DEBUG: launchRiemannKernels (stubs) - maxNumberOfTuples=" << maxNumberOfTuples
              << " maxNumberOfConcurrentFits=" << maxNumberOfConcurrentFits_ << " bField=" << bField_ << std::endl;
    std::cout << "FIT_DEBUG: TrackerTraits::maxHitsOnTrack=" << TrackerTraits::maxHitsOnTrack << std::endl;
#endif

    auto blockSize = 64;
    auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;
    const auto workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const auto workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    // maxN is the Riemann fit's own hit cap, decoupled from maxHitsOnTrackForFullFit: it sizes the
    // hit/geometry buffers below and is the largest N at which the Riemann kernels are instantiated.
    // Tracks with more hits are fit with their first maxN hits.
    constexpr auto maxN = TrackerTraits::maxHitsOnTrackForRiemannFit;
    static_assert(3 <= TrackerTraits::maxHitsOnTrackForRiemannFit &&
                      TrackerTraits::maxHitsOnTrackForRiemannFit <= TrackerTraits::maxHitsOnTrack,
                  "maxHitsOnTrackForRiemannFit must be a valid tuple multiplicity: >= the 3-hit minimum fit and "
                  "<= the largest multiplicity a tuple can have");
    // Invariant of the ladder below: multiplicities 3, 4 and 5 are fit exactly, the cap bin and every
    // larger multiplicity with the first maxN hits. That covers every populated bin only for a cap of 6.
    static_assert(maxN == 6,
                  "the explicit 3/4/5-hit ladder below leaves bins 6..maxN-1 unfitted for a cap != 6: "
                  "generalize the ladder before changing maxHitsOnTrackForRiemannFit");
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));
    auto circle_fit_resultsDevice_holder =
        cms::alpakatools::make_device_buffer<char[]>(queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::CircleFit));
    riemannFit::CircleFit *circle_fit_resultsDevice_ =
        (riemannFit::CircleFit *)(circle_fit_resultsDevice_holder.data());

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_FastFit<3, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          3,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_CircleFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          3,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_LineFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          3,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      // quadruplets
      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_FastFit<4, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          4,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_CircleFit<4, TrackerTraits>{},
                          tupleMultiplicity_,
                          4,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_LineFit<4, TrackerTraits>{},
                          tupleMultiplicity_,
                          4,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      // quintuplets
      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_FastFit<5, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          5,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_CircleFit<5, TrackerTraits>{},
                          tupleMultiplicity_,
                          5,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_LineFit<5, TrackerTraits>{},
                          tupleMultiplicity_,
                          5,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      // the cap bin (maxN hits) - fit all maxN hits
      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_FastFit<maxN, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          maxN,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_CircleFit<maxN, TrackerTraits>{},
                          tupleMultiplicity_,
                          maxN,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_LineFit<maxN, TrackerTraits>{},
                          tupleMultiplicity_,
                          maxN,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      // Above the cap: fit the first maxN hits. Bin index range maxN+1 .. maxHitsOnTrack-1, the same
      // upper bound the BrokenLine ladder uses.
      for (uint32_t nHits = maxN + 1; nHits < TrackerTraits::maxHitsOnTrack; ++nHits) {
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_FastFit<maxN, TrackerTraits>{},
                            tuples_,
                            tupleMultiplicity_,
                            nHits,
                            hv,
                            cm,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_CircleFit<maxN, TrackerTraits>{},
                            tupleMultiplicity_,
                            nHits,
                            bField_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_,
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_LineFit<maxN, TrackerTraits>{},
                            tupleMultiplicity_,
                            nHits,
                            bField_,
                            outputSoa_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_,
                            offset);
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_RiemannFit_h
