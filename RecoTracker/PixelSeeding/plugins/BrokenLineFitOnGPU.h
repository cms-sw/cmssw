//
// Author: Felice Pantaleo, CERN
//

//#define BROKENLINE_DEBUG
//#define BL_DUMP_HITS
#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"

#include "HelixFitOnGPU.h"

template <typename TrackerTraits>
using Tuples = typename TrackSoA<TrackerTraits>::HitContainer;
template <typename TrackerTraits>
using OutputSoAView = TrackSoAView<TrackerTraits>;
template <typename TrackerTraits>
using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

// #define BL_DUMP_HITS

template <int N, typename TrackerTraits>
__global__ void kernel_BLFastFit(Tuples<TrackerTraits> const *__restrict__ foundNtuplets,
                                 TupleMultiplicity<TrackerTraits> const *__restrict__ tupleMultiplicity,
                                 TrackingRecHitSoAConstView<TrackerTraits> hh,
                                 typename TrackerTraits::tindex_type *__restrict__ ptkids,
                                 double *__restrict__ phits,
                                 float *__restrict__ phits_ge,
                                 double *__restrict__ pfast_fit,
                                 uint32_t nHitsL,
                                 uint32_t nHitsH,
                                 int32_t offset) {
  constexpr uint32_t hitsInFit = N;
  constexpr auto invalidTkId = std::numeric_limits<typename TrackerTraits::tindex_type>::max();

  assert(hitsInFit <= nHitsL);
  assert(nHitsL <= nHitsH);
  assert(phits);
  assert(pfast_fit);
  assert(foundNtuplets);
  assert(tupleMultiplicity);

  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;
  int totTK = tupleMultiplicity->end(nHitsH) - tupleMultiplicity->begin(nHitsL);
  assert(totTK <= int(tupleMultiplicity->size()));
  assert(totTK >= 0);

#ifdef BROKENLINE_DEBUG
  if (0 == local_start) {
    printf("%d total Ntuple\n", tupleMultiplicity->size());
    printf("%d Ntuple of size %d/%d for %d hits to fit\n", totTK, nHitsL, nHitsH, hitsInFit);
  }
#endif

  for (int local_idx = local_start, nt = riemannFit::maxNumberOfConcurrentFits; local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    int tuple_idx = local_idx + offset;
    if (tuple_idx >= totTK) {
      ptkids[local_idx] = invalidTkId;
      break;
    }
    // get it from the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHitsL) + tuple_idx);
    assert(int(tkid) < foundNtuplets->nOnes());

    ptkids[local_idx] = tkid;

    auto nHits = foundNtuplets->size(tkid);

    assert(nHits >= nHitsL);
    assert(nHits <= nHitsH);

    riemannFit::Map3xNd<N> hits(phits + local_idx);
    riemannFit::Map4d fast_fit(pfast_fit + local_idx);
    riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

#ifdef BL_DUMP_HITS
    __shared__ int done;
    done = 0;
    __syncthreads();
    bool dump = (foundNtuplets->size(tkid) == 5 && 0 == atomicAdd(&done, 1));
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
      assert(j < int(nHits));
      n += incr;
      auto hit = hitId[j];
      float ge[6];

#ifdef YERR_FROM_DC
      auto const &dp = hh.cpeParams().detParams(hh.detectorIndex(hit));
      auto status = hh[hit].chargeAndStatus().status;
      int qbin = CPEFastParametrisation::kGenErrorQBins - 1 - status.qBin;
      assert(qbin >= 0 && qbin < 5);
      bool nok = (status.isBigY | status.isOneY);
      // compute cotanbeta and use it to recompute error
      dp.frame.rotation().multiply(dx, dy, dz, ux, uy, uz);
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
      dp.frame.toGlobal(hh[hit].xerrLocal(), 0, yerr, ge);
#else
      hh.cpeParams().detParams(hh[hit].detectorIndex()).frame.toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge);
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
    brokenline::fastFit(hits, fast_fit);

    // no NaN here....
    assert(fast_fit(0) == fast_fit(0));
    assert(fast_fit(1) == fast_fit(1));
    assert(fast_fit(2) == fast_fit(2));
    assert(fast_fit(3) == fast_fit(3));
  }
}

template <int N, typename TrackerTraits>
__global__ void kernel_BLFit(TupleMultiplicity<TrackerTraits> const *__restrict__ tupleMultiplicity,
                             double bField,
                             OutputSoAView<TrackerTraits> results_view,
                             typename TrackerTraits::tindex_type const *__restrict__ ptkids,
                             double *__restrict__ phits,
                             float *__restrict__ phits_ge,
                             double *__restrict__ pfast_fit) {
  assert(results_view.pt());
  assert(results_view.eta());
  assert(results_view.chi2());
  assert(pfast_fit);
  constexpr auto invalidTkId = std::numeric_limits<typename TrackerTraits::tindex_type>::max();

  // same as above...
  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;
  for (int local_idx = local_start, nt = riemannFit::maxNumberOfConcurrentFits; local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    if (invalidTkId == ptkids[local_idx])
      break;
    auto tkid = ptkids[local_idx];

    assert(tkid < TrackerTraits::maxNumberOfTuples);

    riemannFit::Map3xNd<N> hits(phits + local_idx);
    riemannFit::Map4d fast_fit(pfast_fit + local_idx);
    riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    brokenline::PreparedBrokenLineData<N> data;

    brokenline::karimaki_circle_fit circle;
    riemannFit::LineFit line;

    brokenline::prepareBrokenLineData(hits, fast_fit, bField, data);
    brokenline::lineFit(hits_ge, fast_fit, bField, data, line);
    brokenline::circleFit(hits, hits_ge, fast_fit, bField, data, circle);

    TracksUtilities<TrackerTraits>::copyFromCircle(
        results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bField), tkid);
    results_view[tkid].pt() = float(bField) / float(std::abs(circle.par(2)));
    results_view[tkid].eta() = asinhf(line.par(0));
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
