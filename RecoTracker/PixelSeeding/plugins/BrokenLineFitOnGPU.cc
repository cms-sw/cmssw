#include "BrokenLineFitOnGPU.h"

template <typename TrackerTraits>
void HelixFitOnGPU<TrackerTraits>::launchBrokenLineKernelsOnCPU(const TrackingRecHitSoAConstView<TrackerTraits> &hv,
                                                                uint32_t hitsInFit,
                                                                uint32_t maxNumberOfTuples) {
  assert(tuples_);

#ifdef BROKENLINE_DEBUG
  setlinebuf(stdout);
#endif

  //  Fit internals
  auto tkidGPU = std::make_unique<typename TrackerTraits::tindex_type[]>(maxNumberOfConcurrentFits_);
  auto hitsGPU =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<6>) / sizeof(double));
  auto hits_geGPU =
      std::make_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<6>) / sizeof(float));
  auto fast_fit_resultsGPU =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernel_BLFastFit<3, TrackerTraits>(tuples_,
                                       tupleMultiplicity_,
                                       hv,
                                       tkidGPU.get(),
                                       hitsGPU.get(),
                                       hits_geGPU.get(),
                                       fast_fit_resultsGPU.get(),
                                       3,
                                       3,
                                       offset);

    kernel_BLFit<3, TrackerTraits>(tupleMultiplicity_,
                                   bField_,
                                   outputSoa_,
                                   tkidGPU.get(),
                                   hitsGPU.get(),
                                   hits_geGPU.get(),
                                   fast_fit_resultsGPU.get());
    if (fitNas4_) {
      riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrack, 1>(
          [this, &hv, &tkidGPU, &hitsGPU, &hits_geGPU, &fast_fit_resultsGPU, &offset](auto i) {
            kernel_BLFastFit<4, TrackerTraits>(tuples_,
                                               tupleMultiplicity_,
                                               hv,
                                               tkidGPU.get(),
                                               hitsGPU.get(),
                                               hits_geGPU.get(),
                                               fast_fit_resultsGPU.get(),
                                               4,
                                               i,
                                               offset);

            kernel_BLFit<4, TrackerTraits>(tupleMultiplicity_,
                                           bField_,
                                           outputSoa_,
                                           tkidGPU.get(),
                                           hitsGPU.get(),
                                           hits_geGPU.get(),
                                           fast_fit_resultsGPU.get());
          });

    } else {
      //Fit these using all the hits they have
      riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrackForFullFit, 1>(
          [this, &hv, &tkidGPU, &hitsGPU, &hits_geGPU, &fast_fit_resultsGPU, &offset](auto i) {
            kernel_BLFastFit<i, TrackerTraits>(tuples_,
                                               tupleMultiplicity_,
                                               hv,
                                               tkidGPU.get(),
                                               hitsGPU.get(),
                                               hits_geGPU.get(),
                                               fast_fit_resultsGPU.get(),
                                               i,
                                               i,
                                               offset);

            kernel_BLFit<i, TrackerTraits>(tupleMultiplicity_,
                                           bField_,
                                           outputSoa_,
                                           tkidGPU.get(),
                                           hitsGPU.get(),
                                           hits_geGPU.get(),
                                           fast_fit_resultsGPU.get());
          });

      static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);

      //Fit all the rest using the maximum from previous call

      kernel_BLFastFit<TrackerTraits::maxHitsOnTrackForFullFit, TrackerTraits>(tuples_,
                                                                               tupleMultiplicity_,
                                                                               hv,
                                                                               tkidGPU.get(),
                                                                               hitsGPU.get(),
                                                                               hits_geGPU.get(),
                                                                               fast_fit_resultsGPU.get(),
                                                                               TrackerTraits::maxHitsOnTrackForFullFit,
                                                                               TrackerTraits::maxHitsOnTrack - 1,
                                                                               offset);

      kernel_BLFit<TrackerTraits::maxHitsOnTrackForFullFit, TrackerTraits>(tupleMultiplicity_,
                                                                           bField_,
                                                                           outputSoa_,
                                                                           tkidGPU.get(),
                                                                           hitsGPU.get(),
                                                                           hits_geGPU.get(),
                                                                           fast_fit_resultsGPU.get());
    }

  }  // loop on concurrent fits
}

template class HelixFitOnGPU<pixelTopology::Phase1>;
template class HelixFitOnGPU<pixelTopology::Phase2>;
