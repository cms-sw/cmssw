#include "RiemannFitOnGPU.h"

template <typename TrackerTraits>
void HelixFitOnGPU<TrackerTraits>::launchRiemannKernelsOnCPU(const TrackingRecHitSoAConstView<TrackerTraits> &hv,
                                                             uint32_t nhits,
                                                             uint32_t maxNumberOfTuples) {
  assert(tuples_);

  //  Fit internals
  auto hitsGPU =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<4>) / sizeof(double));
  auto hits_geGPU =
      std::make_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6x4f) / sizeof(float));
  auto fast_fit_resultsGPU =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));
  auto circle_fit_resultsGPU_holder =
      std::make_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::CircleFit));
  riemannFit::CircleFit *circle_fit_resultsGPU = (riemannFit::CircleFit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernel_FastFit<3, TrackerTraits>(
        tuples_, tupleMultiplicity_, 3, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

    kernel_CircleFit<3, TrackerTraits>(tupleMultiplicity_,
                                       3,
                                       bField_,
                                       hitsGPU.get(),
                                       hits_geGPU.get(),
                                       fast_fit_resultsGPU.get(),
                                       circle_fit_resultsGPU,
                                       offset);

    kernel_LineFit<3, TrackerTraits>(tupleMultiplicity_,
                                     3,
                                     bField_,
                                     outputSoa_,
                                     hitsGPU.get(),
                                     hits_geGPU.get(),
                                     fast_fit_resultsGPU.get(),
                                     circle_fit_resultsGPU,
                                     offset);

    // quads
    kernel_FastFit<4, TrackerTraits>(
        tuples_, tupleMultiplicity_, 4, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

    kernel_CircleFit<4, TrackerTraits>(tupleMultiplicity_,
                                       4,
                                       bField_,
                                       hitsGPU.get(),
                                       hits_geGPU.get(),
                                       fast_fit_resultsGPU.get(),
                                       circle_fit_resultsGPU,
                                       offset);

    kernel_LineFit<4, TrackerTraits>(tupleMultiplicity_,
                                     4,
                                     bField_,
                                     outputSoa_,
                                     hitsGPU.get(),
                                     hits_geGPU.get(),
                                     fast_fit_resultsGPU.get(),
                                     circle_fit_resultsGPU,
                                     offset);

    if (fitNas4_) {
      // penta
      kernel_FastFit<4, TrackerTraits>(
          tuples_, tupleMultiplicity_, 5, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

      kernel_CircleFit<4, TrackerTraits>(tupleMultiplicity_,
                                         5,
                                         bField_,
                                         hitsGPU.get(),
                                         hits_geGPU.get(),
                                         fast_fit_resultsGPU.get(),
                                         circle_fit_resultsGPU,
                                         offset);

      kernel_LineFit<4, TrackerTraits>(tupleMultiplicity_,
                                       5,
                                       bField_,
                                       outputSoa_,
                                       hitsGPU.get(),
                                       hits_geGPU.get(),
                                       fast_fit_resultsGPU.get(),
                                       circle_fit_resultsGPU,
                                       offset);

    } else {
      // penta all 5
      kernel_FastFit<5, TrackerTraits>(
          tuples_, tupleMultiplicity_, 5, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

      kernel_CircleFit<5, TrackerTraits>(tupleMultiplicity_,
                                         5,
                                         bField_,
                                         hitsGPU.get(),
                                         hits_geGPU.get(),
                                         fast_fit_resultsGPU.get(),
                                         circle_fit_resultsGPU,
                                         offset);

      kernel_LineFit<5, TrackerTraits>(tupleMultiplicity_,
                                       5,
                                       bField_,
                                       outputSoa_,
                                       hitsGPU.get(),
                                       hits_geGPU.get(),
                                       fast_fit_resultsGPU.get(),
                                       circle_fit_resultsGPU,
                                       offset);
    }
  }
}

template class HelixFitOnGPU<pixelTopology::Phase1>;
template class HelixFitOnGPU<pixelTopology::Phase2>;
template class HelixFitOnGPU<pixelTopology::HIonPhase1>;
