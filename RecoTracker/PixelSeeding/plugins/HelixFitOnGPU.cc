#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HelixFitOnGPU.h"

template <typename TrackerTraits>
void HelixFitOnGPU<TrackerTraits>::allocateOnGPU(TupleMultiplicity const *tupleMultiplicity,
                                                 OutputSoAView &helix_fit_results) {
  tuples_ = &helix_fit_results.hitIndices();
  tupleMultiplicity_ = tupleMultiplicity;
  outputSoa_ = helix_fit_results;

  assert(tuples_);
  assert(tupleMultiplicity_);
  assert(outputSoa_.chi2());
  assert(outputSoa_.pt());
}

template <typename TrackerTraits>
void HelixFitOnGPU<TrackerTraits>::deallocateOnGPU() {}

template class HelixFitOnGPU<pixelTopology::Phase1>;
template class HelixFitOnGPU<pixelTopology::Phase2>;
