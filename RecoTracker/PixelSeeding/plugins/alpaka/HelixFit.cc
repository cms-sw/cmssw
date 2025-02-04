#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HelixFit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::allocate(TupleMultiplicity const *tupleMultiplicity, OutputSoAView &helix_fit_results) {
    tuples_ = &helix_fit_results.hitIndices();
    tupleMultiplicity_ = tupleMultiplicity;
    outputSoa_ = helix_fit_results;

    ALPAKA_ASSERT_ACC(tuples_);
    ALPAKA_ASSERT_ACC(tupleMultiplicity_);
  }

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::deallocate() {}

  template class HelixFit<pixelTopology::Phase1>;
  template class HelixFit<pixelTopology::Phase2>;
  template class HelixFit<pixelTopology::HIonPhase1>;
  template class HelixFit<pixelTopology::Phase1Strip>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
