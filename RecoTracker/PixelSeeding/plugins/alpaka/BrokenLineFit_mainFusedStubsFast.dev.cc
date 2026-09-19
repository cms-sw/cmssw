// Fused CA main fit, fast phase, Phase2OTStubs (N=3..10). One phase per translation unit: a fused
// kernel pulls in all N of its phase, so splitting by phase bounds each nvcc partition.
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_MAIN_FUSED_FAST_SIG(Phase2OTStubs);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
