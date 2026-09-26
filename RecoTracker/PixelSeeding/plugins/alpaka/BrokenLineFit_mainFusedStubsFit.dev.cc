// Split-build TU: fused CA main fit, fit phase, Phase2OTStubs (N=3..10) -- the heaviest fused
// instantiation (pulls the eight factorized circle+line solvers). Phases split per mainFusedStubsFast.
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_MAIN_FUSED_FIT_SIG(Phase2OTStubs);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
