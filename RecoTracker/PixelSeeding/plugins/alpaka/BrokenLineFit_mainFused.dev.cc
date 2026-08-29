// Fused CA main fit, the four non-stubs traits (N=3..6), both phases. Each (phase, traits) pair
// lives in exactly one translation unit.
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_MAIN_FUSED_FAST_SIG(Phase1);
  BLFIT_MAIN_FUSED_FAST_SIG(Phase2);
  BLFIT_MAIN_FUSED_FAST_SIG(Phase2OT);
  BLFIT_MAIN_FUSED_FAST_SIG(HIonPhase1);

  BLFIT_MAIN_FUSED_FIT_SIG(Phase1);
  BLFIT_MAIN_FUSED_FIT_SIG(Phase2);
  BLFIT_MAIN_FUSED_FIT_SIG(Phase2OT);
  BLFIT_MAIN_FUSED_FIT_SIG(HIonPhase1);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
