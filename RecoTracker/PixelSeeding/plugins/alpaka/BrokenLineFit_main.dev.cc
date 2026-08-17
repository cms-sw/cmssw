// Split-build TU: main-fit runMainBin<N,T> for the 4 non-stubs traits, N=3..6. Phase2OTStubs (N=3..10)
// lives in BrokenLineFit_stubsMainBL.dev.cc. Each (N,traits) in exactly one TU (ODR).
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_MAIN_SIG_BL(3, Phase1);
  BLFIT_MAIN_SIG_BL(4, Phase1);
  BLFIT_MAIN_SIG_BL(5, Phase1);
  BLFIT_MAIN_SIG_BL(6, Phase1);

  BLFIT_MAIN_SIG_BL(3, Phase2);
  BLFIT_MAIN_SIG_BL(4, Phase2);
  BLFIT_MAIN_SIG_BL(5, Phase2);
  BLFIT_MAIN_SIG_BL(6, Phase2);

  BLFIT_MAIN_SIG_BL(3, Phase2OT);
  BLFIT_MAIN_SIG_BL(4, Phase2OT);
  BLFIT_MAIN_SIG_BL(5, Phase2OT);
  BLFIT_MAIN_SIG_BL(6, Phase2OT);

  BLFIT_MAIN_SIG_BL(3, HIonPhase1);
  BLFIT_MAIN_SIG_BL(4, HIonPhase1);
  BLFIT_MAIN_SIG_BL(5, HIonPhase1);
  BLFIT_MAIN_SIG_BL(6, HIonPhase1);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
