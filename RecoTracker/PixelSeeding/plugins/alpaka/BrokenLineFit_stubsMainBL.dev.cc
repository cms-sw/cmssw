// Split-build TU: CA main fit runMainBin<N, Phase2OTStubs>, N=3..10 -- the factorized fast BrokenLine fit.
// Kept separate from BrokenLineFit_main.dev.cc (non-stubs traits) so nvcc parallelizes the two N-ranges.
// Pulls Kernel_BLFastFit<N> (shared) + Kernel_BLFit<N,...> (factorized circle+line fit).
// Each (N,traits) is instantiated in exactly one TU (ODR).
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_MAIN_SIG_BL(3, Phase2OTStubs);
  BLFIT_MAIN_SIG_BL(4, Phase2OTStubs);
  BLFIT_MAIN_SIG_BL(5, Phase2OTStubs);
  BLFIT_MAIN_SIG_BL(6, Phase2OTStubs);
  BLFIT_MAIN_SIG_BL(7, Phase2OTStubs);
  BLFIT_MAIN_SIG_BL(8, Phase2OTStubs);
  BLFIT_MAIN_SIG_BL(9, Phase2OTStubs);
  BLFIT_MAIN_SIG_BL(10, Phase2OTStubs);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
