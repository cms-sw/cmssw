#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h

// Triplet-gate DNN weight bank, auto-generated into its own header
// (CATripletDNNWeights_prompt.h : namespace caTripletDNN_prompt) so a bake run never clobbers
// hand-written code. `caTripletDNN` is the symbol path used by the kernels for the bank-free
// constants (kNFeat / kDefaultThreshold, e.g. the in-kernel feature-vector sizing in
// CATripletCuts.h).

#include "CATripletDNNWeights_prompt.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace caTripletDNN = caTripletDNN_prompt;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h
