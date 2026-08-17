#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATrackDNNWeights_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATrackDNNWeights_h

// Loose->tight (Stage-1) DNN weight bank, auto-generated into its own header
// (CATrackDNNWeights_prompt.h : namespace caTrackDNN_prompt) so a bake run never clobbers
// hand-written code. `caTrackDNN` is the symbol path used by the kernels for the bank-free
// constants (kNFeat / kDefaultThreshold, e.g. the caTrackFeatures::kNFeat ABI static_assert in
// Kernel_classifyTracks).

#include "CATrackDNNWeights_prompt.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace caTrackDNN = caTrackDNN_prompt;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATrackDNNWeights_h
