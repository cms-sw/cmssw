#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAMaskKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAMaskKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::caMasking {

  using MapToHit = reco::TrackingRecHitsMaskingSoACollection;
  using MapToHitView = MapToHit::View;
  using MapToHitConstView = MapToHit::ConstView;
  using TkSoADevice = reco::TracksSoACollection;

  void makeMaskingAsync(Queue& queue, MapToHit& outMask, TkSoADevice const& tracks_d, pixelTrack::Quality minQuality);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caMasking

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAMaskKernels_h
