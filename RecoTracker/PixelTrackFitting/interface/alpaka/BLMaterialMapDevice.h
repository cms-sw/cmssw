#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BLMaterialMapDevice_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BLMaterialMapDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // The material map of the BrokenLine fit in device memory, copied from BLMaterialMapHost once per IOV.
  using BLMaterialMapDevice = PortableObject<blMaterialMap::Map>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(BLMaterialMapDevice, BLMaterialMapHost);

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BLMaterialMapDevice_h
