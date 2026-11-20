#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BLBFieldMapDevice_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BLBFieldMapDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMapHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // The (Bz, Br) lattice of the BrokenLine fit in device memory, copied from BLBFieldMapHost once per IOV.
  using BLBFieldMapDevice = PortableObject<blBFieldMap::Map>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(BLBFieldMapDevice, BLBFieldMapHost);

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BLBFieldMapDevice_h
