#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BLBFieldMapCollection_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BLBFieldMapCollection_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMapDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Serial/CPU backend -> BLBFieldMapHost (data() = host buffer); GPU backend -> BLBFieldMapDevice<Device>.
  // The field map is a single global lattice (TrackerTraits-independent), hence non-templated.
  using BLBFieldMap =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, BLBFieldMapHost, BLBFieldMapDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<BLBFieldMapHost> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, BLBFieldMapHost const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      BLBFieldMapDevice<TDevice> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BLBFieldMapCollection_h
