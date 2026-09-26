#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BLMaterialMapCollection_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BLMaterialMapCollection_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Serial/CPU backend -> BLMaterialMapHost (data() = host buffer); GPU backend -> BLMaterialMapDevice<Device>.
  // The material map is TrackerTraits-independent (single global grid), hence non-templated.
  using BLMaterialMap =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, BLMaterialMapHost, BLMaterialMapDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<BLMaterialMapHost> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, BLMaterialMapHost const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      BLMaterialMapDevice<TDevice> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BLMaterialMapCollection_h
