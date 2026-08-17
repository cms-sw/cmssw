#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapDevice_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapDevice_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

// Device-resident BL-fit material map rho(r,z) [X0/cm] (kSize floats), filled once per IOV from the host
// payload.
template <typename TDev>
class BLMaterialMapDevice {
public:
  using Buffer = cms::alpakatools::device_buffer<TDev, float[]>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, float[]>;

  template <typename TQueue>
  explicit BLMaterialMapDevice(TQueue queue)
      : buffer_(cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kSize)) {}

  // non-copyable
  BLMaterialMapDevice(BLMaterialMapDevice const&) = delete;
  BLMaterialMapDevice& operator=(BLMaterialMapDevice const&) = delete;

  // movable
  BLMaterialMapDevice(BLMaterialMapDevice&&) = default;
  BLMaterialMapDevice& operator=(BLMaterialMapDevice&&) = default;

  ~BLMaterialMapDevice() = default;

  Buffer buffer() { return buffer_; }

  // raw device pointer passed to rhoAt() by the BL fit kernel
  float const* data() const { return buffer_.data(); }

private:
  Buffer buffer_;
};

#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapDevice_h
