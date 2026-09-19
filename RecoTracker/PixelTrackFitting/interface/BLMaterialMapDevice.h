#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapDevice_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapDevice_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

// Device-resident BL-fit material map (one blMaterialMap::Map): per cell the density rho(r,z) [X0/cm] and
// the dE/dx triple, filled once per IOV from the host payload.
template <typename TDev>
class BLMaterialMapDevice {
public:
  using Buffer = cms::alpakatools::device_buffer<TDev, blMaterialMap::Map>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, blMaterialMap::Map>;

  template <typename TQueue>
  explicit BLMaterialMapDevice(TQueue queue)
      : buffer_(cms::alpakatools::make_device_buffer<blMaterialMap::Map>(queue)) {}

  // non-copyable
  BLMaterialMapDevice(BLMaterialMapDevice const&) = delete;
  BLMaterialMapDevice& operator=(BLMaterialMapDevice const&) = delete;

  // movable
  BLMaterialMapDevice(BLMaterialMapDevice&&) = default;
  BLMaterialMapDevice& operator=(BLMaterialMapDevice&&) = default;

  ~BLMaterialMapDevice() = default;

  Buffer buffer() { return buffer_; }

  // device pointer to the map, read by rhoAt() and dedxAt() in the BL fit kernels
  blMaterialMap::Map const* data() const { return buffer_.data(); }

private:
  Buffer buffer_;
};

#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapDevice_h
