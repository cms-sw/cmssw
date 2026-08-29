#ifndef RecoTracker_PixelTrackFitting_interface_BLBFieldMapDevice_h
#define RecoTracker_PixelTrackFitting_interface_BLBFieldMapDevice_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"

// Device-resident normalized (Bz,Br) r-z lattice (kNValues floats); filled once per IOV from the host
// (CopyToDevice in interface/alpaka/BLBFieldMapCollection.h).
template <typename TDev>
class BLBFieldMapDevice {
public:
  using Buffer = cms::alpakatools::device_buffer<TDev, float[]>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, float[]>;

  template <typename TQueue>
  BLBFieldMapDevice(TQueue queue)
      : buffer_(cms::alpakatools::make_device_buffer<float[]>(queue, blBFieldMap::kNValues)) {}

  BLBFieldMapDevice(BLBFieldMapDevice const&) = delete;
  BLBFieldMapDevice& operator=(BLBFieldMapDevice const&) = delete;

  BLBFieldMapDevice(BLBFieldMapDevice&&) = default;
  BLBFieldMapDevice& operator=(BLBFieldMapDevice&&) = default;

  ~BLBFieldMapDevice() = default;

  Buffer buffer() { return buffer_; }

  // the raw device pointer consumed by the BL fit kernels (passed to blBFieldMap::bBendAt())
  float const* data() const { return buffer_.data(); }

private:
  Buffer buffer_;
};

#endif  // RecoTracker_PixelTrackFitting_interface_BLBFieldMapDevice_h
