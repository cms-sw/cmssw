#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h

#include <algorithm>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

// Host-resident BL-fit material map (kBufferFloats floats, 0.5 cm radial lattice): per cell the density
// rho(r,z) [X0/cm] and the dE/dx triple, built from the compiled-in table rather than the conditions DB and
// copied to the device once per IOV.
class BLMaterialMapHost {
public:
  using Buffer = cms::alpakatools::host_buffer<float[]>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<float[]>;

  BLMaterialMapHost() : buffer_(cms::alpakatools::make_host_buffer<float[]>(blMaterialMap::kBufferFloats)) {
    std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kBufferFloats, buffer_.data());
  }

  // non-copyable
  BLMaterialMapHost(BLMaterialMapHost const&) = delete;
  BLMaterialMapHost& operator=(BLMaterialMapHost const&) = delete;

  // movable
  BLMaterialMapHost(BLMaterialMapHost&&) = default;
  BLMaterialMapHost& operator=(BLMaterialMapHost&&) = default;

  ~BLMaterialMapHost() = default;

  ConstBuffer buffer() const { return buffer_; }

  // the whole table: rhoAt() and dedxAt() both read it through this pointer
  float const* data() const { return buffer_.data(); }

private:
  Buffer buffer_;
};

#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
