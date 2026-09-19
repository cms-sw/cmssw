#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

// Host-resident BL-fit material map (one blMaterialMap::Map, 0.5 cm radial lattice): per cell the density
// rho(r,z) [X0/cm] and the dE/dx triple, loaded from the compiled-in table rather than the conditions DB and
// copied to the device once per IOV.
class BLMaterialMapHost {
public:
  using Buffer = cms::alpakatools::host_buffer<blMaterialMap::Map>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<blMaterialMap::Map>;

  BLMaterialMapHost() : buffer_(cms::alpakatools::make_host_buffer<blMaterialMap::Map>()) {
    blMaterialMap::loadTable(*buffer_.data(), blMaterialMap::blMaterialMapData());
  }

  // non-copyable
  BLMaterialMapHost(BLMaterialMapHost const&) = delete;
  BLMaterialMapHost& operator=(BLMaterialMapHost const&) = delete;

  // movable
  BLMaterialMapHost(BLMaterialMapHost&&) = default;
  BLMaterialMapHost& operator=(BLMaterialMapHost&&) = default;

  ~BLMaterialMapHost() = default;

  ConstBuffer buffer() const { return buffer_; }

  // the whole map: rhoAt() and dedxAt() both read it
  blMaterialMap::Map const* data() const { return buffer_.data(); }

private:
  Buffer buffer_;
};

#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
