#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h

#include <algorithm>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

// Host-resident BL-fit Geant4 material map rho(r,z) [X0/cm] (kSize floats, 0.5 cm radial lattice). Built
// once from the compiled-in D121 table (blMaterialMapDataD121()), not from the conditions DB; copied to the
// device once per IOV (CopyToDevice in interface/alpaka/BLMaterialMapCollection.h). See BLMaterialMap.h.
class BLMaterialMapHost {
public:
  using Buffer = cms::alpakatools::host_buffer<float[]>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<float[]>;

  BLMaterialMapHost() : buffer_(cms::alpakatools::make_host_buffer<float[]>(blMaterialMap::kSize)) {
    std::copy_n(blMaterialMap::blMaterialMapDataD121(), blMaterialMap::kSize, buffer_.data());
  }

  // non-copyable
  BLMaterialMapHost(BLMaterialMapHost const&) = delete;
  BLMaterialMapHost& operator=(BLMaterialMapHost const&) = delete;

  // movable (required by the ESProducer ownership / CopyToDevice machinery)
  BLMaterialMapHost(BLMaterialMapHost&&) = default;
  BLMaterialMapHost& operator=(BLMaterialMapHost&&) = default;

  ~BLMaterialMapHost() = default;

  // access the buffer (the CopyToDevice source)
  ConstBuffer buffer() const { return buffer_; }

  float const* data() const { return buffer_.data(); }

private:
  Buffer buffer_;
};

#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
