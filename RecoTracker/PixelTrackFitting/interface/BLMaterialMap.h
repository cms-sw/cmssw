// Phi-averaged material radiation-length density rho(r,z) [X0/cm] of the Tracker and the beam pipe,
// on a 0.5 cm radial lattice; integrating it along a track segment gives the segment X/X0. One table
// per geometry is compiled in (src/BLMaterialMap<geometry>.cc), and test/blMaterialMap/ regenerates
// it. rhoAt reads the table through the passed pointer: the device fits get it from the EventSetup,
// the unit tests read the compiled-in array directly.
#ifndef RecoTracker_PixelTrackFitting_BLMaterialMap_h
#define RecoTracker_PixelTrackFitting_BLMaterialMap_h
namespace blMaterialMap {
  constexpr int kNR = 250;
  constexpr int kNZ = 560;
  constexpr float kDR = 0.5000f;   // cm
  constexpr float kDZ = 1.0000f;   // cm
  constexpr float kZMAX = 280.0f;  // cm; z in [-kZMAX, +kZMAX]
  constexpr int kSize = kNR * kNZ;
  // Host pointer to the kNZ-major density grid [X0/cm].
  const float* blMaterialMapData();  // kSize values, the compiled-in table (src/BLMaterialMap<geometry>.cc)
  // Local density [X0/cm] at (r,z), 0 outside the grid; rho is a host array or a device buffer.
  // constexpr so it is callable from device code.
  constexpr inline float rhoAt(const float* rho, float r, float z) {
    int ir = int(r / kDR), iz = int((z + kZMAX) / kDZ);
    if (ir < 0 || ir >= kNR || iz < 0 || iz >= kNZ)
      return 0.f;
    return rho[ir * kNZ + iz];
  }
}  // namespace blMaterialMap
#endif
