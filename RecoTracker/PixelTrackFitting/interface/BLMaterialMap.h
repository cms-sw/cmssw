// Geant4 material radiation-length density rho(r,z) [X0/cm] (Tracker + BeamPipe), phi-averaged.
// Integrate along a track segment -> its X/X0. One table is compiled in (BLMaterialMapD121.cc,
// blMaterialMapDataD121()), generated from the D121 tracker geometry on the 0.5 cm radial lattice; it is the
// material model the GBL fits are calibrated against. rhoAt reads it through the passed pointer: the device
// fits get it via EventSetup (BLMaterialMapESProducerAlpaka -> BLMaterialMapHost, copied once per IOV), the
// unit tests under test/ read the compiled-in array directly. Track radii are non-negative (rhoAt is only
// sampled between two hit radii).
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
  const float* blMaterialMapDataD121();  // kSize values (BLMaterialMapD121.cc)
  // rhoAt: local density [X0/cm] at (r,z) from rho (host array or device buffer); 0 outside the grid.
  // constexpr so callable from device code (deref at run time).
  constexpr inline float rhoAt(const float* rho, float r, float z) {
    int ir = int(r / kDR), iz = int((z + kZMAX) / kDZ);
    if (ir < 0 || ir >= kNR || iz < 0 || iz >= kNZ)
      return 0.f;
    return rho[ir * kNZ + iz];
  }
}  // namespace blMaterialMap
#endif
