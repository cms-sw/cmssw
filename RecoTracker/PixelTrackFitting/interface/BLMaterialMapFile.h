// The material map on disk: a fixed 84-byte little-endian header, the provenance text, the serialized
// Map as one blob and, optionally, the sensor reference positions of the geometry the map was made for.
//
//    0   8  magic          ASCII "BLMM0001"
//    8   4  formatVersion  uint32, 1
//   12   4  nR             int32
//   16   4  nZ             int32
//   20   4  dR             float32 [cm]
//   24   4  dZ             float32 [cm]
//   28   4  zMax           float32 [cm]
//   32  16  geometryTag    char[16], NUL-padded ("T35")
//   48  16  beamPipeTag    char[16], NUL-padded ("2030/v3")
//   64   4  mapVersion     uint32
//   68   4  reserved       uint32, 0
//   72   8  fingerprint    uint64, the census fingerprint of the geometry (BLMaterialMapFingerprint.h)
//   80   4  provenanceLen  uint32, N
//   84   N  provenance     UTF-8 text, no NUL terminator
//   then the body, sizeof(Map) = 2240000 bytes: rho[kSize] float32, then dedx[kSize] of {rhoE, lnI,
//   lnRhoE} float32, r-major (ir * kNZ + iz); then, if present, 16-byte <uint32 rawId, int32 x, y, z> records, one per
//   sensor, x/y/z in 0.1 mm buckets, strictly ascending in rawId.
//
// The reader refuses a map whose nR, nZ, dR, dZ or zMax differ from BLMaterialMap.h.
// test/blMaterialMap/blMaterialMapEmit.py writes the files.
#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapFile_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapFile_h

#include <cstdint>
#include <string>
#include <vector>

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

namespace blMaterialMap {
  // One sensor's reference position in 0.1 mm buckets, matched by positionsMatch (BLMaterialMapFingerprint.h).
  struct LeafPos {
    uint32_t rawId;
    int32_t x, y, z;
    bool operator==(const LeafPos&) const = default;
  };

  // The header fields that the Map does not carry, and the sensor reference positions, empty if the file has none.
  struct FileHeader {
    std::string geometryTag;
    std::string beamPipeTag;
    uint32_t mapVersion;
    uint64_t fingerprint;
    std::string provenance;
    std::vector<LeafPos> sensorPositions;
  };

  // Fill `map` and return the header. Throw cms::Exception("BLMaterialMap") if the file does not agree with the
  // format above, or holds a value that is not finite or a negative density.
  FileHeader readFile(const std::string& path, Map& map);

  // The same checks and header, without reading the Map body.
  FileHeader readHeader(const std::string& path);
}  // namespace blMaterialMap
#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapFile_h
