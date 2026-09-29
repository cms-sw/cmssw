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
//   lnRhoE} float32, kNZ-major; then, if present, 16-byte <uint32 rawId, int32 x, y, z> records, one per
//   sensor, x/y/z in 0.1 mm buckets, strictly ascending in rawId.
//
// nR/nZ/dR/dZ/zMax repeat the compile-time lattice constants of BLMaterialMap.h so that a map made for
// another lattice is refused. The Map is stored bit-exact. test/blMaterialMap/blMaterialMapEmit.py
// writes the files.
#ifndef RecoTracker_PixelTrackFitting_BLMaterialMapFile_h
#define RecoTracker_PixelTrackFitting_BLMaterialMapFile_h

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

namespace blMaterialMap {
  // One sensor's reference position in 0.1 mm buckets. The producer matches the job's geometry against
  // these with one bucket of tolerance (sourcing modes of one geometry differ by nm, tracker versions
  // by 0.6 mm or more).
  struct LeafPos {
    uint32_t rawId;
    int32_t x, y, z;
    bool operator==(const LeafPos&) const = default;
  };

  // The header fields the Map itself does not carry, plus the sensor reference positions (empty when
  // the file has none).
  struct FileHeader {
    std::string geometryTag;
    std::string beamPipeTag;
    uint32_t mapVersion;
    uint64_t fingerprint;
    std::string provenance;
    std::vector<LeafPos> sensorPositions;
  };

  // Writes the header, the provenance, the Map bit-exact and, if given, the sensor positions. Throws
  // cms::Exception("BLMaterialMap") on a tag that does not fit the 16-byte field or contains a space
  // or a NUL, and on a write failure.
  void writeFile(const std::string& path,
                 const Map& map,
                 std::string_view geometryTag,
                 std::string_view beamPipeTag,
                 uint32_t mapVersion,
                 uint64_t fingerprint,
                 std::string_view provenance,
                 const std::vector<LeafPos>* sensorPositions = nullptr);

  // Fills `map` and returns the header. Throws cms::Exception("BLMaterialMap") on an unreadable file, a
  // bad magic, a format version other than 1, lattice constants different from kNR/kNZ/kDR/kDZ/kZMAX, a
  // nonzero reserved field, a size that is not header + provenance + Map + whole 16-byte records, a short
  // read, or positions not strictly ascending in rawId.
  FileHeader readFile(const std::string& path, Map& map);

  // The same validation and header without reading the Map body: the producer checks its candidates
  // with it and reads the body of the selected file only.
  FileHeader readHeader(const std::string& path);
}  // namespace blMaterialMap
#endif
