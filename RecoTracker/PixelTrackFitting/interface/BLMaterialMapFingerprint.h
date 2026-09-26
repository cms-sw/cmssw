// The key that binds a material map to its tracker geometry, computed on the ideal GeometricDet tree
// (no alignment enters). Two parts:
//  - hash: FNV-1a over the sorted rawIds of all tracker sensors (the census). Identical for every sourcing
//    mode of one geometry (DDD, DD4hep, DB) and for tracker versions with the same sensor inventory
//    (T36/T37/T38 differ only in module placement).
//  - positions: every sensor's ideal position in 0.1 mm buckets. Compared, not hashed: sourcing modes
//    evaluate the same position within a few nm, which can cross a bucket boundary and would flip a hash.
//    positionsMatch() with one bucket of tolerance separates the versions inside a family (their modules
//    move by 0.6 mm or more; measured T36/T37/T38: 6 to 13 buckets) and nothing else.
// The beam pipe and the materials are not sensors; the file header tags identify them.
#ifndef RecoTracker_PixelTrackFitting_BLMaterialMapFingerprint_h
#define RecoTracker_PixelTrackFitting_BLMaterialMapFingerprint_h

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"  // LeafPos

class GeometricDet;

namespace blMaterialMap {
  struct GeometryFingerprint {
    uint64_t hash = 0;                  // FNV-1a over the sorted rawIds, 4 bytes LE each
    std::array<int, 7> sensorCounts{};  // index = DetId::subdetId(), slot 0 unused; diagnostics only
    std::vector<LeafPos> positions;     // sorted by rawId, quantized to 0.1 mm
  };

  GeometryFingerprint geometryFingerprint(const GeometricDet& gd);
  // true when b has the same sensors as a, in the same order, each within tolBuckets per coordinate
  bool positionsMatch(const GeometryFingerprint& a, const std::vector<LeafPos>& b, int tolBuckets = 1);
  std::string fingerprintToHex(uint64_t h);  // "0x" + 16 lowercase hex digits
}  // namespace blMaterialMap

#endif
