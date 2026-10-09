// The key that binds a material map to its tracker geometry, from the ideal GeometricDet tree:
//  - hash: FNV-1a over the sorted rawIds of the tracker sensors;
//  - positions: the ideal position of each sensor in 0.1 mm buckets, compared within one bucket.
// The file header tags identify the beam pipe and the materials.
#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapFingerprint_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapFingerprint_h

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"  // LeafPos

class GeometricDet;

namespace blMaterialMap {
  struct GeometryFingerprint {
    uint64_t hash = 0;                  // FNV-1a over the sorted rawIds, 4 bytes LE each
    std::array<int, 7> sensorCounts{};  // index = DetId::subdetId(), slot 0 unused; diagnostics only
    std::vector<LeafPos> positions;     // sorted by rawId, quantized to 0.1 mm
  };

  // true for a leaf of the GeometricDet tree that the fingerprint counts: subdetId 1..6 of the tracker
  inline bool isTrackerSensor(DetId id) {
    return id.det() == DetId::Tracker && id.subdetId() >= 1 && id.subdetId() <= 6;
  }

  GeometryFingerprint geometryFingerprint(const GeometricDet& gd);
  // true when b has the same sensors as a, in the same order, each within tolBuckets per coordinate
  bool positionsMatch(const GeometryFingerprint& a, const std::vector<LeafPos>& b, int tolBuckets = 1);
  std::string fingerprintToHex(uint64_t h);  // "0x" + 16 lowercase hex digits
}  // namespace blMaterialMap

#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapFingerprint_h
