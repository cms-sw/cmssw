#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFingerprint.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace blMaterialMap {

  GeometryFingerprint geometryFingerprint(const GeometricDet& gd) {
    GeometryFingerprint fp;
    const auto leaves = gd.deepComponents();
    fp.positions.reserve(leaves.size());
    for (const GeometricDet* leaf : leaves) {
      const DetId id = leaf->geographicalId();
      if (!isTrackerSensor(id)) {
        continue;
      }
      const GeometricDet::Translation& t = leaf->translation();  // mm
      fp.positions.push_back(LeafPos{id.rawId(),
                                     static_cast<int32_t>(std::llround(t.x() * 10.0)),
                                     static_cast<int32_t>(std::llround(t.y() * 10.0)),
                                     static_cast<int32_t>(std::llround(t.z() * 10.0))});
      fp.sensorCounts[id.subdetId()]++;
    }
    std::sort(
        fp.positions.begin(), fp.positions.end(), [](const LeafPos& a, const LeafPos& b) { return a.rawId < b.rawId; });

    constexpr uint64_t kFNVOffset = 14695981039346656037ull;
    constexpr uint64_t kFNVPrime = 1099511628211ull;
    uint64_t h = kFNVOffset;
    // 4 bytes per rawId, least significant first, so the hash is the same on every host
    for (const LeafPos& p : fp.positions) {
      for (int i = 0; i < 4; i++) {
        h = (h ^ ((p.rawId >> (8 * i)) & 0xff)) * kFNVPrime;
      }
    }
    fp.hash = h;
    return fp;
  }

  bool positionsMatch(const GeometryFingerprint& a, const std::vector<LeafPos>& b, int tolBuckets) {
    if (a.positions.size() != b.size()) {
      return false;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
      const LeafPos& pa = a.positions[i];
      const LeafPos& pb = b[i];
      if (pa.rawId != pb.rawId) {
        return false;
      }
      if (std::abs(pa.x - pb.x) > tolBuckets || std::abs(pa.y - pb.y) > tolBuckets ||
          std::abs(pa.z - pb.z) > tolBuckets) {
        return false;
      }
    }
    return true;
  }

  std::string fingerprintToHex(uint64_t h) {
    char buf[19];  // "0x" + 16 digits + NUL
    std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(h));
    return buf;
  }

}  // namespace blMaterialMap
