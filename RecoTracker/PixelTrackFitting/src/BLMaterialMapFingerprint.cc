// Leaves: gd.deepComponents() (leaf nodes only) with det() == DetId::Tracker and subdetId() in 1..6.
// Hash: 64-bit FNV-1a over the rawIds sorted ascending, 4 bytes each, little-endian on every host
// (explicit shifts). Positions: llround(mm * 10), i.e. 0.1 mm buckets, kept for positionsMatch().
// sensorCounts is diagnostics for the failure messages.
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
      const int subdet = id.subdetId();
      if (id.det() != DetId::Tracker || subdet < 1 || subdet > 6)
        continue;
      const GeometricDet::Translation& t = leaf->translation();  // mm
      fp.positions.push_back(LeafPos{id.rawId(),
                                     static_cast<int32_t>(std::llround(t.x() * 10.0)),
                                     static_cast<int32_t>(std::llround(t.y() * 10.0)),
                                     static_cast<int32_t>(std::llround(t.z() * 10.0))});
      fp.sensorCounts[subdet]++;
    }
    std::sort(fp.positions.begin(), fp.positions.end(), [](const LeafPos& a, const LeafPos& b) {
      return a.rawId < b.rawId;  //
    });

    constexpr uint64_t kFNVOffset = 14695981039346656037ull;
    constexpr uint64_t kFNVPrime = 1099511628211ull;
    uint64_t h = kFNVOffset;
    for (const LeafPos& p : fp.positions)
      for (int i = 0; i < 4; i++)
        h = (h ^ ((p.rawId >> (8 * i)) & 0xff)) * kFNVPrime;
    fp.hash = h;
    return fp;
  }

  bool positionsMatch(const GeometryFingerprint& a, const std::vector<LeafPos>& b, int tolBuckets) {
    if (a.positions.size() != b.size())
      return false;
    for (std::size_t i = 0; i < b.size(); ++i) {
      const LeafPos& pa = a.positions[i];
      const LeafPos& pb = b[i];
      if (pa.rawId != pb.rawId)
        return false;
      if (std::abs(pa.x - pb.x) > tolBuckets || std::abs(pa.y - pb.y) > tolBuckets ||
          std::abs(pa.z - pb.z) > tolBuckets)
        return false;
    }
    return true;
  }

  std::string fingerprintToHex(uint64_t h) {
    char buf[19];  // "0x" + 16 digits + NUL
    std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(h));
    return buf;
  }

}  // namespace blMaterialMap
