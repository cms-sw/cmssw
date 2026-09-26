// Generator-side half of the material-map geometry key: writes the fingerprint of
// blMaterialMap::geometryFingerprint (FNV-1a over the tracker sensor rawIds) to `out` on the first event,
//   FINGERPRINT 0x0123456789abcdef
//   SENSORS <subdetId> <count>     (one line per subdetector with sensors)
// and, with dumpPositions, one "LEAF <rawId> <x> <y> <z>" line per sensor (mm, full precision, sorted by
// rawId), from which blMaterialMapEmit.py builds the file's reference positions. Every rays job of
// blMaterialMapRun.sh runs it; the script requires all jobs to agree and puts the fingerprint into
// PROVENANCE.txt and the map header.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFingerprint.h"

class BLMaterialMapFingerprintDump : public edm::one::EDAnalyzer<> {
public:
  explicit BLMaterialMapFingerprintDump(const edm::ParameterSet& cfg)
      : out_(cfg.getUntrackedParameter<std::string>("out")),
        dumpPositions_(cfg.getUntrackedParameter<bool>("dumpPositions", false)),
        geomDetToken_(esConsumes<GeometricDet, IdealGeometryRecord>()) {}

private:
  void analyze(const edm::Event&, const edm::EventSetup& setup) override {
    if (written_)
      return;
    written_ = true;
    const GeometricDet& gd = setup.getData(geomDetToken_);
    const blMaterialMap::GeometryFingerprint fp = blMaterialMap::geometryFingerprint(gd);
    const std::string hex = blMaterialMap::fingerprintToHex(fp.hash);
    std::ofstream out(out_);
    if (!out)
      throw cms::Exception("BLMaterialMap") << "cannot write the geometry fingerprint to " << out_;
    out << "FINGERPRINT " << hex << '\n';
    for (std::size_t subdet = 1; subdet < fp.sensorCounts.size(); ++subdet)
      if (fp.sensorCounts[subdet] != 0)
        out << "SENSORS " << subdet << ' ' << fp.sensorCounts[subdet] << '\n';
    edm::LogVerbatim("BLMaterialMap") << "FINGERPRINT " << hex;
    if (!dumpPositions_)
      return;
    // full-precision positions, sorted by rawId; the emitter quantizes them to the file's 0.1 mm buckets
    struct Leaf {
      uint32_t rawId;
      double x, y, z;
    };
    std::vector<Leaf> leaves;
    for (const GeometricDet* leaf : gd.deepComponents()) {
      const DetId id = leaf->geographicalId();
      const int subdet = id.subdetId();
      if (id.det() != DetId::Tracker || subdet < 1 || subdet > 6)
        continue;
      const auto& t = leaf->translation();
      leaves.push_back(Leaf{id.rawId(), t.x(), t.y(), t.z()});
    }
    std::sort(leaves.begin(), leaves.end(), [](const Leaf& a, const Leaf& b) { return a.rawId < b.rawId; });
    out << std::setprecision(17);
    for (const Leaf& l : leaves)
      out << "LEAF " << l.rawId << ' ' << l.x << ' ' << l.y << ' ' << l.z << '\n';
  }

  const std::string out_;
  const bool dumpPositions_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  bool written_ = false;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BLMaterialMapFingerprintDump);
