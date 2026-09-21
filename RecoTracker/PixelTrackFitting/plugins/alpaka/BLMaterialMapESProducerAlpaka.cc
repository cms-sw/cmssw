#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFingerprint.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLMaterialMapCollection.h"
#include "RecoTracker/Record/interface/BLMaterialMapRecord.h"

namespace {
  // the index of the per-geometry maps, next to which the map files sit
  constexpr const char* kIndexFile = "RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index";
}  // namespace

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Serves the BL-fit material map as an EventSetup portable condition: the host payload is read from the
  // binary file that the index binds to the fingerprint of the running ideal geometry
  // (BLMaterialMapFingerprint.h) and copied to the device once per IOV. No configuration parameter, no
  // fallback map. The fingerprint hash (sensor census) selects the candidate files; where several tracker
  // versions share a census (T36/T37/T38) the sensor positions stored in each file pick the one matching
  // the job's within a 0.1 mm bucket. Only the candidates' headers are read; the 2.24 MB body is read for
  // the selected file alone. Every failure is a cms::Exception("BLMaterialMap"): malformed index line or
  // file, no candidate, no position match, two matches, embedded fingerprint different from the job's.
  // Only the stub reconstruction requests BLMaterialMapRecord, so other jobs never call produce().
  class BLMaterialMapESProducerAlpaka : public ESProducer {
  public:
    BLMaterialMapESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      geomToken_ = cc.consumesFrom<GeometricDet, IdealGeometryRecord>(edm::ESInputTag{});
    }

    std::unique_ptr<BLMaterialMapHost> produce(const BLMaterialMapRecord& iRecord) {
      // IdealGeometryRecord is a second-level dependency of BLMaterialMapRecord (through
      // TrackerDigiGeometryRecord), so the get() is routed one dependent record at a time.
      auto const digiRecord = iRecord.getRecord<TrackerDigiGeometryRecord>();
      auto const& geom = digiRecord.get(geomToken_);
      const auto fp = blMaterialMap::geometryFingerprint(geom);

      // The index is parsed on every produce: produce runs once per geometry IOV, so nothing is cached.
      struct Entry {
        uint64_t fingerprint;  // from the leading 0x<16hex> field
        std::string fpText;
        std::string tag;
        std::string beamPipe;
        std::string version;
        std::string file;
        unsigned int line;  // the index line this entry came from, for the failure messages
      };
      const edm::FileInPath indexPath(kIndexFile);
      const std::string indexDir = indexPath.fullPath().substr(0, indexPath.fullPath().find_last_of('/') + 1);
      std::vector<Entry> entries;
      {
        std::ifstream index(indexPath.fullPath());
        std::string line;
        unsigned int lineNr = 0;
        while (std::getline(index, line)) {
          ++lineNr;
          std::istringstream fields(line);
          Entry entry;
          if (not(fields >> entry.fpText) or entry.fpText[0] == '#')  // blank line or comment
            continue;
          std::string extra;
          // a trailing comment after the five fields is allowed, any other extra token is not
          if (not(fields >> entry.tag >> entry.beamPipe >> entry.version >> entry.file) or
              ((fields >> extra) and extra[0] != '#'))
            throw cms::Exception("BLMaterialMap")
                << "malformed line " << lineNr << " of " << indexPath.fullPath() << ":\n"
                << line << "\nexpected: 0x<16hex> <tag> <beampipe> <version> <filename>";
          const bool wellFormed = entry.fpText.starts_with("0x") and entry.fpText.size() == 18 and
                                  std::all_of(entry.fpText.begin() + 2, entry.fpText.end(), [](unsigned char c) {
                                    return std::isxdigit(c) != 0;
                                  });
          if (not wellFormed)
            throw cms::Exception("BLMaterialMap") << "malformed fingerprint '" << entry.fpText << "' on line " << lineNr
                                                  << " of " << indexPath.fullPath() << ": expected 0x<16hex>";
          entry.fingerprint = std::stoull(entry.fpText, nullptr, 16);
          entry.line = lineNr;
          entries.push_back(std::move(entry));
        }
      }

      std::vector<const Entry*> candidates;
      for (const auto& e : entries)
        if (e.fingerprint == fp.hash)
          candidates.push_back(&e);
      if (candidates.empty()) {
        std::ostringstream msg;
        msg << "no material map for the running geometry: sensor-census fingerprint "
            << blMaterialMap::fingerprintToHex(fp.hash) << ", sensors PixelBarrel " << fp.sensorCounts[1]
            << ", PixelEndcap " << fp.sensorCounts[2] << ", TIB " << fp.sensorCounts[3] << ", TID "
            << fp.sensorCounts[4] << ", TOB " << fp.sensorCounts[5] << ", TEC " << fp.sensorCounts[6] << ".\nMaps in "
            << indexPath.fullPath() << ":";
        for (const auto& e : entries)
          msg << "\n  " << e.fpText << ' ' << e.tag << ' ' << e.beamPipe << ' ' << e.version << ' ' << e.file;
        msg << "\nno map for this geometry: generate it with "
               "RecoTracker/PixelTrackFitting/test/blMaterialMap/blMaterialMapRun.sh and add its "
               "line to BLMaterialMap.index";
        throw cms::Exception("BLMaterialMap") << msg.str();
      }

      // inside a family (T36/T37/T38) the positions stored in each candidate decide, within one bucket
      const Entry* it = nullptr;
      std::string path;
      for (const Entry* e : candidates) {
        std::string candPath = indexDir + e->file;
        blMaterialMap::FileHeader candHdr;
        try {
          candHdr = blMaterialMap::readHeader(candPath);  // the header and the positions only
        } catch (const cms::Exception& ex) {
          throw cms::Exception("BLMaterialMap")
              << "line " << e->line << " of " << indexPath.fullPath() << " lists " << e->file << ": " << ex.message();
        }
        if (candHdr.sensorPositions.empty())
          throw cms::Exception("BLMaterialMap") << candPath << " carries no sensor reference positions: regenerate "
                                                << "it with the emitter's --positions option";
        if (!blMaterialMap::positionsMatch(fp, candHdr.sensorPositions))
          continue;
        if (it)
          throw cms::Exception("BLMaterialMap")
              << "two maps are compatible with the running geometry: " << path << " and " << candPath << " ("
              << indexPath.fullPath() << " must list exactly one per geometry)";
        it = e;
        path = std::move(candPath);
      }
      if (!it) {
        std::ostringstream msg;
        msg << "the sensor census of the running geometry matches but its sensor positions do not: "
            << blMaterialMap::fingerprintToHex(fp.hash) << " (" << fp.positions.size()
            << " sensors) vs the reference positions of";
        for (const Entry* e : candidates)
          msg << "\n  " << e->file << " (" << e->tag << ')';
        msg << "\nthis is a geometry with the supported sensors in moved placements: generate its map "
               "with RecoTracker/PixelTrackFitting/test/blMaterialMap/blMaterialMapRun.sh";
        throw cms::Exception("BLMaterialMap") << msg.str();
      }

      auto product = std::make_unique<BLMaterialMapHost>();
      const auto hdr = blMaterialMap::readFile(path, product->map());  // the body, straight into the payload

      // the file must carry the census it was selected by
      if (hdr.fingerprint != fp.hash)
        throw cms::Exception("BLMaterialMap")
            << "the index lies: " << indexPath.fullPath() << " binds fingerprint " << it->fpText << " to " << path
            << ", but the file's embedded fingerprint is " << blMaterialMap::fingerprintToHex(hdr.fingerprint)
            << " and the job's geometry fingerprint is " << blMaterialMap::fingerprintToHex(fp.hash);

      edm::LogVerbatim("BLMaterialMap") << "geometry fingerprint " << blMaterialMap::fingerprintToHex(fp.hash) << " -> "
                                        << path;
      return product;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(BLMaterialMapESProducerAlpaka);
