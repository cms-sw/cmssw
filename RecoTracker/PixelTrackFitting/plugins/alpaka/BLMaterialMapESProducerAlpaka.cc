#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFingerprint.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapHost.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLMaterialMapDevice.h"
#include "RecoTracker/Record/interface/BLMaterialMapRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    // One line of the material-map index: 0x<16hex> <tag> <beampipe> <version> <filename>.
    struct IndexEntry {
      uint64_t fingerprint;  // from the leading 0x<16hex> field
      std::string tag;
      std::string beamPipe;
      std::string version;
      std::string file;
      unsigned int line;  // the index line this entry came from, for the failure messages
    };

    // Read the entries of the index file. Skip blank lines and comments. Throw on a malformed line.
    std::vector<IndexEntry> parseIndex(const std::string& indexPath) {
      std::vector<IndexEntry> entries;
      std::ifstream index(indexPath);
      if (!index) {
        throw cms::Exception("BLMaterialMap") << "cannot open the material-map index \"" << indexPath << "\"";
      }
      std::string line;
      unsigned int lineNr = 0;
      while (std::getline(index, line)) {
        ++lineNr;
        std::istringstream fields(line);
        IndexEntry entry;
        std::string fpText;
        if (not(fields >> fpText) or fpText[0] == '#') {  // blank line or comment
          continue;
        }
        std::string extra;
        // a trailing comment after the five fields is allowed, any other extra token is not
        if (not(fields >> entry.tag >> entry.beamPipe >> entry.version >> entry.file) or
            ((fields >> extra) and extra[0] != '#')) {
          throw cms::Exception("BLMaterialMap")
              << "malformed line " << lineNr << " of " << indexPath << ":\n"
              << line << "\nexpected: 0x<16hex> <tag> <beampipe> <version> <filename>";
        }
        const bool wellFormed =
            fpText.starts_with("0x") and fpText.size() == 18 and
            std::all_of(fpText.begin() + 2, fpText.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
        if (not wellFormed) {
          throw cms::Exception("BLMaterialMap") << "malformed fingerprint '" << fpText << "' on line " << lineNr
                                                << " of " << indexPath << ": expected 0x<16hex>";
        }
        entry.fingerprint = std::stoull(fpText, nullptr, 16);
        entry.line = lineNr;
        entries.push_back(std::move(entry));
      }
      return entries;
    }

    // The index entry of the running geometry. The hash selects the candidates, and the sensor positions in their
    // headers must match for exactly one.
    const IndexEntry& selectMap(const blMaterialMap::GeometryFingerprint& fp,
                                const std::vector<IndexEntry>& entries,
                                const std::string& indexPath,
                                const std::string& indexDir) {
      std::vector<const IndexEntry*> candidates;
      for (const auto& e : entries) {
        if (e.fingerprint == fp.hash) {
          candidates.push_back(&e);
        }
      }
      if (candidates.empty()) {
        std::ostringstream msg;
        msg << "no material map for the running geometry: sensor-census fingerprint "
            << blMaterialMap::fingerprintToHex(fp.hash) << ", sensors PixelBarrel " << fp.sensorCounts[1]
            << ", PixelEndcap " << fp.sensorCounts[2] << ", TIB " << fp.sensorCounts[3] << ", TID "
            << fp.sensorCounts[4] << ", TOB " << fp.sensorCounts[5] << ", TEC " << fp.sensorCounts[6] << ".\nMaps in "
            << indexPath << ":";
        for (const auto& e : entries) {
          msg << "\n  " << blMaterialMap::fingerprintToHex(e.fingerprint) << ' ' << e.tag << ' ' << e.beamPipe << ' '
              << e.version << ' ' << e.file;
        }
        msg << "\nno map for this geometry: generate it with "
               "RecoTracker/PixelTrackFitting/test/blMaterialMap/blMaterialMapRun.sh and add its "
               "line to BLMaterialMap.index";
        throw cms::Exception("BLMaterialMap") << msg.str();
      }

      const IndexEntry* selected = nullptr;
      for (const IndexEntry* e : candidates) {
        const std::string candPath = indexDir + e->file;
        blMaterialMap::FileHeader candHdr;
        try {
          candHdr = blMaterialMap::readHeader(candPath);  // the header and the positions only
        } catch (const cms::Exception& ex) {
          throw cms::Exception("BLMaterialMap")
              << "line " << e->line << " of " << indexPath << " lists " << e->file << ": " << ex.message();
        }
        if (candHdr.sensorPositions.empty()) {
          throw cms::Exception("BLMaterialMap") << candPath << " carries no sensor reference positions: regenerate "
                                                << "it with the emitter's --positions option";
        }
        if (!blMaterialMap::positionsMatch(fp, candHdr.sensorPositions)) {
          continue;
        }
        if (selected) {
          throw cms::Exception("BLMaterialMap")
              << "two maps are compatible with the running geometry: " << indexDir + selected->file << " and "
              << candPath << " (" << indexPath << " must list exactly one per geometry)";
        }
        selected = e;
      }
      if (!selected) {
        std::ostringstream msg;
        msg << "the sensor census of the running geometry matches but its sensor positions do not: "
            << blMaterialMap::fingerprintToHex(fp.hash) << " (" << fp.positions.size()
            << " sensors) vs the reference positions of";
        for (const IndexEntry* e : candidates) {
          msg << "\n  " << e->file << " (" << e->tag << ')';
        }
        msg << "\nthis is a geometry with the supported sensors in moved placements: generate its map "
               "with RecoTracker/PixelTrackFitting/test/blMaterialMap/blMaterialMapRun.sh";
        throw cms::Exception("BLMaterialMap") << msg.str();
      }
      return *selected;
    }

    // Read the selected map file into the payload. Its fingerprint must agree with the running geometry.
    std::unique_ptr<BLMaterialMapHost> readSelected(const IndexEntry& selected,
                                                    const std::string& path,
                                                    const blMaterialMap::GeometryFingerprint& fp,
                                                    const std::string& indexPath) {
      auto product = std::make_unique<BLMaterialMapHost>(cms::alpakatools::host());
      const auto hdr = blMaterialMap::readFile(path, product->value());  // the body, straight into the payload
      if (hdr.fingerprint != fp.hash) {
        throw cms::Exception("BLMaterialMap")
            << "fingerprint mismatch between the index and the map file: " << indexPath << " binds fingerprint "
            << blMaterialMap::fingerprintToHex(selected.fingerprint) << " to " << path
            << ", but the file's embedded fingerprint is " << blMaterialMap::fingerprintToHex(hdr.fingerprint)
            << " and the job's geometry fingerprint is " << blMaterialMap::fingerprintToHex(fp.hash);
      }
      return product;
    }

  }  // namespace

  // The material map of the BrokenLine fit: the map file that indexFile binds to the fingerprint of the running
  // ideal geometry (BLMaterialMapFingerprint.h). Every failure throws cms::Exception("BLMaterialMap").
  // There is no fallback map.
  class BLMaterialMapESProducerAlpaka : public ESProducer {
  public:
    BLMaterialMapESProducerAlpaka(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig), indexFile_(iConfig.getParameter<edm::FileInPath>("indexFile")) {
      auto cc = setWhatProduced(this);
      geomToken_ = cc.consumesFrom<GeometricDet, IdealGeometryRecord>(edm::ESInputTag{});
    }

    std::unique_ptr<BLMaterialMapHost> produce(const BLMaterialMapRecord& iRecord) {
      const auto fp = blMaterialMap::geometryFingerprint(iRecord.get(geomToken_));

      const std::string indexPath = indexFile_.fullPath();
      const std::string indexDir = indexPath.substr(0, indexPath.find_last_of('/') + 1);
      const std::vector<IndexEntry> entries = parseIndex(indexPath);
      const IndexEntry& selected = selectMap(fp, entries, indexPath, indexDir);
      const std::string path = indexDir + selected.file;
      auto product = readSelected(selected, path, fp, indexPath);

      edm::LogInfo("BLMaterialMap") << "geometry fingerprint " << blMaterialMap::fingerprintToHex(fp.hash) << " -> "
                                    << path;
      return product;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("indexFile",
                                edm::FileInPath("RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index"))
          ->setComment("index of the per-geometry material maps; the map files sit next to it");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomToken_;
    const edm::FileInPath indexFile_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(BLMaterialMapESProducerAlpaka);
