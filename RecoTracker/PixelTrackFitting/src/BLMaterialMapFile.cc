// Reader of the material-map file, layout in interface/BLMaterialMapFile.h.

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"

namespace {
  constexpr char kMagic[8] = {'B', 'L', 'M', 'M', '0', '0', '0', '1'};
  constexpr uint32_t kFormatVersion = 1;
  constexpr int kTagBytes = 16;  // the NUL-padded tag fields
  constexpr size_t kHeaderBytes = 8 + 4 + 4 + 4 + 4 + 4 + 4 + 2 * kTagBytes + 4 + 4 + 8 + 4;
  static_assert(kHeaderBytes == 84, "the fixed file header is 84 bytes");

  // the blob copies below assume a little-endian host
  static_assert(std::endian::native == std::endian::little, "BLMaterialMap files are little-endian");
  static_assert(sizeof(blMaterialMap::Map) == 2'240'000, "the Map blob is 2240000 bytes");

  template <typename T>
  void get(const char*& p, T& v) {
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
  }
  std::string getTag(const char*& p) {
    const char* end = static_cast<const char*>(std::memchr(p, '\0', kTagBytes));
    std::string tag(p, end ? end : p + kTagBytes);
    p += kTagBytes;
    return tag;
  }

  // Parse and check the fixed header and the provenance. On return `in` is at the start of the Map body, and
  // `nPos` is the number of LeafPos records after the body.
  blMaterialMap::FileHeader parseHeader(std::ifstream& in, const std::string& path, std::size_t& nPos) {
    char header[kHeaderBytes];
    in.read(header, static_cast<std::streamsize>(kHeaderBytes));
    if (in.gcount() != static_cast<std::streamsize>(kHeaderBytes)) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": header of " << in.gcount() << " bytes, expected " << kHeaderBytes;
    }

    const char* p = header;
    if (std::memcmp(p, kMagic, sizeof(kMagic)) != 0) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": magic \"" << std::string(p, sizeof(kMagic)) << "\", expected \"BLMM0001\"";
    }
    p += sizeof(kMagic);
    uint32_t formatVersion;
    get(p, formatVersion);
    if (formatVersion != kFormatVersion) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": format version " << formatVersion << ", expected " << kFormatVersion;
    }
    int32_t nR, nZ;
    float dR, dZ, zMax;
    get(p, nR);
    get(p, nZ);
    get(p, dR);
    get(p, dZ);
    get(p, zMax);
    if (nR != blMaterialMap::kNR) {
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": nR " << nR << ", expected " << blMaterialMap::kNR;
    }
    if (nZ != blMaterialMap::kNZ) {
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": nZ " << nZ << ", expected " << blMaterialMap::kNZ;
    }
    if (dR != blMaterialMap::kDR) {  // exact: kDR, kDZ and kZMAX are exact in float
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": dR " << std::setprecision(9) << dR << ", expected " << blMaterialMap::kDR;
    }
    if (dZ != blMaterialMap::kDZ) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": dZ " << std::setprecision(9) << dZ << ", expected " << blMaterialMap::kDZ;
    }
    if (zMax != blMaterialMap::kZMAX) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": zMax " << std::setprecision(9) << zMax << ", expected " << blMaterialMap::kZMAX;
    }
    const std::string geometryTag = getTag(p);
    const std::string beamPipeTag = getTag(p);
    uint32_t mapVersion;
    get(p, mapVersion);
    uint32_t reserved;
    get(p, reserved);
    if (reserved != 0) {
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": reserved " << reserved << ", expected 0";
    }
    uint64_t fingerprint;
    get(p, fingerprint);
    uint32_t provenanceLen;
    get(p, provenanceLen);

    in.seekg(0, std::ios::end);
    const std::streamoff fileSize = in.tellg();
    const uint64_t head = static_cast<uint64_t>(kHeaderBytes) + provenanceLen + sizeof(blMaterialMap::Map);
    if (fileSize < 0 || static_cast<uint64_t>(fileSize) < head ||
        (static_cast<uint64_t>(fileSize) - head) % sizeof(blMaterialMap::LeafPos) != 0) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": " << fileSize << " bytes, expected " << head << " + 16*n (" << kHeaderBytes
          << " header + " << provenanceLen << " provenance + " << sizeof(blMaterialMap::Map)
          << " map + whole LeafPos records)";
    }
    nPos = (static_cast<uint64_t>(fileSize) - head) / sizeof(blMaterialMap::LeafPos);

    std::string provenance(provenanceLen, '\0');
    in.seekg(static_cast<std::streamoff>(kHeaderBytes));
    if (provenanceLen > 0 && !in.read(provenance.data(), provenanceLen)) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": provenance read stopped at " << in.gcount() << " of " << provenanceLen << " bytes";
    }
    return blMaterialMap::FileHeader{geometryTag, beamPipeTag, mapVersion, fingerprint, provenance, {}};
  }

  // the appendix, read at the current position: nPos records, strictly ascending in rawId
  void readPositions(std::ifstream& in,
                     const std::string& path,
                     std::size_t nPos,
                     std::vector<blMaterialMap::LeafPos>& out) {
    if (nPos == 0) {
      return;
    }
    static_assert(sizeof(blMaterialMap::LeafPos) == 16);
    out.resize(nPos);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(nPos * sizeof(blMaterialMap::LeafPos)));
    if (!in) {
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": position records read stopped at " << in.gcount()
                                            << " of " << nPos * sizeof(blMaterialMap::LeafPos) << " bytes";
    }
    for (std::size_t i = 1; i < nPos; ++i) {
      if (out[i].rawId <= out[i - 1].rawId) {
        throw cms::Exception("BLMaterialMap")
            << "\"" << path << "\": position records not sorted by rawId: record " << i << " has rawId " << out[i].rawId
            << ", record " << i - 1 << " has " << out[i - 1].rawId;
      }
    }
  }
}  // namespace

namespace blMaterialMap {
  FileHeader readFile(const std::string& path, Map& map) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw cms::Exception("BLMaterialMap") << "cannot open \"" << path << "\" for reading";
    }
    std::size_t nPos = 0;
    FileHeader hdr = parseHeader(in, path, nPos);  // leaves `in` at the body
    in.read(reinterpret_cast<char*>(&map), sizeof(Map));
    if (!in) {
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": map body read stopped at " << in.gcount() << " of " << sizeof(Map) << " bytes";
    }
    // every density finite and >= 0, every log-mean finite
    for (int i = 0; i < kSize; ++i) {
      const DeDx& d = map.dedx[i];
      if (edm::isNotFinite(map.rho[i]) || !(map.rho[i] >= 0.f) || edm::isNotFinite(d.rhoE) || !(d.rhoE >= 0.f) ||
          edm::isNotFinite(d.lnI) || edm::isNotFinite(d.lnRhoE)) {
        throw cms::Exception("BLMaterialMap")
            << "\"" << path << "\": cell (ir " << i / kNZ << ", iz " << i % kNZ << ") has rho " << map.rho[i]
            << ", rhoE " << d.rhoE << ", lnI " << d.lnI << ", lnRhoE " << d.lnRhoE
            << "; expected finite values and densities >= 0";
      }
    }
    readPositions(in, path, nPos, hdr.sensorPositions);
    return hdr;
  }

  FileHeader readHeader(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw cms::Exception("BLMaterialMap") << "cannot open \"" << path << "\" for reading";
    }
    std::size_t nPos = 0;
    FileHeader hdr = parseHeader(in, path, nPos);
    in.seekg(static_cast<std::streamoff>(sizeof(Map)), std::ios::cur);  // past the body: it is not read here
    readPositions(in, path, nPos, hdr.sensorPositions);
    return hdr;
  }
}  // namespace blMaterialMap
