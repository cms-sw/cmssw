// Writer and reader of the material-map file (the offset table lives in interface/BLMaterialMapFile.h):
// the 84-byte header is packed and parsed field by field -- never by dumping a struct -- and the Map
// body is one blob in each direction, exact bytes on a little-endian host. The lattice constants in the
// header stay compile-time in BLMaterialMap.h; readFile only checks the file's copies against them, so
// a map canned for another lattice is refused here instead of being marched at the wrong dR.

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"

namespace {
  constexpr char kMagic[8] = {'B', 'L', 'M', 'M', '0', '0', '0', '1'};
  constexpr uint32_t kFormatVersion = 1;
  constexpr int kTagBytes = 16;  // the NUL-padded tag fields
  constexpr size_t kHeaderBytes = 8 + 4 + 4 + 4 + 4 + 4 + 4 + 2 * kTagBytes + 4 + 4 + 8 + 4;
  static_assert(kHeaderBytes == 84, "the fixed file header is 84 bytes");

  // the header fields are little-endian and the body's floats are the host's own: the blob copies below
  // are exact only on a little-endian host
  static_assert(std::endian::native == std::endian::little, "BLMaterialMap files are little-endian");
  static_assert(sizeof(blMaterialMap::Map) == 2'240'000, "the Map blob is 2240000 bytes");

  template <typename T>
  void put(char*& p, const T& v) {
    std::memcpy(p, &v, sizeof(T));
    p += sizeof(T);
  }
  template <typename T>
  void get(const char*& p, T& v) {
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
  }
  // a tag field: the tag's chars, NUL-padded to 16 bytes
  void putTag(char*& p, std::string_view tag) {
    char field[kTagBytes] = {};
    if (!tag.empty())  // memcpy from an empty view has no valid source pointer
      std::memcpy(field, tag.data(), tag.size());
    std::memcpy(p, field, kTagBytes);
    p += kTagBytes;
  }
  std::string getTag(const char*& p) {
    const char* end = static_cast<const char*>(std::memchr(p, '\0', kTagBytes));
    std::string tag(p, end ? end : p + kTagBytes);
    p += kTagBytes;
    return tag;
  }

  // The one parser of the fixed header: every check readFile and readHeader share lives here, so they
  // cannot diverge. On return `in` sits at the start of the Map body, the header carries the
  // provenance and `nPos` says how many LeafPos records follow the body.
  blMaterialMap::FileHeader parseHeader(std::ifstream& in, const std::string& path, std::size_t& nPos) {
    char header[kHeaderBytes];
    in.read(header, static_cast<std::streamsize>(kHeaderBytes));
    if (in.gcount() != static_cast<std::streamsize>(kHeaderBytes))
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": header of " << in.gcount() << " bytes, expected " << kHeaderBytes;

    const char* p = header;
    if (std::memcmp(p, kMagic, sizeof(kMagic)) != 0)
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": magic \"" << std::string(p, sizeof(kMagic)) << "\", expected \"BLMM0001\"";
    p += sizeof(kMagic);
    uint32_t formatVersion;
    get(p, formatVersion);
    if (formatVersion != kFormatVersion)
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": format version " << formatVersion << ", expected " << kFormatVersion;
    int32_t nR, nZ;
    float dR, dZ, zMax;
    get(p, nR);
    get(p, nZ);
    get(p, dR);
    get(p, dZ);
    get(p, zMax);
    if (nR != blMaterialMap::kNR)
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": nR " << nR << ", expected " << blMaterialMap::kNR;
    if (nZ != blMaterialMap::kNZ)
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": nZ " << nZ << ", expected " << blMaterialMap::kNZ;
    if (dR != blMaterialMap::kDR)  // the lattice steps and zMax are exact decimals: == is exact
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": dR " << std::setprecision(9) << dR << ", expected " << blMaterialMap::kDR;
    if (dZ != blMaterialMap::kDZ)
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": dZ " << std::setprecision(9) << dZ << ", expected " << blMaterialMap::kDZ;
    if (zMax != blMaterialMap::kZMAX)
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": zMax " << std::setprecision(9) << zMax << ", expected " << blMaterialMap::kZMAX;
    const std::string geometryTag = getTag(p);
    const std::string beamPipeTag = getTag(p);
    uint32_t mapVersion;
    get(p, mapVersion);
    uint32_t reserved;
    get(p, reserved);
    if (reserved != 0)
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": reserved " << reserved << ", expected 0";
    uint64_t fingerprint;
    get(p, fingerprint);
    uint32_t provenanceLen;
    get(p, provenanceLen);

    in.seekg(0, std::ios::end);
    const std::streamoff fileSize = in.tellg();
    const uint64_t head = static_cast<uint64_t>(kHeaderBytes) + provenanceLen + sizeof(blMaterialMap::Map);
    if (fileSize < 0 || static_cast<uint64_t>(fileSize) < head ||
        (static_cast<uint64_t>(fileSize) - head) % sizeof(blMaterialMap::LeafPos) != 0)
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": " << fileSize << " bytes, expected " << head << " + 16*n (" << kHeaderBytes
          << " header + " << provenanceLen << " provenance + " << sizeof(blMaterialMap::Map)
          << " map + whole LeafPos records)";
    nPos = (static_cast<uint64_t>(fileSize) - head) / sizeof(blMaterialMap::LeafPos);

    std::string provenance(provenanceLen, '\0');
    in.seekg(static_cast<std::streamoff>(kHeaderBytes));
    if (provenanceLen > 0)
      in.read(provenance.data(), provenanceLen);
    return blMaterialMap::FileHeader{geometryTag, beamPipeTag, mapVersion, fingerprint, provenance, {}};
  }

  // the appendix, read at the current position: nPos records, strictly ascending in rawId
  void readPositions(std::ifstream& in,
                     const std::string& path,
                     std::size_t nPos,
                     std::vector<blMaterialMap::LeafPos>& out) {
    if (nPos == 0)
      return;
    static_assert(sizeof(blMaterialMap::LeafPos) == 16);
    out.resize(nPos);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(nPos * sizeof(blMaterialMap::LeafPos)));
    if (!in)
      throw cms::Exception("BLMaterialMap") << "\"" << path << "\": position records read stopped at " << in.gcount()
                                            << " of " << nPos * sizeof(blMaterialMap::LeafPos) << " bytes";
    for (std::size_t i = 1; i < nPos; ++i)
      if (out[i].rawId <= out[i - 1].rawId)
        throw cms::Exception("BLMaterialMap")
            << "\"" << path << "\": position records not sorted by rawId: record " << i << " has rawId " << out[i].rawId
            << ", record " << i - 1 << " has " << out[i - 1].rawId;
  }
}  // namespace

namespace blMaterialMap {
  void writeFile(const std::string& path,
                 const Map& map,
                 std::string_view geometryTag,
                 std::string_view beamPipeTag,
                 uint32_t mapVersion,
                 uint64_t fingerprint,
                 std::string_view provenance,
                 const std::vector<LeafPos>* sensorPositions) {
    const auto checkTag = [&](std::string_view tag, const char* which) {
      if (tag.size() >= static_cast<size_t>(kTagBytes))
        throw cms::Exception("BLMaterialMap") << which << " \"" << tag << "\" is " << tag.size()
                                              << " characters; at most " << kTagBytes - 1 << " fit the 16-byte field";
      if (tag.find(' ') != std::string_view::npos || tag.find('\0') != std::string_view::npos)
        throw cms::Exception("BLMaterialMap")
            << which << " \"" << tag << "\" contains a space or a NUL; the tag fields hold neither";
    };
    checkTag(geometryTag, "geometry tag");
    checkTag(beamPipeTag, "beam-pipe tag");
    if (provenance.size() > 0xffff'ffffUL)
      throw cms::Exception("BLMaterialMap") << "provenance of " << provenance.size() << " bytes; at most 2^32 - 1";

    char header[kHeaderBytes];
    char* p = header;
    std::memcpy(p, kMagic, sizeof(kMagic));
    p += sizeof(kMagic);
    put(p, kFormatVersion);
    put(p, static_cast<int32_t>(kNR));
    put(p, static_cast<int32_t>(kNZ));
    put(p, kDR);
    put(p, kDZ);
    put(p, kZMAX);
    putTag(p, geometryTag);
    putTag(p, beamPipeTag);
    put(p, mapVersion);
    put(p, uint32_t{0});  // reserved
    put(p, fingerprint);
    put(p, static_cast<uint32_t>(provenance.size()));

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
      throw cms::Exception("BLMaterialMap") << "cannot open \"" << path << "\" for writing";
    out.write(header, static_cast<std::streamsize>(kHeaderBytes));
    if (!provenance.empty())
      out.write(provenance.data(), static_cast<std::streamsize>(provenance.size()));
    out.write(reinterpret_cast<const char*>(&map), sizeof(Map));
    if (sensorPositions && !sensorPositions->empty()) {
      static_assert(sizeof(LeafPos) == 16);
      out.write(reinterpret_cast<const char*>(sensorPositions->data()),
                static_cast<std::streamsize>(sensorPositions->size() * sizeof(LeafPos)));
    }
    out.close();
    if (!out)
      throw cms::Exception("BLMaterialMap") << "writing \"" << path << "\" failed";
  }

  FileHeader readFile(const std::string& path, Map& map) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw cms::Exception("BLMaterialMap") << "cannot open \"" << path << "\" for reading";
    std::size_t nPos = 0;
    FileHeader hdr = parseHeader(in, path, nPos);  // leaves `in` at the body
    in.read(reinterpret_cast<char*>(&map), sizeof(Map));
    if (!in)
      throw cms::Exception("BLMaterialMap")
          << "\"" << path << "\": map body read stopped at " << in.gcount() << " of " << sizeof(Map) << " bytes";
    readPositions(in, path, nPos, hdr.sensorPositions);
    return hdr;
  }

  FileHeader readHeader(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw cms::Exception("BLMaterialMap") << "cannot open \"" << path << "\" for reading";
    std::size_t nPos = 0;
    FileHeader hdr = parseHeader(in, path, nPos);
    in.seekg(static_cast<std::streamoff>(sizeof(Map)), std::ios::cur);  // past the body: it is not read here
    readPositions(in, path, nPos, hdr.sensorPositions);
    return hdr;
  }
}  // namespace blMaterialMap
