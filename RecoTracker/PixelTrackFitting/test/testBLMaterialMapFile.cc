// Unit test of the material-map file writer and reader (BLMaterialMapFile.h):
//  (1) a recognizable, non-trivial fill of the Map -- rho = (i % 251) * 0.25, the dedx triple from
//      three arithmetic sequences, exact zeros, and 1/3 for the rounding-sensitive bits --
//      round-trips through a file bit-exact;
//  (2) the header fields round-trip with it: the two tags, map version 7, a nonzero fingerprint, and a
//      provenance with spaces and a newline;
//  (3) malformed files are all refused with a cms::Exception("BLMaterialMap"): corrupted magic, an
//      unknown format version, a foreign lattice (nR/nZ/dR/dZ/zMax), a nonzero reserved field, a file
//      truncated to half the body, a half LeafPos record, and an appendix out of rawId order;
//  (4) writeFile refuses an unwritable path and a tag that does not fit the 16-byte field (too long, or
//      carrying a space), while a 15-character tag round-trips;
//  (5) readHeader returns readFile's header and positions without reading the body.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"

using blMaterialMap::DeDx;
using blMaterialMap::FileHeader;
using blMaterialMap::Map;

namespace {
  const std::string kPath = "testBLMaterialMapFile.bin";
  const std::string kGeoTag = "T35-unit-test";
  const std::string kBPTag = "2030/v3-unit";
  constexpr uint32_t kMapVersion = 7;
  constexpr uint64_t kFingerprint = 0x0123456789abcdefULL;
  const std::string kProvenance =
      "the test fill of testBLMaterialMapFile: rho[i] = (i % 251) * 0.25\n"
      "dedx from three arithmetic sequences, exact zeros, and 1/3 for the bit check";
  // sensor reference positions: 3 records, exercising negative buckets too
  const std::vector<blMaterialMap::LeafPos> kPositions = {
      {303042565u, 268, -2701, -166575}, {355480620u, -413, 1542, 26520}, {443286622u, 107035, -1, 103682}};

  int fail(const std::string& what) {
    std::printf("FAIL: %s\n", what.c_str());
    return EXIT_FAILURE;
  }

  // overwrite the n bytes at offset off of `path` (to corrupt a header field)
  bool patch(const std::string& path, std::streamoff off, const void* data, size_t n) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(off);
    f.write(static_cast<const char*>(data), n);
    return f.good();
  }

  // readFile(path) must refuse the file with a cms::Exception of category "BLMaterialMap"
  bool refused(const std::string& path, std::string& why) {
    auto map = std::make_unique<Map>();
    try {
      blMaterialMap::readFile(path, *map);
    } catch (const cms::Exception& e) {
      if (e.category() == "BLMaterialMap")
        return true;
      why = "wrong category \"" + e.category() + "\"";
      return false;
    }
    why = "no exception";
    return false;
  }
}  // namespace

int main() {
  // (1)+(2) the round-trip (the Maps live on the heap: 2.2 MB each)
  auto out = std::make_unique<Map>();
  for (int i = 0; i < blMaterialMap::kSize; ++i) {
    out->rho[i] = 0.25f * float(i % 251);  // exact zeros at the multiples of 251
    out->dedx[i] = DeDx{0.5f * float(i % 97), 4.f + 0.01f * float(i % 113), 0.25f * float(i % 89)};
  }
  out->rho[12345] = 1.f / 3.f;                   // not representable: proves bit-fidelity
  out->dedx[54321] = DeDx{1.f / 3.f, 0.f, 0.f};  // an all-zero cell next to one more 1/3
  try {
    blMaterialMap::writeFile(kPath, *out, kGeoTag, kBPTag, kMapVersion, kFingerprint, kProvenance, &kPositions);
  } catch (const cms::Exception& e) {
    return fail(std::string("writeFile threw: ") + e.what());
  }
  std::error_code ec;
  const uint64_t wantBytes = 84 + kProvenance.size() + sizeof(Map) + kPositions.size() * sizeof(blMaterialMap::LeafPos);
  if (std::filesystem::file_size(kPath, ec) != wantBytes)
    return fail(kPath + " is not 84 + provenance + " + std::to_string(sizeof(Map)) + " bytes");

  auto in = std::make_unique<Map>();
  FileHeader h;
  try {
    h = blMaterialMap::readFile(kPath, *in);
  } catch (const cms::Exception& e) {
    return fail(std::string("readFile refused the file just written: ") + e.what());
  }
  if (std::memcmp(in.get(), out.get(), sizeof(Map)) != 0)
    return fail("the Map did not round-trip bit-exact");
  if (h.geometryTag != kGeoTag)
    return fail("geometryTag \"" + h.geometryTag + "\", expected \"" + kGeoTag + "\"");
  if (h.beamPipeTag != kBPTag)
    return fail("beamPipeTag \"" + h.beamPipeTag + "\", expected \"" + kBPTag + "\"");
  if (h.mapVersion != kMapVersion)
    return fail("mapVersion " + std::to_string(h.mapVersion) + ", expected " + std::to_string(kMapVersion));
  if (h.fingerprint != kFingerprint) {
    char buf[96];
    std::snprintf(buf,
                  sizeof(buf),
                  "fingerprint 0x%016llx, expected 0x%016llx",
                  static_cast<unsigned long long>(h.fingerprint),
                  static_cast<unsigned long long>(kFingerprint));
    return fail(buf);
  }
  if (h.provenance != kProvenance)
    return fail("the provenance (spaces, newline) did not round-trip: \"" + h.provenance + "\"");
  if (h.sensorPositions != kPositions)
    return fail("the sensor reference positions did not round-trip");

  // (5) readHeader: the same header, without the body
  try {
    const FileHeader hh = blMaterialMap::readHeader(kPath);
    if (hh.geometryTag != h.geometryTag or hh.beamPipeTag != h.beamPipeTag or hh.mapVersion != h.mapVersion or
        hh.fingerprint != h.fingerprint or hh.provenance != h.provenance)
      return fail("readHeader returned other header fields than readFile");
    if (hh.sensorPositions != h.sensorPositions)
      return fail("readHeader returned other sensor positions than readFile");
  } catch (const cms::Exception& e) {
    return fail(std::string("readHeader threw on the good file: ") + e.what());
  }

  // the appendix is optional: a file without it reads back with an empty reference
  const std::string kNoPos = "testBLMaterialMapFile_nopos.bin";
  try {
    blMaterialMap::writeFile(kNoPos, *out, kGeoTag, kBPTag, kMapVersion, kFingerprint, kProvenance);
    auto in2 = std::make_unique<Map>();
    if (!blMaterialMap::readFile(kNoPos, *in2).sensorPositions.empty())
      return fail("a file without the appendix must read an empty sensorPositions");
    if (std::memcmp(in2.get(), out.get(), sizeof(Map)) != 0)
      return fail("the appendix-free Map did not round-trip bit-exact");
    if (!blMaterialMap::readHeader(kNoPos).sensorPositions.empty())
      return fail("readHeader must report an empty sensorPositions for the appendix-free file");
  } catch (const cms::Exception& e) {
    return fail(std::string("appendix-free round trip threw: ") + e.what());
  }
  std::filesystem::remove(kNoPos);

  // (3) malformed files, each one byte-level edit of the good file
  const std::string badMagic = "testBLMaterialMapFile_badMagic.bin";
  const std::string version2 = "testBLMaterialMapFile_version2.bin";
  const std::string nR249 = "testBLMaterialMapFile_nR249.bin";
  const std::string nZ249 = "testBLMaterialMapFile_nZ249.bin";
  const std::string badDR = "testBLMaterialMapFile_dR.bin";
  const std::string badDZ = "testBLMaterialMapFile_dZ.bin";
  const std::string badZMax = "testBLMaterialMapFile_zMax.bin";
  const std::string reservedSet = "testBLMaterialMapFile_reserved.bin";
  const std::string truncated = "testBLMaterialMapFile_truncated.bin";
  const std::string badTail = "testBLMaterialMapFile_badTail.bin";
  const std::string unsorted = "testBLMaterialMapFile_unsorted.bin";
  for (const std::string* f :
       {&badMagic, &version2, &nR249, &nZ249, &badDR, &badDZ, &badZMax, &reservedSet, &truncated, &badTail, &unsorted})
    std::filesystem::copy_file(kPath, *f, std::filesystem::copy_options::overwrite_existing);

  if (!patch(badMagic, 0, "BLMM0002", 8))  // magic at offset 0
    return fail("cannot patch " + badMagic);
  const uint32_t v2 = 2;
  if (!patch(version2, 8, &v2, sizeof(v2)))  // formatVersion at offset 8
    return fail("cannot patch " + version2);
  const int32_t nr = 249;
  if (!patch(nR249, 12, &nr, sizeof(nr)))  // nR at offset 12
    return fail("cannot patch " + nR249);
  const int32_t nz = 249;
  if (!patch(nZ249, 16, &nz, sizeof(nz)))  // nZ at offset 16
    return fail("cannot patch " + nZ249);
  const float foreignStep = 0.125f;
  if (!patch(badDR, 20, &foreignStep, sizeof(foreignStep)))  // dR at offset 20
    return fail("cannot patch " + badDR);
  if (!patch(badDZ, 24, &foreignStep, sizeof(foreignStep)))  // dZ at offset 24
    return fail("cannot patch " + badDZ);
  const float foreignZMax = 111.f;
  if (!patch(badZMax, 28, &foreignZMax, sizeof(foreignZMax)))  // zMax at offset 28
    return fail("cannot patch " + badZMax);
  const uint32_t one = 1;
  if (!patch(reservedSet, 68, &one, sizeof(one)))  // reserved at offset 68
    return fail("cannot patch " + reservedSet);
  std::filesystem::resize_file(truncated, 84 + kProvenance.size() + sizeof(Map) / 2);  // half the body
  std::filesystem::resize_file(badTail,
                               84 + kProvenance.size() + sizeof(Map) + 8);  // half a LeafPos record
  // the appendix out of order: the first two records swap their rawIds
  const std::streamoff appendix = 84 + kProvenance.size() + sizeof(Map);
  if (!patch(unsorted, appendix, &kPositions[1].rawId, sizeof(uint32_t)) or
      !patch(unsorted, appendix + sizeof(blMaterialMap::LeafPos), &kPositions[0].rawId, sizeof(uint32_t)))
    return fail("cannot patch " + unsorted);

  const std::pair<std::string, const char*> bad[] = {{badMagic, "corrupted magic"},
                                                     {version2, "format version 2"},
                                                     {nR249, "nR 249"},
                                                     {nZ249, "nZ 249"},
                                                     {badDR, "a foreign dR"},
                                                     {badDZ, "a foreign dZ"},
                                                     {badZMax, "a foreign zMax"},
                                                     {reservedSet, "reserved != 0"},
                                                     {truncated, "truncated body"},
                                                     {badTail, "half a position record"},
                                                     {unsorted, "an appendix out of rawId order"}};
  for (const auto& [file, what] : bad) {
    std::string why;
    if (!refused(file, why))
      return fail(file + " (" + what + "): expected cms::Exception(\"BLMaterialMap\"), got " + why);
  }
  for (const auto& [file, what] : bad)
    std::filesystem::remove(file);

  // (4) the writer's own refusals
  const std::string kTagTest = "testBLMaterialMapFile_tag.bin";
  const auto writeRefused = [&out](const std::string& path, const std::string& tag, std::string& why) -> bool {
    try {
      blMaterialMap::writeFile(path, *out, tag, kBPTag, kMapVersion, kFingerprint, kProvenance);
    } catch (const cms::Exception& e) {
      if (e.category() == "BLMaterialMap")
        return true;
      why = "wrong category \"" + e.category() + "\"";
      return false;
    }
    why = "no exception";
    return false;
  };
  const std::pair<std::pair<std::string, std::string>, const char*> badWrites[] = {
      {{"/nonexistent-dir/x.bin", kGeoTag}, "an unwritable path"},
      {{kTagTest, "0123456789abcdef"}, "a 16-character tag"},
      {{kTagTest, "T35 unit test"}, "a tag with a space"}};
  for (const auto& [args, what] : badWrites) {
    std::string why;
    if (!writeRefused(args.first, args.second, why))
      return fail(std::string(what) + ": expected cms::Exception(\"BLMaterialMap\") from writeFile, got " + why);
  }
  // 15 characters are the longest tag that fits the NUL-padded field
  const std::string kTag15 = "0123456789abcde";
  try {
    blMaterialMap::writeFile(kTagTest, *out, kTag15, kBPTag, kMapVersion, kFingerprint, kProvenance);
    if (blMaterialMap::readHeader(kTagTest).geometryTag != kTag15)
      return fail("a 15-character tag did not round-trip");
  } catch (const cms::Exception& e) {
    return fail(std::string("a 15-character tag was refused: ") + e.what());
  }
  std::filesystem::remove(kTagTest);
  std::filesystem::remove(kPath);

  std::printf("testBLMaterialMapFile passed\n");
  return EXIT_SUCCESS;
}
