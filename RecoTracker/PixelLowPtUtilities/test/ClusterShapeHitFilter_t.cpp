// ClusterShapeHitFilter test
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {
  const std::string dataDir("RecoTracker/PixelLowPtUtilities/data/");

  // the limits as they were loaded from the ASCII files before they were moved to the conditions DB
  void loadPixelLimitsFromFile(std::string const& file, PixelLimits* plim) {
    std::ifstream inFile(edm::FileInPath(dataDir + file).fullPath());
    int part, dx, dy;
    while (inFile >> part >> dx >> dy) {
      auto& pl = plim[PixelKeys(part, dx, dy)];
      for (int b = 0; b < 2; b++)
        for (int d = 0; d < 2; d++)
          for (int k = 0; k < 2; k++)
            inFile >> pl.data[b][d][k];
      double f;
      int n;
      inFile >> f >> n >> f >> n;  // density, points (unused)
    }
  }

  // same content as written by CondTools/SiPixel/plugins/SiPixelClusterShapeLimitsWriter.cc
  std::vector<SiPixelClusterShapeLimits::Entry> entriesFromFile(std::string const& file) {
    std::ifstream inFile(edm::FileInPath(dataDir + file).fullPath());
    std::vector<SiPixelClusterShapeLimits::Entry> entries;
    std::string line;
    while (std::getline(inFile, line)) {
      std::istringstream iss(line);
      SiPixelClusterShapeLimits::Entry entry;
      if (!(iss >> entry.part >> entry.dx >> entry.dy))
        continue;
      entry.limits.resize(SiPixelClusterShapeLimits::kNLimits);
      for (auto& limit : entry.limits)
        iss >> limit;
      entries.push_back(entry);
    }
    return entries;
  }

  // one table per file, in the same order
  SiPixelClusterShapeLimits makeLimits(std::vector<std::string> const& files) {
    SiPixelClusterShapeLimits limits;
    for (auto const& file : files)
      limits.addTable(file, entriesFromFile(file));
    limits.addRule({SiPixelClusterShapeLimits::kBPix, SiPixelClusterShapeLimits::kAny, 0});
    limits.addRule({SiPixelClusterShapeLimits::kFPix, SiPixelClusterShapeLimits::kAny, 0});
    return limits;
  }
}  // namespace

namespace test {
  namespace ClusterShapeHitFilterTest {
    int test() {
      const std::string use_PixelShapeFile("RecoTracker/PixelLowPtUtilities/data/pixelShapePhase0.par");

      ClusterShapeHitFilter filter(makeLimits({"pixelShapePhase0.par"}));

      const float out = 10e12;
      const float eps = 0.01;
      std::cout << "dump strip limits" << std::endl;
      for (int i = 0; i != StripKeys::N + 1; i++) {
        assert(!filter.stripLimits[i].isInside(out));
        assert(!filter.stripLimits[i].isInside(-out));
        std::cout << i << ": ";
        float const* p = filter.stripLimits[i].data[0];
        if (p[1] < 1.e9) {
          assert(filter.stripLimits[i].isInside(p[0] + eps));
          assert(filter.stripLimits[i].isInside(p[3] - eps));
        }
        for (int j = 0; j != 4; ++j)
          std::cout << p[j] << ", ";
        std::cout << std::endl;
      }

      const std::pair<float, float> out1(out, out), out2(-out, -out);
      std::cout << "\ndump pixel limits" << std::endl;
      for (int i = 0; i != PixelKeys::N + 1; i++) {
        assert(!filter.pixelLimits[0][i].isInside(out1));
        assert(!filter.pixelLimits[0][i].isInside(out2));
        std::cout << i << ": ";
        float const* p = filter.pixelLimits[0][i].data[0][0];
        if (p[1] < 1.e9) {
          assert(filter.pixelLimits[0][i].isInside(std::pair<float, float>(p[0] + eps, p[3] - eps)));
          assert(filter.pixelLimits[0][i].isInside(std::pair<float, float>(p[5] - eps, p[6] + eps)));
        }
        for (int j = 0; j != 8; ++j)
          std::cout << p[j] << ", ";
        std::cout << std::endl;
      }

      // the limits loaded from the payload must be bit-by-bit identical to the ones loaded from the files
      for (auto const& files :
           std::vector<std::vector<std::string>>{{"pixelShapePhase0.par"},
                                                 {"pixelShapePhase1_noL1.par", "pixelShapePhase1_loose.par"},
                                                 {"ITShapePhase2_all.par"}}) {
        ClusterShapeHitFilter payloadFilter(makeLimits(files));
        assert(payloadFilter.pixelLimits.size() == files.size());
        for (unsigned int t = 0; t < files.size(); ++t) {
          std::cout << "\ncompare pixel limits from payload and from file " << files[t] << std::endl;
          PixelLimits fileLimits[PixelKeys::N + 1];
          loadPixelLimitsFromFile(files[t], fileLimits);
          for (int i = 0; i != PixelKeys::N + 1; i++)
            assert(std::memcmp(payloadFilter.pixelLimits[t][i].data, fileLimits[i].data, sizeof(fileLimits[i].data)) ==
                   0);
        }
      }

      return 0;
    }
  }  // namespace ClusterShapeHitFilterTest
}  // namespace test

int main() { return test::ClusterShapeHitFilterTest::test(); }
