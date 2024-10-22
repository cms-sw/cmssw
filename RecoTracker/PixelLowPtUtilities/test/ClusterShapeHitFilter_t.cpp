// ClusterShapeHitFilter test
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include <iostream>
#include <cassert>

namespace test {
  namespace ClusterShapeHitFilterTest {
    int test() {
      const std::string use_PixelShapeFile("RecoTracker/PixelLowPtUtilities/data/pixelShapePhase0.par");

      ClusterShapeHitFilter filter("RecoTracker/PixelLowPtUtilities/data/pixelShapePhase0.par",
                                   "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase0.par");

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
        assert(!filter.pixelLimits[i].isInside(out1));
        assert(!filter.pixelLimits[i].isInside(out2));
        std::cout << i << ": ";
        float const* p = filter.pixelLimits[i].data[0][0];
        if (p[1] < 1.e9) {
          assert(filter.pixelLimits[i].isInside(std::pair<float, float>(p[0] + eps, p[3] - eps)));
          assert(filter.pixelLimits[i].isInside(std::pair<float, float>(p[5] - eps, p[6] + eps)));
        }
        for (int j = 0; j != 8; ++j)
          std::cout << p[j] << ", ";
        std::cout << std::endl;
      }

      return 0;
    }
  }  // namespace ClusterShapeHitFilterTest
}  // namespace test

int main() { return test::ClusterShapeHitFilterTest::test(); }
