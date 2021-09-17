//#define MK_DEBUG
#include "RecoTracker/TkSeedGenerator/interface/FastCircleFit.h"

#include <iostream>

namespace {
  void test(const std::vector<GlobalPoint>& points, const std::vector<GlobalError>& errors) {
    FastCircleFit c(points, errors);

    std::cout << "origin " << c.x0() << " " << c.y0() << " radius " << c.rho() << " chi2 " << c.chi2() << std::endl;
  };
}  // namespace

int main() {
  {
    std::vector<GlobalPoint> points = {{GlobalPoint(0., -1., 0.), GlobalPoint(-1., 0., 0.), GlobalPoint(0., 1., 0.)}};
    std::vector<GlobalError> errors = {
        {GlobalError(1, 0, 1, 0, 0, 1), GlobalError(1, 0, 1, 0, 0, 1), GlobalError(1, 0, 1, 0, 0, 1)}};

    test(points, errors);
  }

  {
    std::vector<GlobalPoint> points = {{GlobalPoint(0.1, 0., 0.), GlobalPoint(-0.9, 1., 0.), GlobalPoint(0.1, 2., 0.)}};
    std::vector<GlobalError> errors = {
        {GlobalError(1, 0, 1, 0, 0, 1), GlobalError(1, 0, 1, 0, 0, 1), GlobalError(1, 0, 1, 0, 0, 1)}};

    test(points, errors);
  }

  {
    std::vector<GlobalPoint> points = {{GlobalPoint(-3.18318, 0.0479436, 0.),
                                        GlobalPoint(-6.74696, -0.303678, 0.),
                                        GlobalPoint(-10.7557, -1.24485, 0.),
                                        GlobalPoint(-14.6143, -2.79196, 0.)}};
    std::vector<GlobalError> errors = {{GlobalError(2.37751e-07, 8.873e-07, 3.31145e-06, 0, 0, 5.22405e-06),
                                        GlobalError(6.10596e-09, -5.41919e-08, 4.80966e-07, 0, 0, 7.44661e-06),
                                        GlobalError(6.41271e-09, -8.96614e-08, 1.25363e-06, 0, 0, 4.73874e-06),
                                        GlobalError(1.19665e-06, 2.78201e-07, 2.80112e-07, 0, 0, 2.78938e-08)}};

    test(points, errors);
  }
  {
    std::vector<GlobalPoint> points = {{GlobalPoint(-3.18318, 0.0479436, 0.),
                                        GlobalPoint(-6.74696, -0.303678, 0.),
                                        GlobalPoint(-10.7557, -1.24485, 0.),
                                        GlobalPoint(-14.699, -2.83295, 0.)}};
    std::vector<GlobalError> errors = {{GlobalError(2.37751e-07, 8.873e-07, 3.31145e-06, 0, 0, 5.22405e-06),
                                        GlobalError(6.10596e-09, -5.41919e-08, 4.80966e-07, 0, 0, 7.44661e-06),
                                        GlobalError(6.41271e-09, -8.96614e-08, 1.25363e-06, 0, 0, 4.73874e-06),
                                        GlobalError(3.14525e-06, 2.64413e-07, 4.33922e-07, 0, 0, 5.69573e-08)}};

    test(points, errors);
  }

  return 0;
}
