#include "RecoTracker/PixelLowPtUtilities/src/LowPtClusterShapeSeedComparitor.cc"
#include <iostream>

#ifdef OLDCODE
void go(std::vector<GlobalPoint> g) {
  std::vector<GlobalVector> v = LowPtClusterShapeSeedComparitor::getGlobalDirs(g);
  for (int i = 0; i != 3; i++)
    std::cout << g[i] << std::endl;
  if (v.empty())
    std::cout << "failed" << std::endl;
  else
    for (int i = 0; i != 3; i++) {
      std::cout << v[i];
      std::cout << " norm " << v[i].mag2() << std::endl;
    }
}

int main() {
  std::vector<GlobalPoint> g1({GlobalPoint(1., 3., 0.), GlobalPoint(-1., 1., 1.), GlobalPoint(1., -1., 2.)});

  std::vector<GlobalPoint> g2({GlobalPoint(1., -1., 0.), GlobalPoint(-1., 1., 1.), GlobalPoint(1., -3., 2.)});

  std::vector<GlobalPoint> g3({GlobalPoint(1., 3., 0.), GlobalPoint(4., 1., -1.), GlobalPoint(1., -1., -2.)});

  go(g1);
  go(g2);
  go(g3);

  return 0;
}

#else
void go(GlobalPoint const* g) {
  GlobalVector v[3];
  bool ok = getGlobalDirs(g, v);
  for (int i = 0; i != 3; i++)
    std::cout << g[i] << std::endl;
  if (!ok)
    std::cout << "failed" << std::endl;
  else
    for (int i = 0; i != 3; i++) {
      std::cout << v[i];
      std::cout << " norm " << v[i].mag2() << std::endl;
    }
}

int main() {
  GlobalPoint g1[3] = {GlobalPoint(1., 3., 0.), GlobalPoint(-1., 1., 1.), GlobalPoint(1., -1., 2.)};

  GlobalPoint g2[3] = {GlobalPoint(1., -1., 0.), GlobalPoint(-1., 1., 1.), GlobalPoint(1., -3., 2.)};

  GlobalPoint g3[3] = {GlobalPoint(1., 3., 0.), GlobalPoint(4., 1., -1.), GlobalPoint(1., -1., -2.)};

  go(g1);
  go(g2);
  go(g3);

  return 0;
}
#endif
