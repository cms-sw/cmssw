#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "TrackingTools/GsfTools/interface/SingleGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"

#include "FWCore/Utilities/interface/HRRealTime.h"
#include <iostream>
#include <vector>

namespace {
  struct LocalTimer {
    ~LocalTimer() { std::cout << "elapsed time " << t << std::endl; }
    edm::HRTimeType t;
  };

  LocalTimer timer;

}  // namespace

void st() { timer.t = edm::hrRealTime(); }

void en() { timer.t = edm::hrRealTime() - timer.t; }

int main(int argc, char* argv[]) {
  MultiGaussianState1D::SingleState1dContainer v(6);
  v[0] = SingleGaussianState1D(0., 1., 1.);
  v[1] = SingleGaussianState1D(0., 2., 0.5);
  v[2] = SingleGaussianState1D(0.2, 2., 0.5);
  v[3] = SingleGaussianState1D(-0.2, 2., 0.5);
  v[4] = SingleGaussianState1D(0., 4., 1.0);
  v[5] = SingleGaussianState1D(0.1, 4., 0.3);

  MultiGaussianState1D mgs(v);

  // call once to inizialite compiler stuff
  {
    GaussianSumUtilities1D gsu1(mgs);
    std::cout << gsu1.mode().mean() << std::endl;
  }

  GaussianSumUtilities1D gsu(mgs);
  st();
  const SingleGaussianState1D& sg1 = gsu.mode();
  en();

  std::cout << sg1.mean() << " " << sg1.standardDeviation() << " " << sg1.weight() << std::endl;

  return 0;
}
