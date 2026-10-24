#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "TrackingTools/GsfTools/interface/SingleGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"

#include <benchmark/benchmark.h>

namespace {
  MultiGaussianState1D makeState() {
    MultiGaussianState1D::SingleState1dContainer v(6);
    v[0] = SingleGaussianState1D(0., 1., 1.);
    v[1] = SingleGaussianState1D(0., 2., 0.5);
    v[2] = SingleGaussianState1D(0.2, 2., 0.5);
    v[3] = SingleGaussianState1D(-0.2, 2., 0.5);
    v[4] = SingleGaussianState1D(0., 4., 1.0);
    v[5] = SingleGaussianState1D(0.1, 4., 0.3);
    return MultiGaussianState1D(v);
  }

  void BM_GaussianSumMode(benchmark::State& state) {
    MultiGaussianState1D mgs = makeState();
    GaussianSumUtilities1D gsu(mgs);
    for (auto _ : state) {
      auto sg1 = gsu.mode();
      benchmark::DoNotOptimize(sg1);
    }
  }
  BENCHMARK(BM_GaussianSumMode);
}  // namespace

BENCHMARK_MAIN();
