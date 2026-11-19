// test of  GsfMaterialEffectsUpdator

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfCombinedMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsAdapter.h"

#include "TrackingTools/GsfTracking/interface/GsfBetheHeitlerUpdator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public MagneticField {
    M5T() : m(0., 0., 5.) {}
    virtual GlobalVector inTesla(const GlobalPoint&) const { return m; }

    GlobalVector m;
  };

}  // namespace

#include <benchmark/benchmark.h>
#include <iostream>
#include <string>

namespace {
  std::string file("BetheHeitler_cdfmom_nC6_O5.par");

  void updateStateBenchmark(benchmark::State& state, GsfMaterialEffectsUpdator* meu) {
    Basic3DVector<float> axis(0.5, 1., 1);

    Surface::RotationType rot(axis, 0.5 * M_PI);

    Surface::PositionType pos(0., 0., 0.);

    Plane::PlanePointer plane = Plane::build(pos, rot, new RectangularPlaneBounds(1., 1., 1));
    plane->setMediumProperties(MediumProperties(0.1, 0.3));
    M5T const m;

    int i = 0;
    for (auto _ : state) {
      state.PauseTiming();
      LocalTrajectoryParameters tp(1. / (10. + 0.01 * (i++ % 100000)), 1., 1., 0., 0., 1.);
      LocalTrajectoryError lerr(1., 1., 0.1, 0.1, 0.1);
      TrajectoryStateOnSurface tsos(tp, lerr, *plane, &m, SurfaceSideDefinition::beforeSurface);
      state.ResumeTiming();

      TrajectoryStateOnSurface tsos2 = meu->updateState(tsos, alongMomentum);
      benchmark::DoNotOptimize(tsos2);
    }
  }
}  // namespace

int main(int argc, char* argv[]) {
  benchmark::Initialize(&argc, argv);
  if (argc < 2) {
    std::cerr << "parameter file not given in input: default used" << std::endl;
  } else
    file = argv[1];

  std::cout << "using file " << file << std::endl;

  GsfBetheHeitlerUpdator bhu(file, 0);
  GsfBetheHeitlerUpdator bhu1(file, 1);
  GsfBetheHeitlerUpdator bhu2(file, 2);
  GsfMaterialEffectsAdapter msu(MultipleScatteringUpdator(bhu.mass()));

  GsfCombinedMaterialEffectsUpdator comb(msu, bhu);

  benchmark::RegisterBenchmark("MultipleScattering", updateStateBenchmark, &msu);
  benchmark::RegisterBenchmark("BetheHeitler0", updateStateBenchmark, &bhu);
  benchmark::RegisterBenchmark("BetheHeitler1", updateStateBenchmark, &bhu1);
  benchmark::RegisterBenchmark("BetheHeitler2", updateStateBenchmark, &bhu2);
  benchmark::RegisterBenchmark("Combined", updateStateBenchmark, &comb);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
