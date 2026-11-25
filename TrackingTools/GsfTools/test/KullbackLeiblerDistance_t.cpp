#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <benchmark/benchmark.h>
#include <cassert>
#include <iostream>
#include <vector>

bool isAligned(const void* data, long alignment) {
  // check that the alignment is a power of two
  assert((alignment & (alignment - 1)) == 0);
  return ((long)data & (alignment - 1)) == 0;
}

typedef DistanceBetweenComponents<5> Distance;
typedef KullbackLeiblerDistance<5> KDistance;
typedef SingleGaussianState<5> GS;
typedef GS::Vector Vector;
typedef GS::Matrix Matrix;
typedef ROOT::Math::SMatrix<double, 6, 6, ROOT::Math::MatRepSym<double, 6> > Matrix6;

Distance const& distance() {
  static Distance* d = new KDistance;
  return *d;
}

Matrix buildCovariance(float y) {
  // build a resonable covariance matrix as JIJ

  Basic3DVector<float> axis(0.5, 1., 1);

  Surface::RotationType rot(axis, 0.5 * M_PI);

  Surface::PositionType pos(0., 0., 0.);

  Plane plane(pos, rot);
  LocalTrajectoryParameters tp(1., 1., y, 0., 0., 1.);

  JacobianLocalToCartesian jl2c(plane, tp);
  return ROOT::Math::SimilarityT(jl2c.jacobian(), Matrix6(ROOT::Math::SMatrixIdentity()));
  // return  ROOT::Math::Transpose(jl2c.jacobian())* jl2c.jacobian();
}

namespace {
  struct Fixture {
    Fixture()
        : gs1(Vector(1., 1., 1., 1., 1.), buildCovariance(1.)),
          gs2(Vector(2., 2., 2., 2., 2.), buildCovariance(2.)),
          vgs(10000) {
      vgs.front() = gs1;
      vgs.back() = gs2;
    }
    GS gs1;
    GS gs2;
    std::vector<GS> vgs;
  };

  void BM_DistanceSingle(benchmark::State& state) {
    Distance const& d = distance();
    Fixture f;
    for (auto _ : state) {
      double res = d(f.gs1, f.gs2);
      benchmark::DoNotOptimize(res);
    }
  }
  BENCHMARK(BM_DistanceSingle);

  void BM_DistanceFromVector(benchmark::State& state) {
    Distance const& d = distance();
    Fixture f;
    for (auto _ : state) {
      double res = d(f.vgs.front(), f.vgs.back());
      benchmark::DoNotOptimize(res);
    }
  }
  BENCHMARK(BM_DistanceFromVector);

  void BM_DistanceLoop(benchmark::State& state) {
    Distance const& d = distance();
    Fixture f;
    for (auto _ : state) {
      double res = 0;
      for (auto const& s : f.vgs)
        res += d(f.vgs.front(), s);
      benchmark::DoNotOptimize(res);
    }
  }
  BENCHMARK(BM_DistanceLoop);
}  // namespace

int main(int argc, char* argv[]) {
  std::cout << "size of  SingleGaussianState<5>" << sizeof(GS) << std::endl;
  std::cout << "size of  Matrix" << sizeof(Matrix) << std::endl;
  std::cout << "size of  Vector" << sizeof(Vector) << std::endl;

  Distance const& d = distance();

  GS* gs1 = new GS(Vector(1., 1., 1., 1., 1.), buildCovariance(1.));
  GS* gs0 = new GS(Vector(1., 1., 1., 0., 0.), Matrix(ROOT::Math::SMatrixIdentity()));
  GS* gsP = new GS(Vector(1., 1., 1., 10., 10.), Matrix(ROOT::Math::SMatrixIdentity()));
  GS* gs2 = new GS(Vector(2., 2., 2., 2., 2.), buildCovariance(2.));

  std::cout << "GS " << ((isAligned(gs0, 16)) ? "a " : "n ") << std::endl;
  std::cout << "cov " << ((isAligned(&gs0->covariance(), 16)) ? "a " : "n ") << std::endl;
  std::cout << "mean " << ((isAligned(&gs0->mean(), 16)) ? "a " : "n ") << std::endl;
  std::cout << "weightM " << ((isAligned(&gs0->weightMatrix(), 16)) ? "a " : "n ") << std::endl;

  std::vector<GS> vgs(10000);
  vgs.front() = *gs1;
  vgs.back() = *gs2;

  double res = d(*gs0, *gsP);
  double res2 = d(*gs1, *gs2);
  std::cout << res << " " << res2 << std::endl;

  res = 0;
  for (auto const& s : vgs)
    res += d(vgs.front(), s);
  std::cout << res << std::endl;

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
