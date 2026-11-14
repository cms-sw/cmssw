#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include <Geometry/CommonTopologies/interface/GeomDet.h>

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include <iostream>

class ConstMagneticField final : public MagneticField {
public:
  ConstMagneticField() { setNominalValue(); }
  GlobalVector inTesla(const GlobalPoint&) const override { return GlobalVector(0, 0, 4); }
};

#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"

typedef ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5> > Matrix5;
typedef ROOT::Math::SMatrix<double, 6, 6, ROOT::Math::MatRepSym<double, 6> > Matrix6;

Matrix5 buildCovariance(float y) {
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

// A fake Det class

class MyDet : public TrackerGeomDet {
public:
  MyDet(BoundPlane* bp, DetId id) : TrackerGeomDet(bp) { setDetId(id); }

  virtual std::vector<const GeomDet*> components() const { return std::vector<const GeomDet*>(); }

  /// Which subdetector
  virtual SubDetector subDetector() const { return GeomDetEnumerators::DT; }
};

// a 2d hit

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"

class My2DHit : public GenericTransientTrackingRecHit {
public:
  My2DHit(const GeomDet& geom, TrackingRecHit* rh) : GenericTransientTrackingRecHit(geom, rh) {}

  virtual void getKfComponents(KfComponentsHolder& holder) const {
    HelpertRecHit2DLocalPos::getKfComponents(holder, *hit(), *det());
  }
};

void print(TrajectoryStateOnSurface const& ts) {
  using namespace std;

  cout << "transverseCurvature  " << ts.transverseCurvature() << endl;
  cout << "globalMomentum       " << ts.globalMomentum() << endl;
  cout << "localMomentum        " << ts.localMomentum() << endl;
  cout << "localPosition        " << ts.localPosition() << endl;
  cout << "localError           " << ts.localError().matrix() << endl;
  cout << endl;
}

#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>

struct KFUTest {
  TrajectoryStateUpdator const& tsu;

  KFUTest(TrajectoryStateUpdator const* itsu) : tsu(*itsu) {}

  void print(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& hit) const {
    TrajectoryStateOnSurface tsn = tsu.update(tsos, hit);
    ::print(tsn);
  }

  void time(benchmark::State& state, const TrajectoryStateOnSurface& tsos, const TrackingRecHit& hit) const {
    for (auto _ : state) {
      TrajectoryStateOnSurface tsn = tsu.update(tsos, hit);
      benchmark::DoNotOptimize(tsn);
    }
  }
};

struct Chi2Test {
  MeasurementEstimator const& chi2;
  mutable std::pair<bool, double> res;
  Chi2Test(MeasurementEstimator const* ichi2) : chi2(*ichi2) {}

  void print(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& hit) const {
    res = chi2.estimate(tsos, hit);
    std::cout << "chi2 " << res.second << std::endl;
  }

  void time(benchmark::State& state, const TrajectoryStateOnSurface& tsos, const TrackingRecHit& hit) const {
    for (auto _ : state) {
      res = chi2.estimate(tsos, hit);
      benchmark::DoNotOptimize(res);
    }
  }
};

int main(int argc, char* argv[]) {
  benchmark::Initialize(&argc, argv);
  MagneticField* field = new ConstMagneticField;
  GlobalPoint gp(0, 0, 0);
  GlobalVector gv(1, 1, 1);
  GlobalTrajectoryParameters gtp(gp, gv, 1, field);
  CurvilinearTrajectoryError gerr(buildCovariance(1.));
  BoundPlane* plane = new BoundPlane(gp, Surface::RotationType());
  GeomDet* det = new MyDet(plane, 41);
  GeomDet* gdet = new MyDet(plane, 40);

  TrajectoryStateOnSurface ts(gtp, gerr, *plane);
  print(ts);

  LocalPoint lp(0, 0, 0);
  LocalVector lv(1, 1, 1);
  LocalTrajectoryParameters ltp(lp, lv, 1);
  LocalTrajectoryError ler(0.1, 0.1, 0.01, 0.05, 0.1);
  TrajectoryStateOnSurface ts2(ltp, ler, *plane, field);
  print(ts2);

  LocalPoint m(0.1, 0.1, 0);
  LocalError e(0.2, -0.05, 0.1);

  OmniClusterRef cref;
  SiPixelRecHit::ClusterRef pref;
  SiStripRecHit2D dummy;
  SiPixelRecHit hitpx(m, e, 1., *det, pref);
  SiStripRecHit1D hit1d(m, e, *det, cref);
  SiStripRecHit2D hit2d(m, e, *det, cref);
  ProjectedSiStripRecHit2D hitpj(m, e, *gdet, hit2d);
  TrackingRecHit* hit = new SiStripMatchedRecHit2D(m, e, *det, &dummy, &dummy);
  TransientTrackingRecHit* thit = new My2DHit(*det, hit);

  KFUTest kt(new KFUpdator());
  Chi2Test chi2(new Chi2MeasurementEstimator(10.));

  std::cout << "\n** KFU ** \n" << std::endl;

  kt.print(ts, *thit);
  kt.print(ts2, *thit);

  kt.print(ts, hit2d);
  kt.print(ts2, hit2d);

  kt.print(ts, hitpx);
  kt.print(ts2, hitpx);

  kt.print(ts, hitpj);
  kt.print(ts2, hitpj);

  kt.print(ts, hit1d);
  kt.print(ts2, hit1d);

  std::cout << "\n** Chi2 ** \n" << std::endl;

  chi2.print(ts, *thit);
  chi2.print(ts2, *thit);

  chi2.print(ts, hit2d);
  chi2.print(ts2, hit2d);

  chi2.print(ts, hitpx);
  chi2.print(ts2, hitpx);

  chi2.print(ts, hitpj);
  chi2.print(ts2, hitpj);

  chi2.print(ts, hit1d);
  chi2.print(ts2, hit1d);

  benchmark::RegisterBenchmark("KFUpdator/ts", [&](benchmark::State& state) { kt.time(state, ts, *thit); });
  benchmark::RegisterBenchmark("KFUpdator/ts2", [&](benchmark::State& state) { kt.time(state, ts2, *thit); });
  benchmark::RegisterBenchmark("Chi2/ts", [&](benchmark::State& state) { chi2.time(state, ts, *thit); });
  benchmark::RegisterBenchmark("Chi2/ts2", [&](benchmark::State& state) { chi2.time(state, ts2, *thit); });
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
