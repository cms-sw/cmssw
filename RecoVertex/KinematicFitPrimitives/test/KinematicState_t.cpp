#include "RecoVertex/KinematicFitPrimitives/interface/TrackKinematicStatePropagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include <iostream>

class ConstMagneticField : public MagneticField {
public:
  GlobalVector inTesla(const GlobalPoint&) const override { return GlobalVector(0, 0, 4); }
};

int main() {
  using namespace std;

  MagneticField* field = new ConstMagneticField;
  GlobalPoint gp(0, 0, 0);
  GlobalPoint gp2(1, 1, 1);

  GlobalPoint stars(-1.e12, -1.e12, -1);

  GlobalVector gv(1, 1, 1);
  GlobalTrajectoryParameters gtp(gp, gv, 1, field);
  double v[15] = {0.01, -0.01, 0., 0., 0., 0.01, 0., 0., 0., 0.01, 0., 0., 1., 0., 1.};
  AlgebraicSymMatrix55 gerr(v, 15);
  BoundPlane* plane = new BoundPlane(gp, Surface::RotationType());

  TrajectoryStateOnSurface ts(gtp, gerr, *plane);

  cout << "ts.globalPosition() " << ts.globalPosition() << endl;
  cout << "ts.globalMomentum() " << ts.globalMomentum() << endl;
  cout << "ts.localMomentum()  " << ts.localMomentum() << endl;
  cout << "ts.transverseCurvature()  " << ts.transverseCurvature() << endl;
  cout << "ts inversePtErr " << TrajectoryStateAccessor(*ts.freeState()).inversePtError() << std::endl;
  cout << "ts curv err\n" << ts.curvilinearError().matrix() << std::endl;
  cout << "ts cart err\n" << ts.cartesianError().matrix() << std::endl;

  const FreeTrajectoryState* fts = ts.freeTrajectoryState();
  assert(fts);

  KinematicState ks(*fts, 0.151, 0.01);

  std::cout << "ks Par" << ks.kinematicParameters().vector() << std::endl;
  std::cout << "ks Err" << ks.kinematicParametersError().matrix() << std::endl;
  std::cout << "ks charge " << ks.particleCharge() << std::endl;
  cout << "ks.globalPosition() " << ks.globalPosition() << endl;
  cout << "ks globalMomentum() " << ks.globalMomentum() << endl;
  cout << "ks curv err\n" << ks.freeTrajectoryState().curvilinearError().matrix() << std::endl;
  cout << "ks cart err\n" << ks.freeTrajectoryState().cartesianError().matrix() << std::endl;

  auto c = ks.kinematicParametersError().matrix();
  c(0, 6) = -0.0001;
  KinematicParametersError kpe(c);
  KinematicState ks2(ks.kinematicParameters(), kpe, ks.particleCharge(), ks.magneticField());

  std::cout << "ks2 Par" << ks2.kinematicParameters().vector() << std::endl;
  std::cout << "ks2 Err" << ks2.kinematicParametersError().matrix() << std::endl;
  std::cout << "ks2 charge " << ks2.particleCharge() << std::endl;
  cout << "ks2.globalPosition() " << ks2.globalPosition() << endl;
  cout << "ks2 globalMomentum() " << ks2.globalMomentum() << endl;
  cout << "ks2 curv err\n" << ks2.freeTrajectoryState().curvilinearError().matrix() << std::endl;
  cout << "ks2 cart err\n" << ks2.freeTrajectoryState().cartesianError().matrix() << std::endl;

  TrackKinematicStatePropagator p;
  {
    auto ok = p.willPropagateToTheTransversePCA(ks, stars);
    std::cout << "\npropagate ks to stars " << (ok ? "ok\n" : "nope\n") << std::endl;
  }

  auto ok = p.willPropagateToTheTransversePCA(ks, gp2);
  std::cout << "\npropagate ks " << (ok ? "ok\n" : "nope\n") << std::endl;
  auto kst = p.propagateToTheTransversePCA(ks, gp2);

  std::cout << "kst Par" << kst.kinematicParameters().vector() << std::endl;
  std::cout << "kst Err" << kst.kinematicParametersError().matrix() << std::endl;
  cout << "kst.globalPosition() " << kst.globalPosition() << endl;
  cout << "kst globalMomentum() " << kst.globalMomentum() << endl;
  cout << "kst curv err\n" << kst.freeTrajectoryState().curvilinearError().matrix() << std::endl;
  cout << "kst cart err\n" << kst.freeTrajectoryState().cartesianError().matrix() << std::endl;

  auto ok2 = p.willPropagateToTheTransversePCA(ks2, gp2);
  std::cout << "\npropagate ks2 " << (ok2 ? "ok\n" : "nope\n") << std::endl;
  auto kst2 = p.propagateToTheTransversePCA(ks2, gp2);

  std::cout << "kst2 Par" << kst2.kinematicParameters().vector() << std::endl;
  std::cout << "kst2 Err" << kst2.kinematicParametersError().matrix() << std::endl;
  cout << "kst2.globalPosition() " << kst2.globalPosition() << endl;
  cout << "kst2 globalMomentum() " << kst2.globalMomentum() << endl;
  cout << "kst2 curv err\n" << kst2.freeTrajectoryState().curvilinearError().matrix() << std::endl;
  cout << "kst2 cart err\n" << kst2.freeTrajectoryState().cartesianError().matrix() << std::endl;

  return 0;
}
