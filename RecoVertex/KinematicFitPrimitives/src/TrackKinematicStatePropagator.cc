#include "RecoVertex/KinematicFitPrimitives/interface/TrackKinematicStatePropagator.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/OpenBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

KinematicState TrackKinematicStatePropagator::propagateToTheTransversePCA(const KinematicState& state,
                                                                          const GlobalPoint& referencePoint) const {
  if (state.particleCharge() == 0.) {
    return propagateToTheTransversePCANeutral(state, referencePoint);
  } else {
    return propagateToTheTransversePCACharged(state, referencePoint);
  }
}

namespace {
  inline pair<HelixBarrelPlaneCrossingByCircle, BoundPlane::BoundPlanePointer> planeCrossing(
      const FreeTrajectoryState& state, const GlobalPoint& point) {
    typedef Point3DBase<double, GlobalTag> GlobalPointDouble;
    typedef Vector3DBase<double, GlobalTag> GlobalVectorDouble;

    GlobalPoint inPos = state.position();
    GlobalVector inMom = state.momentum();
    double kappa = state.transverseCurvature();
    auto bz = state.parameters().magneticFieldInInverseGeV(point).z();
    if (std::abs(bz) < 1e-6) {
      LogDebug("RecoVertex/TrackKinematicStatePropagator") << "planeCrossing is not possible";
      return {HelixBarrelPlaneCrossingByCircle(inPos, inMom, kappa), BoundPlane::BoundPlanePointer()};
    }
    double fac = state.charge() / bz;

    GlobalVectorDouble xOrig2Centre(fac * inMom.y(), -fac * inMom.x(), 0.);
    GlobalVectorDouble xOrigProj(inPos.x(), inPos.y(), 0.);
    GlobalVectorDouble xRefProj(point.x(), point.y(), 0.);
    GlobalVectorDouble deltax = xRefProj - xOrigProj - xOrig2Centre;
    GlobalVectorDouble ndeltax = deltax.unit();

    PropagationDirection direction = anyDirection;
    Surface::PositionType pos(point);

    // Need to define plane with orientation as the ImpactPointSurface
    GlobalVector X(ndeltax.x(), ndeltax.y(), ndeltax.z());
    GlobalVector Y(0., 0., 1.);
    Surface::RotationType rot(X, Y);
    Plane::PlanePointer plane = Plane::build(pos, rot);
    HelixBarrelPlaneCrossingByCircle planeCrossing(HelixPlaneCrossing::PositionType(inPos.x(), inPos.y(), inPos.z()),
                                                   HelixPlaneCrossing::DirectionType(inMom.x(), inMom.y(), inMom.z()),
                                                   kappa,
                                                   direction);
    return std::pair<HelixBarrelPlaneCrossingByCircle, Plane::PlanePointer>(planeCrossing, plane);
  }
}  // namespace

bool TrackKinematicStatePropagator::willPropagateToTheTransversePCA(const KinematicState& state,
                                                                    const GlobalPoint& point) const {
  if (state.particleCharge() == 0.)
    return true;

  // copied from below...
  FreeTrajectoryState const& fState = state.freeTrajectoryState();
  std::pair<HelixBarrelPlaneCrossingByCircle, BoundPlane::BoundPlanePointer> cros = planeCrossing(fState, point);
  if (cros.second == nullptr)
    return false;

  HelixBarrelPlaneCrossingByCircle planeCrossing = cros.first;
  BoundPlane::BoundPlanePointer plane = cros.second;
  std::pair<bool, double> propResult = planeCrossing.pathLength(*plane);
  return propResult.first;
}

KinematicState TrackKinematicStatePropagator::propagateToTheTransversePCACharged(
    const KinematicState& state, const GlobalPoint& referencePoint) const {
  //first use the existing FTS propagator to obtain parameters at PCA
  //in transverse plane to the given point

  //final parameters and covariance

  //initial parameters as class and vectors:
  //making a free trajectory state and looking
  //for helix barrel plane crossing
  FreeTrajectoryState const& fState = state.freeTrajectoryState();
  const GlobalPoint& iP = referencePoint;
  std::pair<HelixBarrelPlaneCrossingByCircle, BoundPlane::BoundPlanePointer> cros = planeCrossing(fState, iP);
  if (cros.second == nullptr)
    return KinematicState();

  HelixBarrelPlaneCrossingByCircle planeCrossing = cros.first;
  BoundPlane::BoundPlanePointer plane = cros.second;
  std::pair<bool, double> propResult = planeCrossing.pathLength(*plane);
  if (!propResult.first) {
    LogDebug("RecoVertex/TrackKinematicStatePropagator") << "Propagation failed! State is invalid\n";
    return KinematicState();
  }
  double s = propResult.second;

  GlobalTrajectoryParameters const& inPar = state.trajectoryParameters();
  ParticleMass mass = state.mass();
  GlobalVector inMom = state.globalMomentum();

  HelixPlaneCrossing::PositionType xGen = planeCrossing.position(s);
  GlobalPoint nPosition(xGen.x(), xGen.y(), xGen.z());
  HelixPlaneCrossing::DirectionType pGen = planeCrossing.direction(s);
  pGen *= inMom.mag() / pGen.mag();
  GlobalVector nMomentum(pGen.x(), pGen.y(), pGen.z());
  AlgebraicVector7 par;
  AlgebraicSymMatrix77 cov;
  par(0) = nPosition.x();
  par(1) = nPosition.y();
  par(2) = nPosition.z();
  par(3) = nMomentum.x();
  par(4) = nMomentum.y();
  par(5) = nMomentum.z();
  par(6) = mass;

  //covariance matrix business
  //elements of 7x7 covariance matrix responcible for the mass and
  //mass - momentum projections corellations do change under such a transformation:
  //special Jacobian needed
  GlobalTrajectoryParameters fPar(nPosition, nMomentum, state.particleCharge(), state.magneticField());

  // check if correlation are present between mass and others
  bool thereIsNoCorr = true;

  for (auto i = 0; i < 6; ++i)
    thereIsNoCorr &= (0 == state.kinematicParametersError().matrix()(i, 6));

  if (thereIsNoCorr) {
    //  easy life
    AnalyticalCurvilinearJacobian prop(inPar, nPosition, nMomentum, s);
    AlgebraicSymMatrix55 cov2 = ROOT::Math::Similarity(prop.jacobian(), fState.curvilinearError().matrix());
    FreeTrajectoryState fts(fPar, CurvilinearTrajectoryError(cov2));

    return KinematicState(fts, state.mass(), std::sqrt(state.kinematicParametersError().matrix()(6, 6)));

    //KinematicState kRes(fts, state.mass(), std::sqrt(state.kinematicParametersError().matrix()(6,6)));
    //std::cout << "\n\ncart from final Kstate\n" << kRes.kinematicParametersError().matrix() << std::endl;
    // std::cout << "curv from final K\n" << kRes.freeTrajectoryState().curvilinearError().matrix() << std::endl;

  } else {
    JacobianCartesianToCurvilinear cart2curv(inPar);
    JacobianCurvilinearToCartesian curv2cart(fPar);

    AlgebraicMatrix67 ca2cu;
    AlgebraicMatrix76 cu2ca;
    ca2cu.Place_at(cart2curv.jacobian(), 0, 0);
    cu2ca.Place_at(curv2cart.jacobian(), 0, 0);
    ca2cu(5, 6) = 1;
    cu2ca(6, 5) = 1;

    //now both transformation jacobians: cartesian to curvilinear and back are done
    //We transform matrix to curv frame, then propagate it and translate it back to
    //cartesian frame.
    AlgebraicSymMatrix66 cov1 = ROOT::Math::Similarity(ca2cu, state.kinematicParametersError().matrix());

    /*
      std::cout << "\n\ncurv from Kstate\n" << cov1 << std::endl;
      std::cout << "curv from fts\n" << fState.curvilinearError().matrix() << std::endl;
    */

    //propagation jacobian
    AnalyticalCurvilinearJacobian prop(inPar, nPosition, nMomentum, s);
    AlgebraicMatrix66 pr;
    pr(5, 5) = 1;
    pr.Place_at(prop.jacobian(), 0, 0);

    //transportation
    AlgebraicSymMatrix66 cov2 = ROOT::Math::Similarity(pr, cov1);

    //now geting back to 7-parametrization from curvilinear
    cov = ROOT::Math::Similarity(cu2ca, cov2);

    /*
      std::cout << "curv prop \n" << cov2 << std::endl;
   std::cout << "cart prop\n" << cov << std::endl;
    */

    //return parameters as a kiematic state
    KinematicParameters resPar(par);
    KinematicParametersError resEr(cov);

    return KinematicState(resPar, resEr, state.particleCharge(), state.magneticField());

    /*
    KinematicState resK(resPar,resEr,state.particleCharge(), state.magneticField());
    std::cout << "\ncurv from K prop\n" << resK.freeTrajectoryState().curvilinearError().matrix() << std::endl;
    return resK;
  */
  }
}

KinematicState TrackKinematicStatePropagator::propagateToTheTransversePCANeutral(
    const KinematicState& state, const GlobalPoint& referencePoint) const {
  //new parameters vector and covariance:
  AlgebraicVector7 par;
  AlgebraicSymMatrix77 cov;

  //AlgebraicVector7 inStatePar = state.kinematicParameters().vector();
  GlobalTrajectoryParameters const& inPar = state.trajectoryParameters();

  //first making a free trajectory state and propagating it according
  //to the algorithm provided by Thomas Speer and Wolfgang Adam
  FreeTrajectoryState const& fState = state.freeTrajectoryState();

  GlobalPoint xvecOrig = fState.position();
  double phi = fState.momentum().phi();
  double theta = fState.momentum().theta();
  double xOrig = xvecOrig.x();
  double yOrig = xvecOrig.y();
  double zOrig = xvecOrig.z();
  double xR = referencePoint.x();
  double yR = referencePoint.y();

  double s2D = (xR - xOrig) * cos(phi) + (yR - yOrig) * sin(phi);
  double s = s2D / sin(theta);
  double xGen = xOrig + s2D * cos(phi);
  double yGen = yOrig + s2D * sin(phi);
  double zGen = zOrig + s2D / tan(theta);
  GlobalPoint xPerigee = GlobalPoint(xGen, yGen, zGen);

  //new parameters
  GlobalVector pPerigee = fState.momentum();
  par(0) = xPerigee.x();
  par(1) = xPerigee.y();
  par(2) = xPerigee.z();
  par(3) = pPerigee.x();
  par(4) = pPerigee.y();
  par(5) = pPerigee.z();
  // par(6) = inStatePar(7);
  par(6) = state.mass();

  //covariance matrix business:
  //everything lake it was before: jacobains are smart enouhg to
  //distinguish between neutral and charged states themselves

  GlobalTrajectoryParameters fPar(xPerigee, pPerigee, state.particleCharge(), state.magneticField());

  JacobianCartesianToCurvilinear cart2curv(inPar);
  JacobianCurvilinearToCartesian curv2cart(fPar);

  AlgebraicMatrix67 ca2cu;
  AlgebraicMatrix76 cu2ca;
  ca2cu.Place_at(cart2curv.jacobian(), 0, 0);
  cu2ca.Place_at(curv2cart.jacobian(), 0, 0);
  ca2cu(5, 6) = 1;
  cu2ca(6, 5) = 1;

  //now both transformation jacobians: cartesian to curvilinear and back are done
  //We transform matrix to curv frame, then propagate it and translate it back to
  //cartesian frame.
  AlgebraicSymMatrix66 cov1 = ROOT::Math::Similarity(ca2cu, state.kinematicParametersError().matrix());

  //propagation jacobian
  AnalyticalCurvilinearJacobian prop(inPar, xPerigee, pPerigee, s);
  AlgebraicMatrix66 pr;
  pr(5, 5) = 1;
  pr.Place_at(prop.jacobian(), 0, 0);

  //transportation
  AlgebraicSymMatrix66 cov2 = ROOT::Math::Similarity(pr, cov1);

  //now geting back to 7-parametrization from curvilinear
  cov = ROOT::Math::Similarity(cu2ca, cov2);

  // FreeTrajectoryState fts(fPar);

  //return parameters as a kiematic state
  KinematicParameters resPar(par);
  KinematicParametersError resEr(cov);
  return KinematicState(resPar, resEr, state.particleCharge(), state.magneticField());

  //return  KinematicState(fts,state.mass(), cov(6,6));
}
