#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/GetComponents.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"

#include <iostream>

bool
MultiTrajectoryStateMode::momentumFromModeCartesian (const TrajectoryStateOnSurface tsos,
						     GlobalVector& momentum) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  momentum = GlobalVector(0.,0.,0.);
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // 1D mode computation for px, py and pz
  // 
  GetComponents comps(tsos);
  auto const & components = comps();
  auto numb = components.size();
  // vectors of components in x, y and z
  std::vector<SingleGaussianState1D> pxStates; pxStates.reserve(numb);
  std::vector<SingleGaussianState1D> pyStates; pyStates.reserve(numb);
  std::vector<SingleGaussianState1D> pzStates; pzStates.reserve(numb);
  // iteration over components
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
	ic!=components.end(); ++ic ) {
    // extraction of parameters and variances
    GlobalVector mom(ic->globalMomentum());
    AlgebraicSymMatrix66 cov(ic->cartesianError().matrix());
    pxStates.push_back(SingleGaussianState1D(mom.x(),cov(3,3),ic->weight()));
    pyStates.push_back(SingleGaussianState1D(mom.y(),cov(4,4),ic->weight()));
    pzStates.push_back(SingleGaussianState1D(mom.z(),cov(5,5),ic->weight()));
  }
  //
  // transformation in 1D multi-states and creation of utility classes
  //
  MultiGaussianState1D pxState(pxStates);
  MultiGaussianState1D pyState(pyStates);
  MultiGaussianState1D pzState(pzStates);
  GaussianSumUtilities1D pxUtils(pxState);
  GaussianSumUtilities1D pyUtils(pyState);
  GaussianSumUtilities1D pzUtils(pzState);
  //
  // cartesian momentum vector from modes
  //
  momentum = GlobalVector(pxUtils.mode().mean(),pyUtils.mode().mean(),pzUtils.mode().mean());
  return true;
}

bool
MultiTrajectoryStateMode::positionFromModeCartesian (const TrajectoryStateOnSurface tsos,
						     GlobalPoint& position) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  position = GlobalPoint(0.,0.,0.);
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // 1D mode computation for x, y and z
  // 
  
  GetComponents comps(tsos);
  auto const & components = comps();
  auto numb = components.size();
  // vectors of components in x, y and z
  std::vector<SingleGaussianState1D> xStates; xStates.reserve(numb);
  std::vector<SingleGaussianState1D> yStates; yStates.reserve(numb);
  std::vector<SingleGaussianState1D> zStates; zStates.reserve(numb);
  // iteration over components
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
	ic!=components.end(); ++ic ) {
    // extraction of parameters and variances
    GlobalPoint pos(ic->globalPosition());
    AlgebraicSymMatrix66 cov(ic->cartesianError().matrix());
    xStates.push_back(SingleGaussianState1D(pos.x(),cov(0,0),ic->weight()));
    yStates.push_back(SingleGaussianState1D(pos.y(),cov(1,1),ic->weight()));
    zStates.push_back(SingleGaussianState1D(pos.z(),cov(2,2),ic->weight()));
  }
  //
  // transformation in 1D multi-states and creation of utility classes
  //
  MultiGaussianState1D xState(xStates);
  MultiGaussianState1D yState(yStates);
  MultiGaussianState1D zState(zStates);
  GaussianSumUtilities1D xUtils(xState);
  GaussianSumUtilities1D yUtils(yState);
  GaussianSumUtilities1D zUtils(zState);
  //
  // cartesian position vector from modes
  //
  position = GlobalPoint(xUtils.mode().mean(),yUtils.mode().mean(),zUtils.mode().mean());
  return true;
}

bool
MultiTrajectoryStateMode::momentumFromModeLocal (const TrajectoryStateOnSurface tsos,
						 GlobalVector& momentum) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  momentum = GlobalVector(0.,0.,0.);
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // mode computation for local co-ordinates q/p, dx/dz, dy/dz
  //
  double qpMode(0);
  double dxdzMode(0);
  double dydzMode(0);
  //
  // first 3 elements of local parameters = q/p, dx/dz, dy/dz
  //
  for ( unsigned int iv=0; iv<3; ++iv ) {
    // extraction of multi-state using helper class
    MultiGaussianState1D state1D = MultiGaussianStateTransform::multiState1D(tsos,iv);
    GaussianSumUtilities1D utils(state1D);
    // mode (in case of failure: mean)
    double result = utils.mode().mean();
    if ( !utils.modeIsValid() )  result = utils.mean();
    if ( iv==0 )  qpMode = result;
    else if ( iv==1 )  dxdzMode = result;
    else  dydzMode = result;
  }
  // local momentum vector from dx/dz, dy/dz and q/p + sign of local pz
  LocalVector localP(dxdzMode,dydzMode,1.);
  localP *= tsos.localParameters().pzSign()/fabs(qpMode)
    /sqrt(dxdzMode*dxdzMode+dydzMode*dydzMode+1.);
  // conversion to global coordinates
  momentum = tsos.surface().toGlobal(localP);
  return true;
}

bool
MultiTrajectoryStateMode::momentumFromModeQP (const TrajectoryStateOnSurface tsos,
					      double& momentum) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  momentum = 0.;
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // mode computation for local co-ordinates q/p, dx/dz, dy/dz
  //
  double qpMode(0);
  //
  // first element of local parameters = q/p
  //
  // extraction of multi-state using helper class
  MultiGaussianState1D state1D = MultiGaussianStateTransform::multiState1D(tsos,0);
  GaussianSumUtilities1D utils(state1D);
  // mode (in case of failure: mean)
  qpMode = utils.mode().mean();
  if ( !utils.modeIsValid() )  qpMode = utils.mean();

  momentum = 1./fabs(qpMode);
  return true;
}

bool
MultiTrajectoryStateMode::momentumFromModeP (const TrajectoryStateOnSurface tsos,
					     double& momentum) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  momentum = 0.;
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // first element of local parameters = q/p
  //
  // extraction of multi-state using helper class
  MultiGaussianState1D qpMultiState = MultiGaussianStateTransform::multiState1D(tsos,0);
  std::vector<SingleGaussianState1D> states(qpMultiState.components());
  // transform from q/p to p
  for ( unsigned int i=0; i<states.size(); ++i ) {
    SingleGaussianState1D& qpState = states[i];
    double wgt = qpState.weight();
    double qp = qpState.mean();
    double varQp = qpState.variance();
    double p = 1./fabs(qp);
    double varP = p*p*p*p*varQp;
    states[i] = SingleGaussianState1D(p,varP,wgt);
  }
  MultiGaussianState1D pMultiState(states);
  GaussianSumUtilities1D utils(pMultiState);
  // mode (in case of failure: mean)
  momentum = utils.mode().mean();
  if ( !utils.modeIsValid() )  momentum = utils.mean();

  return true;
}

bool
MultiTrajectoryStateMode::positionFromModeLocal (const TrajectoryStateOnSurface tsos,
						 GlobalPoint& position) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  position = GlobalPoint(0.,0.,0.);
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // mode computation for local co-ordinates x, y
  //
  double xMode(0);
  double yMode(0);
  //
  // last 2 elements of local parameters = x, y
  //
  for ( unsigned int iv=3; iv<5; ++iv ) {
    // extraction of multi-state using helper class
    MultiGaussianState1D state1D = MultiGaussianStateTransform::multiState1D(tsos,iv);
    GaussianSumUtilities1D utils(state1D);
    // mode (in case of failure: mean)
    double result = utils.mode().mean();
    if ( !utils.modeIsValid() )  result = utils.mean();
    if ( iv==3 )  xMode = result;
    else  yMode = result;
  }
  // local position vector from x, y
  LocalPoint localP(xMode,yMode,0.);
  // conversion to global coordinates
  position = tsos.surface().toGlobal(localP);
  return true;
}

bool
MultiTrajectoryStateMode::momentumFromModePPhiEta (const TrajectoryStateOnSurface tsos,
						   GlobalVector& momentum) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  momentum = GlobalVector(0.,0.,0.);
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // 1D mode computation for p, phi, eta
  // 
  GetComponents comps(tsos);
  auto const & components = comps();
  auto numb = components.size();
  // vectors of components in p, phi and eta
  std::vector<SingleGaussianState1D> pStates; pStates.reserve(numb);
  std::vector<SingleGaussianState1D> phiStates; phiStates.reserve(numb);
  std::vector<SingleGaussianState1D> etaStates; etaStates.reserve(numb);
  // covariances in cartesian and p-phi-eta and jacobian
  AlgebraicMatrix33 jacobian;
  AlgebraicSymMatrix33 covCart;
  AlgebraicSymMatrix33 covPPhiEta;
  // iteration over components
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
	ic!=components.end(); ++ic ) {
    // parameters
    GlobalVector mom(ic->globalMomentum());
    auto px = mom.x();
    auto py = mom.y();
    auto pz = mom.z();
    auto op = 1./mom.mag();
    auto opt2 = 1./mom.perp2();
    auto phi = mom.phi();
    auto eta = mom.eta();
    // jacobian
    jacobian(0,0) = px*op;
    jacobian(0,1) = py*op;
    jacobian(0,2) = pz*op;
    jacobian(1,0) = py*opt2;
    jacobian(1,1) = -px*opt2;
    jacobian(1,2) = 0;
    jacobian(2,0) = px*pz*opt2*op;
    jacobian(2,1) = py*pz*opt2*op;
    jacobian(2,2) = -op;
    // extraction of the momentum part from the 6x6 cartesian error matrix
    // and conversion to p-phi-eta
    covCart = ic->cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3);
    covPPhiEta = ROOT::Math::Similarity(jacobian,covCart);
    pStates.push_back(SingleGaussianState1D(1/op,covPPhiEta(0,0),ic->weight()));
    phiStates.push_back(SingleGaussianState1D(phi,covPPhiEta(1,1),ic->weight()));
    etaStates.push_back(SingleGaussianState1D(eta,covPPhiEta(2,2),ic->weight()));
  }
  //
  // transformation in 1D multi-states and creation of utility classes
  //
  MultiGaussianState1D pState(pStates);
  MultiGaussianState1D phiState(phiStates);
  MultiGaussianState1D etaState(etaStates);
  GaussianSumUtilities1D pUtils(pState);
  GaussianSumUtilities1D phiUtils(phiState);
  GaussianSumUtilities1D etaUtils(etaState);
  //
  // parameters from mode (in case of failure: mean)
  //
  auto p = pUtils.modeIsValid() ? pUtils.mode().mean() : pUtils.mean();
  auto phi = phiUtils.modeIsValid() ? phiUtils.mode().mean() : phiUtils.mean();
  auto eta = etaUtils.modeIsValid() ? etaUtils.mode().mean() : etaUtils.mean();
//   double theta = 2*atan(exp(-eta));
  auto tanth2 = std::exp(-eta);
  auto pt = p*2*tanth2/(1+tanth2*tanth2);  // p*sin(theta)
  auto pz = p*(1-tanth2*tanth2)/(1+tanth2*tanth2);  // p*cos(theta)
  // conversion to a cartesian momentum vector
  momentum = GlobalVector(pt*cos(phi),pt*sin(phi),pz);
  return true;
}

int
MultiTrajectoryStateMode::chargeFromMode (const TrajectoryStateOnSurface tsos) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return 0;
  }
  //  
  // mode computation for local co-ordinates q/p
  // extraction of multi-state using helper class
  MultiGaussianState1D state1D = MultiGaussianStateTransform::multiState1D(tsos,0);
  GaussianSumUtilities1D utils(state1D);
  // mode (in case of failure: mean)
  double result = utils.mode().mean();
  if ( !utils.modeIsValid() )  result = utils.mean();

  return result>0. ? 1 : -1;
}

