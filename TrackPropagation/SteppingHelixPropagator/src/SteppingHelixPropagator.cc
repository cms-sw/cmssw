/** \class SteppingHelixPropagator
 *  Propagator implementation using steps along a helix.
 *  Minimal geometry navigation.
 *  Material effects (multiple scattering and energy loss) are based on tuning
 *  to MC and (eventually) data. 
 *  Implementation file contents follow.
 *
 *  $Date: 2007/01/24 01:06:40 $
 *  $Revision: 1.22 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Fri Mar  3 16:01:24 CST 2006
// $Id: SteppingHelixPropagator.cc,v 1.22 2007/01/24 01:06:40 slava77 Exp $
//
//


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "Utilities/Timing/interface/TimingReport.h"

#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"
#include "Geometry/Surface/interface/Cone.h"

#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "CLHEP/Matrix/DiagMatrix.h"


SteppingHelixPropagator::SteppingHelixPropagator() :
  Propagator(anyDirection)
{
  field_ = 0;
}

SteppingHelixPropagator::SteppingHelixPropagator(const MagneticField* field, 
						 PropagationDirection dir):
  Propagator(dir),
  unit66_(6,1)
{
  field_ = field;
  covRot_ = HepMatrix(6,6,0);
  dCTransform_ = unit66_;
  debug_ = false;
  noMaterialMode_ = false;
  noErrorPropagation_ = false;
  applyRadX0Correction_ = false;
  useMagVolumes_ = false;
  for (int i = 0; i <= MAX_POINTS; i++){
    svBuf_[i].cov = HepSymMatrix(6,0);
    svBuf_[i].matDCov = HepSymMatrix(6,0);
    svBuf_[i].isComplete = true;
    svBuf_[i].isValidInfo = true;
  }
}

TrajectoryStateOnSurface 
SteppingHelixPropagator::propagate(const FreeTrajectoryState& ftsStart, const Plane& pDest) const {
  return propagateWithPath(ftsStart, pDest).first;
}

TrajectoryStateOnSurface 
SteppingHelixPropagator::propagate(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const
{
  return propagateWithPath(ftsStart, cDest).first;
}

FreeTrajectoryState
SteppingHelixPropagator::propagate(const FreeTrajectoryState& ftsStart, const GlobalPoint& pDest) const
{
  return propagateWithPath(ftsStart, pDest).first;
}

FreeTrajectoryState
SteppingHelixPropagator::propagate(const FreeTrajectoryState& ftsStart, 
				   const GlobalPoint& pDest1, const GlobalPoint& pDest2) const
{
  return propagateWithPath(ftsStart, pDest1, pDest2).first;
}


std::pair<TrajectoryStateOnSurface, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const Plane& pDest) const {

  setIState(SteppingHelixStateInfo(ftsStart));

  const StateInfo& svCurrent = propagate(svBuf_[0], pDest);

  return TsosPP(svCurrent.getStateOnSurface(pDest), svCurrent.path);
}

std::pair<TrajectoryStateOnSurface, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const Cylinder& cDest) const {

  setIState(SteppingHelixStateInfo(ftsStart));

  const StateInfo& svCurrent = propagate(svBuf_[0], cDest);

  return TsosPP(svCurrent.getStateOnSurface(cDest), svCurrent.path);
}


std::pair<FreeTrajectoryState, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const GlobalPoint& pDest) const {
  setIState(SteppingHelixStateInfo(ftsStart));

  const StateInfo& svCurrent = propagate(svBuf_[0], pDest);

  FreeTrajectoryState ftsDest;
  svCurrent.getFreeState(ftsDest);

  return FtsPP(ftsDest, svCurrent.path);
}

std::pair<FreeTrajectoryState, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const GlobalPoint& pDest1, const GlobalPoint& pDest2) const {

  if ((pDest1-pDest2).mag() < 1e-10) return FtsPP();
  setIState(SteppingHelixStateInfo(ftsStart));
  
  const StateInfo& svCurrent = propagate(svBuf_[0], pDest1, pDest2);

  FreeTrajectoryState ftsDest;
  svCurrent.getFreeState(ftsDest);

  return FtsPP(ftsDest, svCurrent.path);
}


const SteppingHelixStateInfo&
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const Surface& sDest) const {
  
  if (! sStart.isValid()) return invalidState_;

  const Plane* pDest = dynamic_cast<const Plane*>(&sDest);
  if (pDest != 0) return propagate(sStart, *pDest);

  const Cylinder* cDest = dynamic_cast<const Cylinder*>(&sDest);
  if (cDest != 0) return propagate(sStart, *cDest);

  throw PropagationException("The surface is neither Cylinder nor Plane");

}

const SteppingHelixStateInfo&
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const Plane& pDest) const {
  
  if (! sStart.isValid()) return invalidState_;
  setIState(sStart);
  
  GlobalPoint rPlane = pDest.toGlobal(LocalPoint(0,0,0));
  GlobalVector nPlane = pDest.toGlobal(LocalVector(0,0,1.)); nPlane = nPlane.unit();
  double pars[6] = { rPlane.x(), rPlane.y(), rPlane.z(),
		     nPlane.x(), nPlane.y(), nPlane.z() };
  
  propagate(PLANE_DT, pars);
  
  return svBuf_[cIndex_(nPoints_-1)];
}

const SteppingHelixStateInfo&
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const Cylinder& cDest) const {
  
  if (! sStart.isValid()) return invalidState_;
  setIState(sStart);
  
  double pars[6];
  pars[RADIUS_P] = cDest.radius();

  
  propagate(RADIUS_DT, pars);
  
  return svBuf_[cIndex_(nPoints_-1)];
}

const SteppingHelixStateInfo&
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const GlobalPoint& pDest) const {
  
  if (! sStart.isValid()) return invalidState_;
  setIState(sStart);
  
  double pars[6] = {pDest.x(), pDest.y(), pDest.z(), 0, 0, 0};

  
  propagate(POINT_PCA_DT, pars);
  
  return svBuf_[cIndex_(nPoints_-1)];
}

const SteppingHelixStateInfo&
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const GlobalPoint& pDest1, const GlobalPoint& pDest2) const {
  
  if ((pDest1-pDest2).mag() < 1e-10 || !sStart.isValid()) return invalidState_;
  setIState(sStart);
  
  double pars[6] = {pDest1.x(), pDest1.y(), pDest1.z(),
		    pDest2.x(), pDest2.y(), pDest2.z()};
  
  propagate(LINE_PCA_DT, pars);
  
  return svBuf_[cIndex_(nPoints_-1)];
}


void SteppingHelixPropagator::setIState(const FreeTrajectoryState& ftsStart) const {
  //need to get rid of these conversions .. later
  GlobalVector p3GV = ftsStart.momentum();
  GlobalPoint r3GP = ftsStart.position();
  Vector p3(p3GV.x(), p3GV.y(), p3GV.z());
  Point  r3(r3GP.x(), r3GP.y(), r3GP.z());
  
  int charge = ftsStart.charge();
  
  setIState(p3, r3, charge,  
	    (ftsStart.hasError() && !noErrorPropagation_) 
	    ? ftsStart.cartesianError().matrix() : HepSymMatrix(1,0),
	    propagationDirection());
  
}

void SteppingHelixPropagator::setIState(const SteppingHelixStateInfo& sStart) const {
  nPoints_ = 0;
  if (sStart.isComplete ) {
    svBuf_[cIndex_(nPoints_)] = sStart;
    nPoints_++;
  } else {
    setIState(sStart.p3, sStart.r3, sStart.q, 
	      !noErrorPropagation_ ? sStart.cov : HepSymMatrix(1,0),
	      propagationDirection());
  }
}

void SteppingHelixPropagator::setIState(const SteppingHelixPropagator::Vector& p3, 
					const SteppingHelixPropagator::Point& r3, int charge, 
					const HepSymMatrix& cov, PropagationDirection dir) const {
  nPoints_ = 0;
  loadState(svBuf_[cIndex_(nPoints_)], p3, r3, charge, cov, dir);
  nPoints_++;
}


void SteppingHelixPropagator::getFState(FreeTrajectoryState& ftsDest) const{
  Vector p3F;
  Point  r3F;
  HepSymMatrix covF;

  getFState(p3F, r3F, covF); 
  GlobalVector p3FGV(p3F.x(), p3F.y(), p3F.z());
  GlobalPoint r3FGP(r3F.x(), r3F.y(), r3F.z());
  GlobalTrajectoryParameters tParsDest(r3FGP, p3FGV, svBuf_[cIndex_(nPoints_-1)].q, field_);
  CartesianTrajectoryError tCovDest(covF);

  ftsDest = (covF.num_row() >=5  && !noErrorPropagation_) 
    ? FreeTrajectoryState(tParsDest, tCovDest) 
    : FreeTrajectoryState(tParsDest);
  if (ftsDest.hasError()) ftsDest.curvilinearError(); //call it so it gets created
}

void SteppingHelixPropagator::getFState(SteppingHelixPropagator::Vector& p3, 
					SteppingHelixPropagator::Point& r3, 
					HepSymMatrix& cov) const{
  const StateInfo& svCurrent = svBuf_[cIndex_(nPoints_-1)];
  p3 = svCurrent.p3;
  r3 = svCurrent.r3;
  //update Emat only if it's valid
  if (svCurrent.cov.num_row() >=5){
    cov = svCurrent.cov;
  } else {
    cov = svCurrent.cov;
  }

}


SteppingHelixPropagator::Result 
SteppingHelixPropagator::propagate(SteppingHelixPropagator::DestType type, 
				   const double pars[6], double epsilon)  const{

  StateInfo* svCurrent = &svBuf_[cIndex_(nPoints_-1)];

  //check if it's going to work at all
  double tanDist = 0;
  double dist = 0;
  PropagationDirection refDirection = anyDirection;
  Result result = refToDest(type, (*svCurrent), pars, dist, tanDist, refDirection);

  if (result != SteppingHelixStateInfo::OK ){
    svCurrent->status_ = result;
    svCurrent->isValidInfo = result == SteppingHelixStateInfo::OK;
    svCurrent->field = field_;
    return result;
  }

  result = SteppingHelixStateInfo::UNDEFINED;
  bool makeNextStep = true;
  double dStep = 1.;
  PropagationDirection dir,oldDir;
  dir = propagationDirection(); 
  oldDir = dir;
  int nOsc = 0;

  double distMag = 1e12;
  double tanDistMag = 1e12;
  
  while (makeNextStep){
    dStep = 1.;
    svCurrent = &svBuf_[cIndex_(nPoints_-1)];
    double curZ = svCurrent->r3.z();
    double curR = svCurrent->r3.perp();
    refDirection = propagationDirection();
    refToDest(type, (*svCurrent), pars, dist, tanDist, refDirection);

    if (propagationDirection() == anyDirection){
      dir = refDirection;
    } else {
      dir = propagationDirection();
    }
    if (useMagVolumes_){//need to know the general direction
      refToMagVolume((*svCurrent), dir, distMag, tanDistMag);
    }

    double rDotP = svCurrent->r3.dot(svCurrent->p3);
    if ((fabs(curZ) > 1.5e3 || curR >800.) 
	&& ((dir == alongMomentum && rDotP > 0) 
	    || (dir == oppositeToMomentum && rDotP < 0) )
	){
      dStep = fabs(tanDist) -1e-12;
    }
    if (fabs(tanDist) < dStep){
      dStep = fabs(tanDist); 
      if (type == POINT_PCA_DT){
	//being lazy here; the best is to take into account the curvature
	dStep = fabs(tanDist)*0.5; 
      }
    }
    if (dStep > 1e-10){
      StateInfo* svNext = &svBuf_[cIndex_(nPoints_)];
      makeAtomStep((*svCurrent), (*svNext), dStep, dir, HEL_AS_F);
      nPoints_++;    svCurrent = &svBuf_[cIndex_(nPoints_-1)];
    }
    if (oldDir != dir) nOsc++;
    oldDir = dir;

    if (fabs(dist) < fabs(epsilon)  ) result = SteppingHelixStateInfo::OK;

    if ((type == POINT_PCA_DT || type == LINE_PCA_DT )
	&& fabs(dStep) < fabs(epsilon)  ){
      //now check if it's not a branch point (peek ahead at 1 cm)
      double nextDist = 0;
      double nextTanDist = 0;
      PropagationDirection nextRefDirection = anyDirection;
      StateInfo* svNext = &svBuf_[cIndex_(nPoints_)];
      makeAtomStep((*svCurrent), (*svNext), 1., dir, HEL_AS_F);
      nPoints_++;     svCurrent = &svBuf_[cIndex_(nPoints_-1)];
      refToDest(type, (*svCurrent), pars, nextDist, nextTanDist, nextRefDirection);
      if ( fabs(nextDist) > fabs(dist)){
	nPoints_--;      svCurrent = &svBuf_[cIndex_(nPoints_-1)];
	result = SteppingHelixStateInfo::OK;
	if (debug_){
	  std::cout<<"Found real local minimum in PCA"<<std::endl;
	}
      } else {
	//keep this trial point and continue
	dStep = 1.;
	if (debug_){
	  std::cout<<"Found branch point in PCA"<<std::endl;
	}
      }
    }

    if (nPoints_ > MAX_STEPS || nOsc > 6) result = SteppingHelixStateInfo::FAULT;

    if (svCurrent->p3.mag() < 0.1 ) result = SteppingHelixStateInfo::RANGEOUT;

    if ( curR > 20000 || fabs(curZ) > 20000 ) result = SteppingHelixStateInfo::INACC;

    makeNextStep = result == SteppingHelixStateInfo::UNDEFINED;
    svCurrent->status_ = result;
    svCurrent->isValidInfo = result == SteppingHelixStateInfo::OK;
    svCurrent->field = field_;
  }

  if (debug_){
    switch (type) {
    case RADIUS_DT:
      std::cout<<"going to radius "<<pars[RADIUS_P]<<std::endl;
      break;
    case Z_DT:
      std::cout<<"going to z "<<pars[Z_P]<<std::endl;
      break;
    case PATHL_DT:
      std::cout<<"going to pathL "<<pars[PATHL_P]<<std::endl;
      break;
    case PLANE_DT:
      {
	Point rPlane(pars[0], pars[1], pars[2]);
	Vector nPlane(pars[3], pars[4], pars[5]);
	std::cout<<"going to plane r0:"<<rPlane<<" n:"<<nPlane<<std::endl;
      }
      break;
    case POINT_PCA_DT:
      {
	Point rDest(pars[0], pars[1], pars[2]);
	std::cout<<"going to PCA to point "<<rDest<<std::endl;
      }
      break;
    case LINE_PCA_DT:
      {
	Point rDest1(pars[0], pars[1], pars[2]);
	Point rDest2(pars[3], pars[4], pars[5]);
	std::cout<<"going to PCA to line "<<rDest1<<" - "<<rDest2<<std::endl;
      }
      break;
    default:
      std::cout<<"going to NOT IMPLEMENTED"<<std::endl;
      break;
    }
    std::cout<<"Made "<<nPoints_-1<<" steps and stopped at(cur step) "<<svCurrent->r3<<std::endl;
  }
  
  return result;
}
  
void SteppingHelixPropagator::loadState(SteppingHelixPropagator::StateInfo& svCurrent, 
					const SteppingHelixPropagator::Vector& p3, 
					const SteppingHelixPropagator::Point& r3, int charge,
					const HepSymMatrix& cov, PropagationDirection dir) const{
  svCurrent.q = charge;
  svCurrent.p3 = p3;
  svCurrent.r3 = r3;
  svCurrent.dir = dir == alongMomentum ? 1.: -1.;

  svCurrent.path = 0; // this could've held the initial path
  svCurrent.radPath = 0;

  GlobalPoint gPoint(r3.x(), r3.y(), r3.z());
  GlobalVector bf = field_->inTesla(gPoint);
  if (useMagVolumes_){
    GlobalPoint gPointNegZ(svCurrent.r3.x(), svCurrent.r3.y(), svCurrent.r3.z() > 0. ? -svCurrent.r3.z() : svCurrent.r3.z());
    const VolumeBasedMagneticField* vbField = dynamic_cast<const VolumeBasedMagneticField*>(field_);
    if (vbField ){
      svCurrent.magVol = vbField->findVolume(gPointNegZ);
    } else {
      std::cout<<"Failed to cast into VolumeBasedMagneticField"<<std::endl;
    }
    if (debug_){
      std::cout<<"Got volume at "<<svCurrent.magVol<<std::endl;
    }
  }
  
  svCurrent.bf.set(bf.x(), bf.y(), bf.z());
  if (svCurrent.bf.mag() < 1e-6) svCurrent.bf.set(0., 0., 1e-6);

  setRep(svCurrent);

  svCurrent.cov.assign(cov);

  svCurrent.isComplete = true;

  if (debug_){
    std::cout<<"Loaded at  path: "<<svCurrent.path<<" radPath: "<<svCurrent.radPath
	     <<" p3 "<<" pt: "<<svCurrent.p3.perp()<<" phi: "<<svCurrent.p3.phi()
	     <<" eta: "<<svCurrent.p3.eta()
	     <<" "<<svCurrent.p3
	     <<" r3: "<<svCurrent.r3
	     <<" bField: "<<svCurrent.bf.mag()
	     <<std::endl;
    std::cout<<"Input Covariance in Global RF "<<cov<<std::endl;
  }
}

void SteppingHelixPropagator::getNextState(const SteppingHelixPropagator::StateInfo& svPrevious, 
					   SteppingHelixPropagator::StateInfo& svNext,
					   double dP, SteppingHelixPropagator::Vector tau,
					   double dX, double dY, double dZ, double dS, double dX0,
					   const HepMatrix& dCovTransform) const{
  svNext.q = svPrevious.q;
  svNext.dir = dS > 0.0 ? 1.: -1.; 
  svNext.p3 = tau;  svNext.p3*=(svPrevious.p3.mag() - svNext.dir*fabs(dP));

  svNext.r3 = svPrevious.r3;
  Vector tmpR3 = svPrevious.rep.lX; tmpR3*=dX;
  svNext.r3+= tmpR3;
  tmpR3 = svPrevious.rep.lY; tmpR3*=dY;
  svNext.r3+= tmpR3;
  tmpR3 = svPrevious.rep.lZ; tmpR3*=dZ;
  svNext.r3+= tmpR3;
  svNext.path = svPrevious.path + dS;
  svNext.radPath = svPrevious.radPath + dX0;


  GlobalPoint gPoint(svNext.r3.x(), svNext.r3.y(), svNext.r3.z());

  GlobalVector bf = field_->inTesla(gPoint);
  svNext.bf.set(bf.x(), bf.y(), bf.z());
  if (svNext.bf.mag() < 1e-6) svNext.bf.set(0., 0., 1e-6);
  if (useMagVolumes_){
    GlobalPoint gPointNegZ(svNext.r3.x(), svNext.r3.y(), svNext.r3.z() > 0. ? -svNext.r3.z() : svNext.r3.z());
    const VolumeBasedMagneticField* vbField = dynamic_cast<const VolumeBasedMagneticField*>(field_);
    if (vbField ){
      svNext.magVol = vbField->findVolume(gPointNegZ);
    } else {
      std::cout<<"Failed to cast into VolumeBasedMagneticField"<<std::endl;
    }
  }
  
  setRep(svNext);
  //  getLocBGrad(ind, 1e-1);
  
  
  //update Emat only if it's valid
  if (svPrevious.cov.num_row() >=5){
    svNext.cov = svPrevious.cov.similarity(dCovTransform);
    svNext.cov += svPrevious.matDCov;
  } else {
    svNext.cov.assign(svPrevious.cov);
  }

  if (debug_){
    std::cout<<"Now at  path: "<<svNext.path<<" radPath: "<<svNext.radPath
	     <<" p3 "<<" pt: "<<svNext.p3.perp()<<" phi: "<<svNext.p3.phi()
	     <<" eta: "<<svNext.p3.eta()
	     <<" "<<svNext.p3
	     <<" r3: "<<svNext.r3
	     <<" dPhi: "<<acos(svNext.p3.unit().dot(svPrevious.p3.unit()))
	     <<" bField: "<<svNext.bf.mag()
	     <<std::endl;
    std::cout<<"New Covariance "<<svNext.cov<<std::endl;
    std::cout<<"Transf by dCovTransform "<<dCovTransform<<std::endl;
  }
}

void SteppingHelixPropagator::setRep(SteppingHelixPropagator::StateInfo& sv) const{
  Vector tau = sv.p3/(sv.p3.mag());
  setRep(sv.rep, tau);
}

void SteppingHelixPropagator::setRep(SteppingHelixPropagator::Basis& rep, 
				     const SteppingHelixPropagator::Vector& tau) const{
  Vector zRep(0., 0., 1.);
  rep.lX = tau;
  rep.lY = zRep.cross(tau); rep.lY /= tau.perp();
  rep.lZ = rep.lX.cross(rep.lY);
}

bool SteppingHelixPropagator::makeAtomStep(SteppingHelixPropagator::StateInfo& svCurrent,
					   SteppingHelixPropagator::StateInfo& svNext,
					   double dS, 
					   PropagationDirection dir, 
					   SteppingHelixPropagator::Fancy fancy) const{
  if (debug_){
    std::cout<<"Make atom step "<<svCurrent.path<<" with step "<<dS<<" in direction "<<dir<<std::endl;
  }

  double dP = 0;
  Vector tau = svCurrent.p3; tau/=tau.mag();
  Vector tauNext(tau);

  dS = dir == alongMomentum ? fabs(dS) : -fabs(dS);

  double p0 = svCurrent.p3.mag();
  double b0 = svCurrent.bf.mag();
  double phi = 0.0029979*svCurrent.q*b0/p0*dS;
  bool phiSmall = fabs(phi) < 3e-8;

  double cosPhi = cos(phi);
  double oneLessCosPhi = phiSmall ? 0.5*phi*phi*(1.- phi*phi/12.)  : 1.-cosPhi;
  double oneLessCosPhiOPhi = phiSmall ? 0.5*phi*(1.- phi*phi/12.)  : oneLessCosPhi/phi;
  double sinPhi = sin(phi);
  double sinPhiOPhi = phiSmall ? 1. - phi*phi/6. : sinPhi/phi;
  double phiLessSinPhiOPhi = phiSmall ? phi*phi/6.*(1. - phi*phi/20.) : (phi - sinPhi)/phi;

  Vector bHat = svCurrent.bf; bHat /= bHat.mag();
  double bxC = svCurrent.rep.lX.dot(bHat);
  double by = svCurrent.rep.lY.dot(bHat);
  double bz = svCurrent.rep.lZ.dot(bHat);
  double oneLessBxC2 = (1.-bxC*bxC);

  //components in local rf
  double dX =0.;
  double dY =0.;
  double dZ =0.;
  double tauX =0.;
  double tauY =0.;
  double tauZ =0.;
  
  double dEdXPrime = 0;
  double radX0 = 1e24;
  double dEdx = getDeDx(svCurrent, dEdXPrime, radX0);
  double theta02 = 14.e-3/p0*sqrt(fabs(dS)/radX0); // .. drop log term (this is non-additive)
  theta02 *=theta02;
  if (applyRadX0Correction_){
    // this provides the integrand for theta^2
    // if summed up along the path, should result in 
    // theta_total^2 = Int_0^x0{ f(x)dX} = (13.6/p0)^2*x0*(1+0.036*ln(x0+1))
    // x0+1 above is to make the result infrared safe.
    double x0 = fabs(svCurrent.radPath);
    double dX0 = fabs(dS)/radX0;
    double alphaX0 = 13.6e-3/p0; alphaX0 *= alphaX0;
    double betaX0 = 0.038;
    theta02 = dX0*alphaX0*(1+betaX0*log(x0+1))*(1 + betaX0*log(x0+1) + 2.*betaX0*x0/(x0+1) );
  }

  double epsilonP0 = 0;
  double omegaP0 = 0;

  switch (fancy){
  case HEL_AS_F:
  case HEL_ALL_F:
    dP = dEdx*dS;
    tauX = (1.0 - oneLessCosPhi*oneLessBxC2);
    tauY = (oneLessCosPhi*bxC*by - sinPhi*bz);
    tauZ = (oneLessCosPhi*bxC*bz + sinPhi*by);

    tauNext = svCurrent.rep.lX*tauX 
      + svCurrent.rep.lY*tauY + svCurrent.rep.lZ*tauZ;

    dX = dS*(1. - phiLessSinPhiOPhi*oneLessBxC2);
    dY = dS*(bxC*by*phiLessSinPhiOPhi - oneLessCosPhiOPhi*bz);
    dZ = dS*(bxC*bz*phiLessSinPhiOPhi + oneLessCosPhiOPhi*by);

    if (svCurrent.cov.num_row() >=5){
      epsilonP0 = 1.+ dP/p0;
      omegaP0 = -dP/p0 + dS*dEdXPrime;      
      
      Vector btVec(bHat.cross(tau));
      Vector bbtVec(bHat.cross(btVec));
      Vector tbtVec(tau.cross(btVec));

      double dsp = dS/p0;

      dCTransform_ = unit66_;
      //make everything in global
      //case I: no "spatial" derivatives |--> dCtr({1,2,3,4,5,6}{1,2,3}) = 0    
      dCTransform_(1,4) = dsp*(bHat.x()*tbtVec.x() 
			       + cosPhi*tau.x()*bbtVec.x()
			       + ((1.-bHat.x()*bHat.x()) + phi*tau.x()*btVec.x())*sinPhiOPhi);

      dCTransform_(1,5) = dsp*(bHat.z()*oneLessCosPhiOPhi + bHat.x()*tbtVec.y()
			       + cosPhi*tau.y()*bbtVec.x() 
			       + (-bHat.x()*bHat.y() + phi*tau.y()*btVec.x())*sinPhiOPhi);
      dCTransform_(1,6) = dsp*(-bHat.y()*oneLessCosPhiOPhi + bHat.x()*tbtVec.z()
			       + cosPhi*tau.z()*bbtVec.x()
			       + (-bHat.x()*bHat.z() + phi*tau.z()*btVec.x())*sinPhiOPhi);

      dCTransform_(2,4) = dsp*(-bHat.z()*oneLessCosPhiOPhi + bHat.y()*tbtVec.x()
			       + cosPhi*tau.x()*bbtVec.y()
			       + (-bHat.x()*bHat.y() + phi*tau.x()*btVec.y())*sinPhiOPhi);
      dCTransform_(2,5) = dsp*(bHat.y()*tbtVec.y() 
			       + cosPhi*tau.y()*bbtVec.y()
			       + ((1.-bHat.y()*bHat.y()) + phi*tau.y()*btVec.y())*sinPhiOPhi);
      dCTransform_(2,6) = dsp*(bHat.x()*oneLessCosPhiOPhi + bHat.y()*tbtVec.z()
			       + cosPhi*tau.z()*bbtVec.y() 
			       + (-bHat.y()*bHat.z() + phi*tau.z()*btVec.y())*sinPhiOPhi);

      dCTransform_(3,4) = dsp*(bHat.y()*oneLessCosPhiOPhi + bHat.z()*tbtVec.x()
			       + cosPhi*tau.x()*bbtVec.z() 
			       + (-bHat.x()*bHat.z() + phi*tau.x()*btVec.z())*sinPhiOPhi);
      dCTransform_(3,5) = dsp*(-bHat.x()*oneLessCosPhiOPhi + bHat.z()*tbtVec.y()
			       + cosPhi*tau.y()*bbtVec.z()
			       + (-bHat.y()*bHat.z() + phi*tau.y()*btVec.z())*sinPhiOPhi);
      dCTransform_(3,6) = dsp*(bHat.z()*tbtVec.z() 
			       + cosPhi*tau.z()*bbtVec.z()
			       + ((1.-bHat.z()*bHat.z()) + phi*tau.z()*btVec.z())*sinPhiOPhi);


      dCTransform_(4,4) = epsilonP0*(1. - oneLessCosPhi*(1.-bHat.x()*bHat.x())
				     + phi*tau.x()*(cosPhi*btVec.x() - sinPhi*bbtVec.x()))
	+ omegaP0*tau.x()*tauNext.x();
      dCTransform_(4,5) = epsilonP0*(bHat.x()*bHat.y()*oneLessCosPhi + bHat.z()*sinPhi
				     + phi*tau.y()*(cosPhi*btVec.x() - sinPhi*bbtVec.x()))
	+ omegaP0*tau.y()*tauNext.x();
      dCTransform_(4,6) = epsilonP0*(bHat.x()*bHat.z()*oneLessCosPhi - bHat.y()*sinPhi
				     + phi*tau.z()*(cosPhi*btVec.x() - sinPhi*bbtVec.x()))
	+ omegaP0*tau.z()*tauNext.x();

      dCTransform_(5,4) = epsilonP0*(bHat.x()*bHat.y()*oneLessCosPhi - bHat.z()*sinPhi
				     + phi*tau.x()*(cosPhi*btVec.y() - sinPhi*bbtVec.y()))
	+ omegaP0*tau.x()*tauNext.y();
      dCTransform_(5,5) = epsilonP0*(1. - oneLessCosPhi*(1.-bHat.y()*bHat.y())
				     + phi*tau.y()*(cosPhi*btVec.y() - sinPhi*bbtVec.y()))
	+ omegaP0*tau.y()*tauNext.y();
      dCTransform_(5,6) = epsilonP0*(bHat.y()*bHat.z()*oneLessCosPhi + bHat.x()*sinPhi
				     + phi*tau.z()*(cosPhi*btVec.y() - sinPhi*bbtVec.y()))
	+ omegaP0*tau.z()*tauNext.y();
    
      dCTransform_(6,4) = epsilonP0*(bHat.x()*bHat.z()*oneLessCosPhi + bHat.y()*sinPhi
				     + phi*tau.x()*(cosPhi*btVec.z() - sinPhi*bbtVec.z()))
	+ omegaP0*tau.x()*tauNext.z();
      dCTransform_(6,5) = epsilonP0*(bHat.y()*bHat.z()*oneLessCosPhi - bHat.x()*sinPhi
				     + phi*tau.y()*(cosPhi*btVec.z() - sinPhi*bbtVec.z()))
	+ omegaP0*tau.y()*tauNext.z();
      dCTransform_(6,6) = epsilonP0*(1. - oneLessCosPhi*(1.-bHat.z()*bHat.z())
				     + phi*tau.z()*(cosPhi*btVec.z() - sinPhi*bbtVec.z()))
	+ omegaP0*tau.z()*tauNext.z();
    

      Basis rep; setRep(rep, tauNext);
      //mind the sign of dS and dP (dS*dP < 0 allways)
      //covariance should grow no matter which direction you propagate
      //==> take abs values.
      svCurrent.matDCov *= 0.;
      double mulRR = theta02*dS*dS/3.;
      double mulRP = theta02*fabs(dS)*p0/2.;
      double mulPP = theta02*p0*p0;
      double losPP = dP*dP*1.6/fabs(dS)*(1.0 + p0*1e-3);
      //another guess .. makes sense for 1 cm steps 2./dS == 2 [cm] / dS [cm] at low pt
      //double it by 1TeV
      //not gaussian anyways
      // derived from the fact that sigma_p/eLoss ~ 0.08 after ~ 200 steps

      //symmetric RR part
      svCurrent.matDCov(1,1) = mulRR*(rep.lY.x()*rep.lY.x() + rep.lZ.x()*rep.lZ.x());
      svCurrent.matDCov(1,2) = mulRR*(rep.lY.x()*rep.lY.y() + rep.lZ.x()*rep.lZ.y());
      svCurrent.matDCov(1,3) = mulRR*(rep.lY.x()*rep.lY.z() + rep.lZ.x()*rep.lZ.z());
      svCurrent.matDCov(2,2) = mulRR*(rep.lY.y()*rep.lY.y() + rep.lZ.y()*rep.lZ.y());
      svCurrent.matDCov(2,3) = mulRR*(rep.lY.y()*rep.lY.z() + rep.lZ.y()*rep.lZ.z());
      svCurrent.matDCov(3,3) = mulRR*(rep.lY.z()*rep.lY.z() + rep.lZ.z()*rep.lZ.z());

      //symmetric PP part
      svCurrent.matDCov(4,4) = mulPP*(rep.lY.x()*rep.lY.x() + rep.lZ.x()*rep.lZ.x())
	+ losPP*rep.lX.x()*rep.lX.x();
      svCurrent.matDCov(4,5) = mulPP*(rep.lY.x()*rep.lY.y() + rep.lZ.x()*rep.lZ.y())
	+ losPP*rep.lX.x()*rep.lX.y();
      svCurrent.matDCov(4,6) = mulPP*(rep.lY.x()*rep.lY.z() + rep.lZ.x()*rep.lZ.z())
	+ losPP*rep.lX.x()*rep.lX.z();
      svCurrent.matDCov(5,5) = mulPP*(rep.lY.y()*rep.lY.y() + rep.lZ.y()*rep.lZ.y())
	+ losPP*rep.lX.y()*rep.lX.y();
      svCurrent.matDCov(5,6) = mulPP*(rep.lY.y()*rep.lY.z() + rep.lZ.y()*rep.lZ.z())
	+ losPP*rep.lX.y()*rep.lX.z();
      svCurrent.matDCov(6,6) = mulPP*(rep.lY.z()*rep.lY.z() + rep.lZ.z()*rep.lZ.z())
	+ losPP*rep.lX.z()*rep.lX.z();

      //still symmetric but fill 9 elements
      double mXX = mulRP*(rep.lY.x()*rep.lY.x() + rep.lZ.x()*rep.lZ.x());
      double mXY = mulRP*(rep.lY.x()*rep.lY.y() + rep.lZ.x()*rep.lZ.y());
      double mXZ = mulRP*(rep.lY.x()*rep.lY.z() + rep.lZ.x()*rep.lZ.z());
      double mYY = mulRP*(rep.lY.y()*rep.lY.y() + rep.lZ.y()*rep.lZ.y());
      double mYZ = mulRP*(rep.lY.y()*rep.lY.z() + rep.lZ.y()*rep.lZ.z());
      double mZZ = mulRP*(rep.lY.z()*rep.lY.z() + rep.lZ.z()*rep.lZ.z());
      svCurrent.matDCov(1,4) = mXX;
      svCurrent.matDCov(1,5) = mXY;
      svCurrent.matDCov(1,6) = mXZ;
      svCurrent.matDCov(2,4) = mXY;
      svCurrent.matDCov(2,5) = mYY;
      svCurrent.matDCov(2,6) = mYZ;
      svCurrent.matDCov(3,4) = mXZ;
      svCurrent.matDCov(3,5) = mYZ;
      svCurrent.matDCov(3,6) = mZZ;
      
    }

    break;
//   case POL_1_F:
//   case POL_2_F:
//   case POL_M_F:
//     break;
  default:
    break;
  }

  if (dir == oppositeToMomentum) dP = -fabs(dP);
  dP = dP > p0 ? p0-1e-5 : dP;
  getNextState(svCurrent, svNext, dP, tauNext, dX, dY, dZ, dS, dS/radX0,
		 dCTransform_);
  return true;
}

double SteppingHelixPropagator::getDeDx(const SteppingHelixPropagator::StateInfo& sv, 
					double& dEdXPrime, double& radX0) const{
  radX0 = 1.e24;
  dEdXPrime = 0.;
  if (noMaterialMode_) return 0;

  double dEdx = 0.;

  double lR = sv.r3.perp();
  double lZ = fabs(sv.r3.z());
  double lEtaDet = sv.r3.eta();

  //assume "Iron" .. seems to be quite the same for brass/iron/PbW04
  //good for Fe within 3% for 0.2 GeV to 10PeV
  double p0 = sv.p3.mag();

  //0.065 (PDG) --> 0.044 to better match with MPV
  double dEdX_mat = -(11.4 + 0.96*fabs(log(p0*2.8)) + 0.033*p0*(1.0 - pow(p0, -0.33)) )*1e-3; 
  //in GeV/cm .. 0.8 to get closer to the median or MPV
  double dEdX_HCal = 0.95*dEdX_mat; //extracted from sim
  double dEdX_ECal = 0.45*dEdX_mat;
  double dEdX_coil = 0.35*dEdX_mat; //extracted from sim .. closer to 40% in fact
  double dEdX_Fe =   dEdX_mat;
  double dEdX_MCh =  0.053*dEdX_mat; //chambers on average
  double dEdX_Trk =  0.0114*dEdX_mat;
  double dEdX_Vac =  0.0;

  double radX0_HCal = 1.44/0.8; //guessing
  double radX0_ECal = 0.89/0.7;
  double radX0_coil = 4.; //
  double radX0_Fe =   1.76;
  double radX0_MCh =  1e3; //
  double radX0_Trk =  320.;
  double radX0_Air =  3.e4;
  double radX0_Vac =  3.e9; //"big" number for vacuum


  //this should roughly figure out where things are 
  //(numbers taken from Fig1.1.2 TDR and from geom xmls)
  if (lR < 2.9){ //inside beampipe
    dEdx = dEdX_Vac; radX0 = radX0_Vac;
  }
  else if (lR < 129){
    if (lZ < 294){ 
      dEdx = dEdx = dEdX_Trk; radX0 = radX0_Trk; 
      //somewhat empirical formula that ~ matches the average if going from 0,0,0
      //assuming "uniform" tracker material
      //doesn't really track material layer to layer
      double scaleRadX = lEtaDet > 1.5 ? 0.7724 : sin(2.*atan(exp(-0.5*lEtaDet)));
      scaleRadX *= scaleRadX;
      if (lEtaDet > 2 && lZ > 20) scaleRadX *= (lEtaDet-1.);
      if (lEtaDet > 2.5 && lZ > 20) scaleRadX *= (lEtaDet-1.);
      radX0 *= scaleRadX;
    }
    else if (lZ < 372){ dEdx = dEdX_ECal; radX0 = radX0_ECal; }//EE averaged out over a larger space
    else if (lZ < 398){ dEdx = dEdX_HCal*0.05; radX0 = radX0_Air; }//betw EE and HE
    else if (lZ < 555){ dEdx = dEdX_HCal*0.96; radX0 = radX0_HCal/0.96; } //HE calor abit less dense
    else {
      //iron .. don't care about no material in front of HF (too forward)
      if (! (lZ > 568 && lZ < 625 && lR > 85 ) // HE support 
	   && ! (lZ > 785 && lZ < 850 && lR > 118)) {dEdx = dEdX_Fe; radX0 = radX0_Fe; }
      else  { dEdx = dEdX_MCh; radX0 = radX0_MCh; } //ME at eta > 2.2
    }
  }
  else if (lR < 287){
    if (lZ < 372 && lR < 177){ 
      if (!(lR > 135 && lZ <343 && lZ > 304 )
	   && ! (lR > 156 && lZ < 372 && lZ > 343 && ((lZ-343.)< (lR-156.)*1.38)))
	{
	  //the crystals are the same length, but they are not 100% of material
	  double cosThetaEquiv = 0.8/sqrt(1.+lZ*lZ/lR/lR) + 0.2;
	  if (lZ > 343) cosThetaEquiv = 1.;
	  dEdx = dEdX_ECal*cosThetaEquiv; radX0 = radX0_ECal/cosThetaEquiv; 
	} //EB
      else { 
	if ( (lZ > 304 && lZ < 328 && lR < 177 && lR > 135) 
	     && ! (lZ > 311 && lR < 172) ) {dEdx = dEdX_Fe; radX0 = radX0_Fe; } //Tk_Support
	else {dEdx = dEdX_ECal*0.2; radX0 = radX0_Air;} //cables go here
      }
    }
    else if (lZ < 554){ 
      if ((lZ < 433 || lR < 264) && (lZ < 402 || lR < 275) && (lZ < 517 || lR < 246) //notches
	  //I should've made HE and HF different .. now need to shorten HE to match
	  && lZ < 548
	  && ! (lZ < 389 && lZ > 335 && lR < 193 ) //not a gap
	  && ! (lZ > 307 && lZ < 335 && lR < 193 && ((lZ - 307) > (lR - 177.)*1.739)) //not a gap
	  && ! (lR < 177 && lZ < 398) //under the HE nose
	  && ! (lR < 264 && lR > 175 && fabs(441.5 - lZ + (lR - 269.)/1.327) < 8.5) ) //not a gap
	{ dEdx = dEdX_HCal; radX0 = radX0_HCal; }//hcal
      else {dEdx = dEdX_HCal*0.05; radX0 = radX0_Air; }//endcap gap
    }
    else if (lZ < 564){
      if (lR < 251) {dEdx = dEdX_Fe; radX0 = radX0_Fe; }//HE support
      else { dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    }
    else if (lZ < 625){ dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    else if (lZ < 785){
      if (! (lR > 275 && lZ < 720)) { dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
      else { dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    }
    else if (lZ < 850){ dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    else if (lZ < 910){ dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
    else if (lZ < 975){ dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    else if (lZ < 1000){ dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
    else { dEdx = 0; radX0 = radX0_Air;}
  }
  else if (lR <380 && lZ < 667){
    if (lZ < 630) { dEdx = dEdX_coil; radX0 = radX0_coil; }//a guess for the solenoid average
    else {dEdx = 0; radX0 = radX0_Air; }//endcap gap
  }
  else {
    if (lZ < 667) {
      double bMag = sv.bf.mag();
      if (bMag > 0.75 && ! (lZ > 500 && lR <500 && bMag < 1.15)
	  && ! (lZ < 450 && lR > 420 && bMag < 1.15 ) )
	{ dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
      else { dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    } 
    else if (lZ < 724){ dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    else if (lZ < 785){ dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
    else if (lZ < 850){ dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    else if (lZ < 910){ dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
    else if (lZ < 975){ dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    else if (lZ < 1000){ dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
    else {dEdx = 0; radX0 = radX0_Air; }//air
  }
  
  dEdXPrime = dEdx == 0 ? 0 : -dEdx/dEdX_mat*(0.96/p0 + 0.033 - 0.022*pow(p0, -0.33))*1e-3; //== d(dEdX)/dp

  return dEdx;
}


int SteppingHelixPropagator::cIndex_(int ind) const{
  int result = ind%MAX_POINTS;  
  if (ind != 0 && result == 0){
    result = MAX_POINTS;
  }
  return result;
}

SteppingHelixPropagator::Result
SteppingHelixPropagator::refToDest(SteppingHelixPropagator::DestType dest, 
				   const SteppingHelixPropagator::StateInfo& sv,
				   const double pars[6], 
				   double& dist, double& tanDist, 
				   PropagationDirection& refDirection) const{
  Result result = SteppingHelixStateInfo::NOT_IMPLEMENTED;
  double curZ = sv.r3.z();
  double curR = sv.r3.perp();

  switch (dest){
  case RADIUS_DT:
    {
      double cosDPhiPR = cos((sv.r3.deltaPhi(sv.p3)));
      dist = pars[RADIUS_P] - curR;
      tanDist = dist/sv.p3.perp()*sv.p3.mag();
      refDirection = dist*cosDPhiPR > 0 ?
	alongMomentum : oppositeToMomentum;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case Z_DT:
    {
      dist = pars[Z_P] - curZ;
      tanDist = dist/sv.p3.z()*sv.p3.mag();
      refDirection = sv.p3.z()*dist > 0. ?
	alongMomentum : oppositeToMomentum;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case PLANE_DT:
    {
      Point rPlane(pars[0], pars[1], pars[2]);
      Vector nPlane(pars[3], pars[4], pars[5]);
      
      double dRDotN = (sv.r3 - rPlane).dot(nPlane);
      
      dist = fabs(dRDotN);
      tanDist = dist/sv.p3.dot(nPlane)*sv.p3.mag();
      refDirection = (sv.p3.dot(nPlane))*dRDotN < 0. ?
	alongMomentum : oppositeToMomentum;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case CONE_DT:
    {
      //assumes the cone axis/vertex is along z
      Point cVertex(pars[0], pars[1], pars[2]);
      Vector relV3 = sv.r3 - cVertex;
      double theta(pars[3]);
      if (cVertex.perp() < 1e-5){
	double sinDTheta = sin(theta-relV3.theta());
	double cosDTheta = cos(theta-relV3.theta());
	bool isInside = sin(theta) > sin(relV3.theta()) 
	  && cos(theta)*cos(relV3.theta()) > 0;
	dist = isInside || cosDTheta > 0 ? 
	  relV3.mag()*sinDTheta : relV3.mag();
	double normPhi = isInside ? 
	  Geom::pi() - relV3.phi() : relV3.phi();
	double normTheta = theta > Geom::pi()/2. ? 
	  (isInside ? 1.5*Geom::pi() - theta : theta - Geom::pi()/2.) 
	  : (isInside ? Geom::pi()/2 - theta : theta + Geom::pi()/2);
	//this is a normVector from the cone to the point
	Vector norm; norm.setRThetaPhi(fabs(dist), normTheta, normPhi);
	double cosDThetaP = cos(norm.theta() - sv.p3.theta());
	tanDist = dist/fabs(cosDThetaP);
	refDirection = norm.dot(sv.p3) > 0 ?
	  oppositeToMomentum : alongMomentum;
	if (debug_){
	  std::cout<<"refToDest:toCone the point is "
		   <<(isInside? "in" : "out")<<"side the cone"
		   <<std::endl;
	}
      }
    }
    break;
    //   case CYLINDER_DT:
    //     break;
  case PATHL_DT:
    {
      double curS = fabs(sv.path);
      dist = pars[PATHL_P] - curS;
      tanDist = dist;
      refDirection = pars[PATHL_P] > 0 ? 
	alongMomentum : oppositeToMomentum;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case POINT_PCA_DT:
    {
      Point pDest(pars[0], pars[1], pars[2]);
      dist = (sv.r3 - pDest).mag()+ 1e-24;//add a small number to avoid 1/0
      tanDist = (sv.r3 - pDest).dot(sv.p3)/(sv.p3.mag());
      refDirection = tanDist < 0 ?
	alongMomentum : oppositeToMomentum;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case LINE_PCA_DT:
    {
      Point rLine(pars[0], pars[1], pars[2]);
      Vector dLine(pars[3], pars[4], pars[5]);
      dLine = (dLine - rLine);
      dLine /= dLine.mag();

      Vector dR = sv.r3 - rLine;
      Vector dRPerp = dR - dLine*(dR.dot(dLine));
      dist = dRPerp.mag() + 1e-24;//add a small number to avoid 1/0
      tanDist = dRPerp.dot(sv.p3)/(sv.p3.mag());
      //angle wrt line
      double cosAlpha = dLine.dot(sv.p3)/sv.p3.mag();
      tanDist *= fabs(1./sqrt(fabs(1.-cosAlpha*cosAlpha)+1e-96));
      refDirection = tanDist < 0 ?
	alongMomentum : oppositeToMomentum;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  default:
    {
      //some large number
      dist = 1e12;
      tanDist = 1e12;
      refDirection = anyDirection;
      result = SteppingHelixStateInfo::NOT_IMPLEMENTED;
    }
    break;
  }

  if (debug_){
    std::cout<<"refToDest input: dest"<<dest<<" pars[]: ";
    for (int i = 0; i < 6; i++){
      std::cout<<", "<<i<<" "<<pars[i];
    }
    std::cout<<std::endl;
    std::cout<<"refToDest output: "
	     <<"\t dist"<< dist
	     <<"\t tanDist"<< tanDist      
      	     <<"\t refDirection"<< refDirection
	     <<std::endl;
  }

  return result;
}

SteppingHelixPropagator::Result
SteppingHelixPropagator::refToMagVolume(const SteppingHelixPropagator::StateInfo& sv,
					PropagationDirection dir,
					double& dist, double& tanDist) const{

  Result result = SteppingHelixStateInfo::NOT_IMPLEMENTED;
  const MagVolume* cVol = sv.magVol;

  if (cVol == 0) return result;
  const std::vector<VolumeSide> cVolFaces = cVol->faces();

  double distToFace[6];
  double tanDistToFace[6];
  PropagationDirection refDirectionToFace[6];
  Result resultToFace[6];
  int iFDest = -1;
  
  if (debug_){
    std::cout<<"Trying volume "<<DDSolidShapesName::name(cVol->shapeType())
	     <<" with "<<cVolFaces.size()<<" faces"<<std::endl;
  }

  for (uint iFace = 0; iFace < cVolFaces.size(); iFace++){
    if (iFace > 5){
      std::cout<<"Too many faces"<<std::endl;
    }
    if (debug_){
      std::cout<<"Start with face "<<iFace<<std::endl;
    }
    const Plane* cPlane = dynamic_cast<const Plane*>(&cVolFaces[iFace].surface());
    const Cylinder* cCyl = dynamic_cast<const Cylinder*>(&cVolFaces[iFace].surface());
    const Cone* cCone = dynamic_cast<const Cone*>(&cVolFaces[iFace].surface());
    if (debug_){
      if (cPlane!=0) std::cout<<"The face is a plane at "<<cPlane<<std::endl;
      if (cCyl!=0) std::cout<<"The face is a cylinder at "<<cCyl<<std::endl;
    }

    double pars[6];
    DestType dType = UNDEFINED_DT;
    if (cPlane != 0){
      GlobalPoint rPlane = cPlane->toGlobal(LocalPoint(0,0,0));
      GlobalVector nPlane = cPlane->toGlobal(LocalVector(0,0,1.)); nPlane = nPlane.unit();
      
      if (sv.r3.z() < 0){
	pars[0] = rPlane.x(); pars[1] = rPlane.y(); pars[2] = rPlane.z();
	pars[3] = nPlane.x(); pars[4] = nPlane.y(); pars[5] = nPlane.z();
      } else {
	pars[0] = rPlane.x(); pars[1] = rPlane.y(); pars[2] = -rPlane.z();
	pars[3] = nPlane.x(); pars[4] = nPlane.y(); pars[5] = -nPlane.z();
      }
      dType = PLANE_DT;
    } else if (cCyl != 0){
      if (debug_){
	std::cout<<"Cylinder at "<<cCyl->position()
		 <<" rorated by "<<cCyl->rotation()
		 <<std::endl;
      }
      pars[RADIUS_P] = cCyl->radius();
      dType = RADIUS_DT;
    } else if (cCone != 0){
      if (debug_){
	std::cout<<"Cone at "<<cCone->position()
		 <<" rorated by "<<cCone->rotation()
		 <<" vertex at "<<cCone->vertex()
		 <<" angle of "<<cCone->openingAngle()
		 <<std::endl;
      }
      if (sv.r3.z() < 0){
	pars[0] = cCone->vertex().x(); pars[1] = cCone->vertex().y(); 
	pars[2] = cCone->vertex().z();
	pars[3] = cCone->openingAngle();
      } else {
	pars[0] = cCone->vertex().x(); pars[1] = cCone->vertex().y(); 
	pars[2] = -cCone->vertex().z();
	pars[3] = Geom::pi() - cCone->openingAngle();
      }
      dType = CONE_DT;
    } else {
      std::cout<<"Unknown surface"<<std::endl;
      resultToFace[iFace] = SteppingHelixStateInfo::UNDEFINED;
      continue;
    }
    resultToFace[iFace] = 
      refToDest(dType, sv, pars, 
		distToFace[iFace], tanDistToFace[iFace], refDirectionToFace[iFace]);
    
    if (refDirectionToFace[iFace] == dir){
      double sign = dir == alongMomentum ? 1. : -1.;
      GlobalPoint gPointEst(sv.r3.x(), sv.r3.y(), sv.r3.z());
      GlobalVector gDir(sv.p3.x(), sv.p3.y(), sv.p3.z());
      gDir /= sv.p3.mag();
      gPointEst += sign*fabs(fabs(distToFace[iFace])-2e-4)*gDir;
      if (debug_){
	std::cout<<"Linear est point closer to the face less 2 um "<<gPointEst
		 <<std::endl;
      }
      GlobalPoint gPointEstNegZ(gPointEst.x(), gPointEst.y(),
				gPointEst.z() > 0 ? -gPointEst.z() : gPointEst.z());
      if ( cVol->inside(gPointEstNegZ) ){
	if (debug_){
	  std::cout<<"The point is inside the volume"<<std::endl;
	}
	//OK, guessed a point still inside the volume
	if (iFDest == -1){
	  iFDest = iFace;
	} else {
	  if (fabs(tanDistToFace[iFDest]) > fabs(tanDistToFace[iFace])){
	    iFDest = iFace;
	  }
	}
      } else {
	if (debug_){
	  std::cout<<"The point is NOT inside the volume"<<std::endl;
	}
      }
    }

  }
  if (iFDest != -1){
    result = SteppingHelixStateInfo::OK;
    dist = distToFace[iFDest];
    tanDist = tanDistToFace[iFDest];
    if (debug_){
      std::cout<<"Got a point near closest boundary -- face "<<iFDest<<std::endl;
    }
  } else {
    if (debug_){
      std::cout<<"Failed to find a dest point inside the volume"<<std::endl;
    }
  }

  return result;
}

//transforms 6x6 "local" cov matrix (r_3x3, p_3x3)
void SteppingHelixPropagator::initCovRotation(const SteppingHelixPropagator::Vector* repI[3], 
					      const SteppingHelixPropagator::Vector* repF[3],
					      HepMatrix& covRot) const{
  

  covRot*=0; //reset
  //fill in a block-diagonal rotation
  for (int i = 1; i <=3; i++){
    for (int j = 1; j <=3; j++){
      double r_ij = (*repF[i-1]).dot(*repI[j-1]);
      covRot(i,j) = r_ij;
      covRot(i+3,j+3) = r_ij;
    }
  }
}


void SteppingHelixPropagator::getLocBGrad(SteppingHelixPropagator::StateInfo& sv,
					  double delta) const{
  Point r3[3];
  r3[0] = sv.r3 + sv.rep.lX*delta;
  r3[1] = sv.r3 + sv.rep.lY*delta;
  r3[2] = sv.r3 + sv.rep.lZ*delta;

  double bVal[3];
  double bVal0 = sv.bf.mag();
  for (int i = 0; i < 3; i++){
    bVal[i] = field_->inTesla(GlobalPoint(r3[i].x(), r3[i].y(), r3[i].z())).mag();
  }
  sv.bfGradLoc.set((bVal[0] -bVal0)/delta, (bVal[1] -bVal0)/delta, (bVal[2] -bVal0)/delta);
}

