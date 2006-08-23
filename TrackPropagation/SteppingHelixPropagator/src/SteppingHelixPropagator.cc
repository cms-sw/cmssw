/** \class SteppingHelixPropagator
 *  Propagator implementation using steps along a helix.
 *  Minimal geometry navigation.
 *  Material effects (multiple scattering and energy loss) are based on tuning
 *  to MC and (eventually) data. 
 *  Implementation file contents follow.
 *
 *  $Date: 2006/08/23 19:07:48 $
 *  $Revision: 1.9 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Fri Mar  3 16:01:24 CST 2006
// $Id: SteppingHelixPropagator.cc,v 1.9 2006/08/23 19:07:48 slava77 Exp $
//
//


#include "MagneticField/Engine/interface/MagneticField.h"
#include "Utilities/Timing/interface/TimingReport.h"

#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"

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
  for (int i = 0; i <= MAX_POINTS; i++){
    covLoc_[i] = HepSymMatrix(6,0);
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
  GlobalPoint rPlane = pDest.toGlobal(LocalPoint(0,0,0));
  GlobalVector nPlane = pDest.toGlobal(LocalVector(0,0,1.)); nPlane = nPlane.unit();

  double pars[6] = { rPlane.x(), rPlane.y(), rPlane.z(),
		     nPlane.x(), nPlane.y(), nPlane.z() };

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
  Result result = propagate(PLANE_DT, pars);
  if (result != OK ) return TsosPP();

  Vector p3F;
  Point  r3F;
  HepSymMatrix covF;

  
  getFState(p3F, r3F, covF); 
  GlobalVector p3FGV(p3F.x(), p3F.y(), p3F.z());
  GlobalPoint r3FGP(r3F.x(), r3F.y(), r3F.z());
  SurfaceSide side = atCenterOfSurface;
  GlobalTrajectoryParameters tParsDest(r3FGP, p3FGV, charge, field_);
  CartesianTrajectoryError tCovDest(covF);
  FreeTrajectoryState ftsDest = (ftsStart.hasError()  && !noErrorPropagation_) 
    ? FreeTrajectoryState(tParsDest, tCovDest) 
    : FreeTrajectoryState(tParsDest);
  if (ftsDest.hasError()) ftsDest.curvilinearError(); //call it so it gets created

  TrajectoryStateOnSurface tsosDest = TrajectoryStateOnSurface(ftsDest, pDest, side);
  int cInd = cIndex_(nPoints_-1);
  
  return TsosPP(tsosDest, path_[cInd]);
}

std::pair<TrajectoryStateOnSurface, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const Cylinder& cDest) const {
  //need to get rid of these conversions .. later
  GlobalVector p3GV = ftsStart.momentum();
  GlobalPoint r3GP = ftsStart.position();
  Vector p3(p3GV.x(), p3GV.y(), p3GV.z());
  Point  r3(r3GP.x(), r3GP.y(), r3GP.z());
  double pars[6];
  pars[RADIUS_P] = cDest.radius();

  int charge = ftsStart.charge();

  setIState(p3, r3, charge,  
	    (ftsStart.hasError() && !noErrorPropagation_) 
	    ? ftsStart.cartesianError().matrix() : HepSymMatrix(1,0),
	    propagationDirection());
  Result result = propagate(RADIUS_DT, pars);
  if (result != OK) return TsosPP();


  Vector p3F;
  Point  r3F;
  HepSymMatrix covF;

  
  getFState(p3F, r3F, covF); 
  GlobalVector p3FGV(p3F.x(), p3F.y(), p3F.z());
  GlobalPoint r3FGP(r3F.x(), r3F.y(), r3F.z());
  SurfaceSide side = atCenterOfSurface;
  GlobalTrajectoryParameters tParsDest(r3FGP, p3FGV, charge, field_);
  CartesianTrajectoryError tCovDest(covF);
  FreeTrajectoryState ftsDest = (ftsStart.hasError()  && !noErrorPropagation_) 
    ? FreeTrajectoryState(tParsDest, tCovDest) 
    : FreeTrajectoryState(tParsDest);
  if (ftsDest.hasError()) ftsDest.curvilinearError(); //call it so it gets created

  TrajectoryStateOnSurface tsosDest = TrajectoryStateOnSurface(ftsDest, cDest, side);
  int cInd = cIndex_(nPoints_-1);

  return TsosPP(tsosDest, path_[cInd]);
}


std::pair<FreeTrajectoryState, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const GlobalPoint& pDest) const {
  //need to get rid of these conversions .. later
  GlobalVector p3GV = ftsStart.momentum();
  GlobalPoint r3GP = ftsStart.position();
  Vector p3(p3GV.x(), p3GV.y(), p3GV.z());
  Point  r3(r3GP.x(), r3GP.y(), r3GP.z());
  double pars[6] = {pDest.x(), pDest.y(), pDest.z(), 0, 0, 0};

  int charge = ftsStart.charge();

  setIState(p3, r3, charge,  
	    (ftsStart.hasError() && !noErrorPropagation_) 
	    ? ftsStart.cartesianError().matrix() : HepSymMatrix(1,0),
	    propagationDirection());
  Result result = propagate(POINT_PCA_DT, pars);
  if (result != OK) return FtsPP();


  Vector p3F;
  Point  r3F;
  HepSymMatrix covF;

  
  getFState(p3F, r3F, covF); 
  GlobalVector p3FGV(p3F.x(), p3F.y(), p3F.z());
  GlobalPoint r3FGP(r3F.x(), r3F.y(), r3F.z());
  GlobalTrajectoryParameters tParsDest(r3FGP, p3FGV, charge, field_);
  CartesianTrajectoryError tCovDest(covF);

  FreeTrajectoryState ftsDest = (ftsStart.hasError()  && !noErrorPropagation_) 
    ? FreeTrajectoryState(tParsDest, tCovDest) 
    : FreeTrajectoryState(tParsDest);
  if (ftsDest.hasError()) ftsDest.curvilinearError(); //call it so it gets created
  int cInd = cIndex_(nPoints_-1);

  return FtsPP(ftsDest, path_[cInd]);
}

std::pair<FreeTrajectoryState, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const GlobalPoint& pDest1, const GlobalPoint& pDest2) const {
  if ((pDest1-pDest2).mag() < 1e-10) return FtsPP();
  //need to get rid of these conversions .. later
  GlobalVector p3GV = ftsStart.momentum();
  GlobalPoint r3GP = ftsStart.position();
  Vector p3(p3GV.x(), p3GV.y(), p3GV.z());
  Point  r3(r3GP.x(), r3GP.y(), r3GP.z());
  double pars[6] = {pDest1.x(), pDest1.y(), pDest1.z(),
		    pDest2.x(), pDest2.y(), pDest2.z()};

  int charge = ftsStart.charge();

  setIState(p3, r3, charge,  
	    (ftsStart.hasError() && !noErrorPropagation_) 
	    ? ftsStart.cartesianError().matrix() : HepSymMatrix(1,0),
	    propagationDirection());
  Result result = propagate(LINE_PCA_DT, pars);
  if (result != OK) return FtsPP();


  Vector p3F;
  Point  r3F;
  HepSymMatrix covF;

  
  getFState(p3F, r3F, covF); 
  GlobalVector p3FGV(p3F.x(), p3F.y(), p3F.z());
  GlobalPoint r3FGP(r3F.x(), r3F.y(), r3F.z());
  GlobalTrajectoryParameters tParsDest(r3FGP, p3FGV, charge, field_);
  CartesianTrajectoryError tCovDest(covF);

  FreeTrajectoryState ftsDest = (ftsStart.hasError()  && !noErrorPropagation_) 
    ? FreeTrajectoryState(tParsDest, tCovDest) 
    : FreeTrajectoryState(tParsDest);
  if (ftsDest.hasError()) ftsDest.curvilinearError(); //call it so it gets created
  int cInd = cIndex_(nPoints_-1);

  return FtsPP(ftsDest, path_[cInd]);
}


void SteppingHelixPropagator::setIState(const SteppingHelixPropagator::Vector& p3, 
					const SteppingHelixPropagator::Point& r3, int charge, 
					const HepSymMatrix& cov, PropagationDirection dir) const {
  nPoints_ = 0;
  loadState(0, p3, r3, charge, cov, dir);
  nPoints_++;
}

void SteppingHelixPropagator::getFState(SteppingHelixPropagator::Vector& p3, 
					SteppingHelixPropagator::Point& r3, 
					HepSymMatrix& cov) const{
  int cInd = cIndex_(nPoints_-1);
  p3 = p3_[cInd];
  r3 = r3_[cInd];
  //update Emat only if it's valid
  if (covLoc_[cInd].num_row() >=5){
    Vector xRep(1., 0., 0.);
    Vector yRep(0., 1., 0.);
    Vector zRep(0., 0., 1.);
    const Vector* repI[3] = {&reps_[cInd].lX, &reps_[cInd].lY, &reps_[cInd].lZ};
    const Vector* repF[3] = {&xRep, &yRep, &zRep};
    initCovRotation(repI, repF, covRot_);
    cov = covLoc_[cInd].similarity(covRot_);
  } else {
    cov = covLoc_[cInd];
  }

}


SteppingHelixPropagator::Result 
SteppingHelixPropagator::propagate(SteppingHelixPropagator::DestType type, 
				   const double pars[6], double epsilon)  const{

  //check if it's going to work at all
  double secTheta = 0;
  double dist = 0;
  bool isIncoming;
  Result result = refToDest(type, nPoints_-1, pars, dist, secTheta, isIncoming);

  if (result != OK ) return result;

  result = UNDEFINED;
  bool makeNextStep = true;
  double dStep = 1.;
  PropagationDirection dir,oldDir;
  dir = propagationDirection(); 
  oldDir = dir;
  int nOsc = 0;
  int cInd = 0;

  while (makeNextStep){
    cInd = cIndex_(nPoints_-1);
    double curZ = r3_[cInd].z();
    double curR = r3_[cInd].perp();
    refToDest(type, nPoints_-1, pars, dist, secTheta, isIncoming);

    if (propagationDirection() == anyDirection){
      dir = isIncoming ? alongMomentum : oppositeToMomentum;
    } else {
      dir = propagationDirection();
    }

    if ((fabs(curZ) > 1.5e3 || curR >800.) && dir == alongMomentum) 
      dStep = fabs(dist*secTheta) -1e-12;
    if (fabs(dist*secTheta) < dStep){
      dStep = fabs(dist*secTheta); 
    }
    if (dStep > 1e-10){
      makeAtomStep(nPoints_-1, dStep, dir, HEL_AS_F);
      nPoints_++;   cInd = cIndex_(nPoints_-1);
    }
    if (oldDir != dir) nOsc++;
    oldDir = dir;

    if (fabs(dist) < fabs(epsilon)  ) result = OK;

    if ((type == POINT_PCA_DT || type == LINE_PCA_DT )
	&& fabs(dStep) < fabs(epsilon)  ){
      //now check if it's not a branch point (peek ahead at 1 cm)
      double nextDist = 0;
      double nextSecTheta = 0;
      bool nextIsIncoming = false;
      makeAtomStep(nPoints_-1, 1., dir, HEL_AS_F);
      nPoints_++; cInd = cIndex_(nPoints_-1);
      refToDest(type, nPoints_-1, pars, nextDist, nextSecTheta, nextIsIncoming);
      if ( fabs(nextDist) > fabs(dist)){
	nPoints_--;   cInd = cIndex_(nPoints_-1);
	result = OK;
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

    if (nPoints_ > MAX_STEPS || nOsc > 6) result = FAULT;

    if (p3_[cInd].mag() < 0.1 ) result = RANGEOUT;

    if ( curR > 20000 || fabs(curZ) > 20000 ) result = INACC;

    makeNextStep = result == UNDEFINED;
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
    std::cout<<"Made "<<nPoints_-1<<" steps and stopped at(cur step) "<<r3_[cInd]<<std::endl;
  }
  
  return result;
}
  
void SteppingHelixPropagator::loadState(int ind, 
					const SteppingHelixPropagator::Vector& p3, 
					const SteppingHelixPropagator::Point& r3, int charge,
					const HepSymMatrix& cov, PropagationDirection dir) const{
  int cInd = cIndex_(ind);
  q_[cInd] = charge;
  p3_[cInd] = p3;
  r3_[cInd] = r3;
  dir_[cInd] = dir == alongMomentum ? 1.: -1.;

  path_[cInd] = 0; // this could've held the initial path
  radPath_[cInd] = 0;

  GlobalVector bf = field_->inTesla(GlobalPoint(r3.x(), r3.y(), r3.z()));
  
  bf_[cInd].set(bf.x(), bf.y(), bf.z());
  if (bf_[cInd].mag() < 1e-6) bf_[cInd].set(0., 0., 1e-6);

  setReps(ind);
  //  getLocBGrad(ind, 1e-1);

  covLoc_[cInd].assign(cov);

  
  //update Emat only if it's valid
  if (covLoc_[cInd].num_row() >=5){
    Vector xRep(1., 0., 0.);
    Vector yRep(0., 1., 0.);
    Vector zRep(0., 0., 1.);
    const Vector* repI[3] = {&xRep, &yRep, &zRep};
    const Vector* repF[3] = {&reps_[cInd].lX, &reps_[cInd].lY, &reps_[cInd].lZ};
    initCovRotation(repI, repF, covRot_);
    covLoc_[cInd] = covLoc_[cInd].similarity(covRot_);
    //    for (int ii = 1; ii<= 6; ii++) covLoc_[cInd](1,ii) = 0;
  }

  if (debug_){
    std::cout<<"Loaded at "<<ind<<" path: "<<path_[cInd]<<" radPath: "<<radPath_[cInd]
	     <<" p3 "<<" pt: "<<p3_[cInd].perp()<<" phi: "<<p3_[cInd].phi()
	     <<" eta: "<<p3_[cInd].eta()
	     <<" "<<p3_[cInd]
	     <<" r3: "<<r3_[cInd]
	     <<" bField: "<<bf_[cInd].mag()
	     <<std::endl;
    std::cout<<"Input Covariance in Global RF "<<cov<<std::endl;
    std::cout<<"Covariance in Local RF "<<covLoc_[cInd]<<std::endl;
    std::cout<<"Rotated by "<<covRot_<<std::endl;
  }
  //   std::cout<<"Load at "<<ind<<" path: "<<path_[cInd]
  //  	   <<" p3 "<<" pt: "<<p3_[cInd].perp()<<" phi: "<<p3_[cInd].phi()<<" eta: "<<p3_[cInd].eta()
  // 	   <<" "<<p3_[cInd]
  //  	   <<" r3: "<<r3_[cInd]<<std::endl;
}

void SteppingHelixPropagator::incrementState(int ind, 
					     double dP, SteppingHelixPropagator::Vector tau,
					     double dX, double dY, double dZ, double dS, double dX0,
					     const HepMatrix& dCovTransform) const{
  //  TimeMe locTimer("SteppingHelixPropagator::incrementState");
  if (ind ==0) return;
  int iPrev = ind-1;
  int cInd = cIndex_(ind);
  int cPrev = cIndex_(iPrev);
  q_[cInd] = q_[cPrev];
  dir_[cInd] = dS > 0.0 ? 1.: -1.; 
  //  std::cout<<tau.deltaPhi(p3_[cPrev])<<std::endl;
  p3_[cInd] = tau;  p3_[cInd]*=(p3_[cPrev].mag() - dir_[cInd]*fabs(dP));

  r3_[cInd] = r3_[cPrev];
  Vector tmpR3 = reps_[cPrev].lX; tmpR3*=dX;
  r3_[cInd]+= tmpR3;
  tmpR3 = reps_[cPrev].lY; tmpR3*=dY;
  r3_[cInd]+= tmpR3;
  tmpR3 = reps_[cPrev].lZ; tmpR3*=dZ;
  r3_[cInd]+= tmpR3;
  path_[cInd] = path_[cPrev] + dS;
  radPath_[cInd] = radPath_[cPrev] + dX0;


  GlobalVector bf = field_->inTesla(GlobalPoint(r3_[cInd].x(), r3_[cInd].y(), r3_[cInd].z()));
  
  bf_[cInd].set(bf.x(), bf.y(), bf.z());
  if (bf_[cInd].mag() < 1e-6) bf_[cInd].set(0., 0., 1e-6);
  
  setReps(ind);
  //  getLocBGrad(ind, 1e-1);
  
  
  //update Emat only if it's valid
  if (covLoc_[cPrev].num_row() >=5){
    const Vector* repI[3] = {&reps_[cPrev].lX, &reps_[cPrev].lY, &reps_[cPrev].lZ};
    const Vector* repF[3] = {&reps_[cInd].lX, &reps_[cInd].lY, &reps_[cInd].lZ};
    initCovRotation(repI, repF, covRot_);
    covRot_ = covRot_*dCovTransform;
    covLoc_[cInd] = covLoc_[cPrev].similarity(covRot_);
  } else {
    covLoc_[cInd].assign(covLoc_[cPrev]);
  }

  if (debug_){
    std::cout<<"Now at "<<ind<<" path: "<<path_[cInd]<<" radPath: "<<radPath_[cInd]
	     <<" p3 "<<" pt: "<<p3_[cInd].perp()<<" phi: "<<p3_[cInd].phi()
	     <<" eta: "<<p3_[cInd].eta()
	     <<" "<<p3_[cInd]
	     <<" r3: "<<r3_[cInd]
	     <<" dPhi: "<<acos(p3_[cInd].unit().dot(p3_[cPrev].unit()))
	     <<" bField: "<<bf_[cInd].mag()
	     <<std::endl;
    std::cout<<"Covariance in Local RF "<<covLoc_[cInd]<<std::endl;
    std::cout<<"Transformed from prev by "<<covRot_<<std::endl;
    std::cout<<"dCovTransform "<<dCovTransform<<std::endl;

    Vector xRep(1., 0., 0.);
    Vector yRep(0., 1., 0.);
    Vector zRep(0., 0., 1.);
    const Vector* repF[3] = {&xRep, &yRep, &zRep};
    const Vector* repI[3] = {&reps_[cInd].lX, &reps_[cInd].lY, &reps_[cInd].lZ};
    initCovRotation(repI, repF, covRot_);    
    HepSymMatrix cov = covLoc_[cInd].similarity(covRot_);
    std::cout<<"Covariance in Global RF "<<cov<<std::endl;
    std::cout<<"Rotated by "<<covRot_<<std::endl;
  }
}

void SteppingHelixPropagator::setReps(int ind) const{
  int cInd = cIndex_(ind);

  Vector zRep(0., 0., 1.);
  Vector tau = p3_[cInd]/(p3_[cInd].mag());
  reps_[cInd].lX = tau;
  reps_[cInd].lY = zRep.cross(tau); reps_[cInd].lY /= tau.perp();
  reps_[cInd].lZ = reps_[cInd].lX.cross(reps_[cInd].lY);
}

bool SteppingHelixPropagator::makeAtomStep(int iIn, double dS, 
					   PropagationDirection dir, 
					   SteppingHelixPropagator::Fancy fancy) const{
  //  TimeMe locTimer("SteppingHelixPropagator::makeAtomStep");
  int cInd = cIndex_(iIn);
  if (debug_){
    std::cout<<"Make atom step "<<iIn<<" with step "<<dS<<" in direction "<<dir<<std::endl;
  }

  //  HepMatrix dCTr(HepDiagMatrix(6,1));//unit transform is the default
  double dP = 0;
  Vector tau = p3_[cInd]; tau/=tau.mag();

  dS = dir == alongMomentum ? fabs(dS) : -fabs(dS);

  double p0 = p3_[cInd].mag();
  double b0 = bf_[cInd].mag();
  double kappa0 = 0.0029979*q_[cInd]*b0/p0;
  if (fabs(kappa0) < 1e-12) kappa0 = 1e-12;

  double cosTheta = tau.z();
  double sinTheta = sin(acos(cosTheta));
  double cotTheta = fabs(sinTheta) > 1e-21 ? cosTheta/sinTheta : 1e21;
  //  double tanTheta = fabs(cosTheta) > 1e-21 ? sinTheta/cosTheta : 1e21;
  double phi = kappa0*dS;
  double cosPhi = cos(phi);
  double oneLessCosPhi = 1.-cosPhi;
  double sinPhi = sin(phi);
  double phiLessSinPhi = phi - sinPhi;
  double oneLessCpLessPSp = oneLessCosPhi - phi*sinPhi;
  double pCpLessSp = phi*cosPhi - sinPhi;
  Vector bHat = bf_[cInd]; bHat /= bHat.mag();
  double bx = reps_[cInd].lX.dot(bHat);
  double by = reps_[cInd].lY.dot(bHat);
  double bz = reps_[cInd].lZ.dot(bHat);
  double oneLessBx2 = (1.-bx*bx);

  //components in local rf
  double dX =0.;
  double dY =0.;
  double dZ =0.;
  double tauX =0.;
  double tauY =0.;
  double tauZ =0.;
  
//   double bfLGL[3];// grad(log(B))
//   for (int i = 0; i < 3; i++){
//     if (b0 < 1e-6){
//       bfLGL[i] = 0.;
//     } else {
//       bfLGL[i] = bfGradLoc_[cInd][i]; bfLGL[i]/=b0;
//     }
//   }

  double dEdXPrime = 0;
  double radX0 = 1e24;
  double dEdx = getDeDx(iIn, dEdXPrime, radX0);
  double theta02 = 14.e-3/p0*sqrt(fabs(dS)/radX0); // .. drop log term (this is non-additive)
  theta02 *=theta02;
  if (applyRadX0Correction_){
    // this provides the integrand for theta^2
    // if summed up along the path, should result in 
    // theta_total^2 = Int_0^x0{ f(x)dX} = (13.6/p0)^2*x0*(1+0.036*ln(x0+1))
    // x0+1 above is to make the result infrared safe.
    double x0 = fabs(radPath_[cInd]);
    double dX0 = fabs(dS)/radX0;
    double alphaX0 = 13.6e-3/p0; alphaX0 *= alphaX0;
    double betaX0 = 0.038;
    theta02 = dX0*alphaX0*(1+betaX0*log(x0+1))*(1 + betaX0*log(x0+1) + 2.*betaX0*x0/(x0+1) );
  }

  Vector tmpR3;
  
  double epsilonP0 = 0;
  double omegaP0 = 0;

  switch (fancy){
  case HEL_AS_F:
  case HEL_ALL_F:
    dP = dEdx*dS;
    tauX = (1.0 - oneLessCosPhi*oneLessBx2);
    tauY = (oneLessCosPhi*bx*by - sinPhi*bz);
    tauZ = (oneLessCosPhi*bx*bz + sinPhi*by);

    epsilonP0 = 1.+ dP/p0;
    omegaP0 = 1.0 + dS*dEdXPrime;

    tmpR3 = reps_[cInd].lX; tmpR3*=tauX;
    tau = tmpR3;
    tmpR3 = reps_[cInd].lY;  tmpR3*=tauY;
    tau+=tmpR3;
    tmpR3 = reps_[cInd].lZ;  tmpR3*=tauZ;
    tau+=tmpR3;
    //the stuff above is

    dX = dS - phiLessSinPhi/kappa0*oneLessBx2;
    dY = 1./kappa0*(bx*by*phiLessSinPhi - oneLessCosPhi*bz);
    dZ = 1./kappa0*(bx*bz*phiLessSinPhi + oneLessCosPhi*by);

    dCTransform_ = unit66_;
    //     //yuck
    //case I: no "spatial" derivatives |--> dCtr({1,2,3,4,5,6}{1,2,3}) = 0    
    dCTransform_(1,4) += -dS/(phi*p0)*pCpLessSp*oneLessBx2;
    dCTransform_(1,5) += - dY/p0;
    dCTransform_(1,6) +=   dZ/p0;

    dCTransform_(2,4) += dS/phi/p0*(bx*by*pCpLessSp - bz*oneLessCpLessPSp);
    dCTransform_(2,5) +=   dX/p0;
    dCTransform_(2,6) += - cotTheta*dY/p0;

    //    dCTransform_(3,4) += dS/phi/p0*(bx*by*pCpLessSp + by*oneLessCpLessPSp) - 2.*dZ/p0;
    dCTransform_(3,4) += dS/phi/p0*(bx*by*pCpLessSp + by*oneLessCpLessPSp) - 3.*dZ/p0;
    dCTransform_(3,5) += cotTheta*dY/p0;
    dCTransform_(3,6) += dX/p0;


    dCTransform_(4,4) += tauX*omegaP0 - 1.0 + phi*epsilonP0*oneLessBx2*sinPhi;
    dCTransform_(4,5) += -tauY*epsilonP0;
    dCTransform_(4,6) +=  tauZ*epsilonP0;

    dCTransform_(5,4) += tauY*omegaP0 - phi*epsilonP0*(bx*by*sinPhi - bz*cosPhi);
    dCTransform_(5,5) += tauX*epsilonP0 - 1.; 
    dCTransform_(5,6) += - cotTheta*tauY*epsilonP0;
    
    //    dCTransform_(6,4) += tauZ*omegaP0 - phi*epsilonP0*(bx*bz*sinPhi + by*cosPhi) 
    // - 2.*tauZ*epsilonP0;
    dCTransform_(6,4) += tauZ*omegaP0 - phi*epsilonP0*(bx*bz*sinPhi + by*cosPhi) - 3.*tauZ*epsilonP0;
    dCTransform_(6,5) += cotTheta*tauY*epsilonP0;
    dCTransform_(6,6) += tauX*epsilonP0 - 1.;
    
    covLoc_[cInd](2,2) += theta02*dS*dS/3.;
    covLoc_[cInd](3,3) += theta02*dS*dS/3.;
    covLoc_[cInd](5,5) += theta02*p0*p0;
    covLoc_[cInd](6,6) += theta02*p0*p0;
    covLoc_[cInd](2,5) += theta02*dS*p0/2.;
    covLoc_[cInd](3,6) += theta02*dS*p0/2.;

    covLoc_[cInd](4,4) += dP*dP*1.6/dS*(1.0 + p0*1e-3); 
    //another guess .. makes sense for 1 cm steps 2./dS == 2 [cm] / dS [cm] at low pt
    //double it by 1TeV
    //not gaussian anyways
    // derived from the fact that sigma_p/eLoss ~ 0.08 after ~ 200 steps


    break;
  case POL_1_F:
  case POL_2_F:
  case POL_M_F:
    //FIXME: this is still in Bfield rf
    tau = reps_[cInd].lX*phi*sinTheta + reps_[cInd].lY*sinTheta + reps_[cInd].lZ*cosTheta;
    dP = dEdx*dS;
    dX = phi*dS/2.*sinTheta;
    dY = dS*sinTheta;
    dZ = dS*cosTheta;    
    break;
  default:
    break;
  }

  if (dir == oppositeToMomentum) dP = -fabs(dP);
  dP = dP > p0 ? p0-1e-5 : dP;
  incrementState(iIn+1, dP, tau, dX, dY, dZ, dS, dS/radX0,
		 dCTransform_);
  return true;
}

double SteppingHelixPropagator::getDeDx(int iIn, double& dEdXPrime, double& radX0) const{
  radX0 = 1.e24;
  dEdXPrime = 0.;
  if (noMaterialMode_) return 0;
  int cInd = cIndex_(iIn);

  double dEdx = 0.;

  double lR = r3_[cInd].perp();
  double lZ = fabs(r3_[cInd].z());

  //assume "Iron" .. seems to be quite the same for brass/iron/PbW04
  //good for Fe within 3% for 0.2 GeV to 10PeV
  double p0 = p3_[cInd].mag();

  //0.065 (PDG) --> 0.044 to better match with MPV
  double dEdX_mat = -(11.4 + 0.96*fabs(log(p0*2.8)) + 0.033*p0*(1.0 - pow(p0, -0.33)) )*1e-3; 
  //in GeV/cm .. 0.8 to get closer to the median or MPV
  double dEdX_HCal = 0.95*dEdX_mat; //extracted from sim
  double dEdX_ECal = 0.45*dEdX_mat;
  double dEdX_coil = 0.35*dEdX_mat; //extracted from sim .. closer to 40% in fact
  double dEdX_Fe =   dEdX_mat;
  double dEdX_MCh =  0.053*dEdX_mat; //chambers on average
  double dEdX_Trk =  0.0114*dEdX_mat;

  double radX0_HCal = 1.44/0.8; //guessing
  double radX0_ECal = 0.89/0.7;
  double radX0_coil = 4.; //
  double radX0_Fe =   1.76;
  double radX0_MCh =  1e3; //
  double radX0_Trk =  500.;
  double radX0_Air =  3.e4;


  //this should roughly figure out where things are 
  //(numbers taken from Fig1.1.2 TDR and from geom xmls)
  if (lR < 129){
    if (lZ < 294){ dEdx = dEdx = dEdX_Trk; radX0 = radX0_Trk; }
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
      if (!(lR > 134 && lZ <343 && lZ > 304 )
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
      double bMag = bf_[cInd].mag();
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
  
  dEdXPrime = dEdx == 0 ? 0 : dEdx/dEdX_mat*(2.4/p0)*1e-3*0.935; //== d(dEdX)/dp

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
				   int ind, const double pars[6], 
				   double& dist, double& secTheta, bool& isIncoming) const{
  Result result = NOT_IMPLEMENTED;
  int cInd = cIndex_(ind);
  double curZ = r3_[cInd].z();
  double curR = r3_[cInd].perp();
  
  switch (dest){
  case RADIUS_DT:
    {
      dist = pars[RADIUS_P] - curR;
      double cosDPhiPR = cos((r3_[cInd].deltaPhi(p3_[cInd])));
      secTheta = 1./p3_[cInd].perp()*p3_[cInd].mag();
      isIncoming = (dist*cosDPhiPR > 0 || curR < 2e-1);
      result = OK;
    }
    break;
  case Z_DT:
    {
      dist = pars[Z_P] - curZ;
      secTheta = 1./p3_[cInd].z()*p3_[cInd].mag();
      isIncoming = p3_[cInd].z()*dist > 0.;
      result = OK;
    }
    break;
  case PLANE_DT:
    {
      Point rPlane(pars[0], pars[1], pars[2]);
      Vector nPlane(pars[3], pars[4], pars[5]);
      
      double dRDotN = (r3_[cInd] - rPlane).dot(nPlane);
      
      dist = fabs(dRDotN);
      secTheta = 1./p3_[cInd].dot(nPlane)*p3_[cInd].mag();
      isIncoming = (p3_[cInd].dot(nPlane))*dRDotN < 0.;
      result = OK;
    }
    break;
//   case CONE_DT:
//     break;
//   case CYLINDER_DT:
//     break;
  case PATHL_DT:
    {
      double curS = fabs(path_[cInd]);
      dist = pars[PATHL_P] - curS;
      secTheta = 1.;
      isIncoming = pars[PATHL_P] > 0 ? true : false;
      result = OK;
    }
    break;
  case POINT_PCA_DT:
    {
      Point pDest(pars[0], pars[1], pars[2]);
      dist = (r3_[cInd] - pDest).mag()+ 1e-24;//add a small number to avoid 1/0
      secTheta = (r3_[cInd] - pDest).dot(p3_[cInd])/(dist*p3_[cInd].mag());
      isIncoming = secTheta < 0;
      result = OK;
    }
    break;
  case LINE_PCA_DT:
    {
      Point rLine(pars[0], pars[1], pars[2]);
      Vector dLine(pars[3], pars[4], pars[5]);
      dLine = (dLine - rLine);
      dLine /= dLine.mag();

      Vector dR = r3_[cInd] - rLine;
      Vector dRPerp = dR - dLine*(dR.dot(dLine));
      dist = dRPerp.mag() + 1e-24;//add a small number to avoid 1/0
      secTheta = dRPerp.dot(p3_[cInd])/(dist*p3_[cInd].mag());
      //angle wrt line
      double cosAlpha = dLine.dot(p3_[cInd])/p3_[cInd].mag();
      secTheta *= fabs(1./sqrt(fabs(1.-cosAlpha*cosAlpha)+1e-96));
      isIncoming = secTheta < 0;
      result = OK;
    }
    break;
  default:
    {
      //some large number
      dist = 1e12;
      secTheta = 1e12;
      isIncoming = true;
      result = NOT_IMPLEMENTED;
    }
    break;
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


void SteppingHelixPropagator::getLocBGrad(int ind, double delta) const{
  int cInd = cIndex_(ind);
  //yuck
  Point r3[3];
  r3[0] = r3_[cInd] + reps_[cInd].lX*delta;
  r3[1] = r3_[cInd] + reps_[cInd].lY*delta;
  r3[2] = r3_[cInd] + reps_[cInd].lZ*delta;

  double bVal[3];
  double bVal0 = bf_[cInd].mag();
  for (int i = 0; i < 3; i++){
    bVal[i] = field_->inTesla(GlobalPoint(r3[i].x(), r3[i].y(), r3[i].z())).mag();
  }
  bfGradLoc_[cInd].set((bVal[0] -bVal0)/delta, (bVal[1] -bVal0)/delta, (bVal[2] -bVal0)/delta);
}

