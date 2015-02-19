/** \class SteppingHelixPropagator
 *  Propagator implementation using steps along a helix.
 *  Minimal geometry navigation.
 *  Material effects (multiple scattering and energy loss) are based on tuning
 *  to MC and (eventually) data. 
 *  Implementation file contents follow.
 *
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Fri Mar  3 16:01:24 CST 2006
//
//


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"

#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>
#include <typeinfo>



void SteppingHelixPropagator::initStateArraySHPSpecific(StateArray& svBuf, bool flagsOnly) const{
  for (int i = 0; i <= MAX_POINTS; i++){
    svBuf[i].isComplete = true;
    svBuf[i].isValid_ = true;
    svBuf[i].hasErrorPropagated_ = !noErrorPropagation_;
    if (!flagsOnly){
      svBuf[i].p3 = Vector(0,0,0);
      svBuf[i].r3 = Point(0,0,0);
      svBuf[i].bf = Vector(0,0,0);
      svBuf[i].bfGradLoc =  Vector(0,0,0);
      svBuf[i].covCurv = AlgebraicSymMatrix55();
      svBuf[i].matDCovCurv = AlgebraicSymMatrix55();
    }
  }
}

SteppingHelixPropagator::~SteppingHelixPropagator() {}


SteppingHelixPropagator::SteppingHelixPropagator() :
  Propagator(anyDirection)
{
  field_ = 0;
}

SteppingHelixPropagator::SteppingHelixPropagator(const MagneticField* field, 
						 PropagationDirection dir):
  Propagator(dir),
  unit55_(AlgebraicMatrixID())
{
  field_ = field;
  vbField_ = dynamic_cast<const VolumeBasedMagneticField*>(field_);
  debug_ = false;
  noMaterialMode_ = false;
  noErrorPropagation_ = false;
  applyRadX0Correction_ = true;
  useMagVolumes_ = true;
  useIsYokeFlag_ = true;
  useMatVolumes_ = true;
  useInTeslaFromMagField_ = false; //overrides behavior only if true
  returnTangentPlane_ = true;
  sendLogWarning_ = false;
  useTuningForL2Speed_ = false;
  defaultStep_ = 5.;

  ecShiftPos_ = 0;
  ecShiftNeg_ = 0;

}


std::pair<TrajectoryStateOnSurface, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const Plane& pDest) const {

  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(SteppingHelixStateInfo(ftsStart),svBuf,nPoints);

  StateInfo svCurrent; 
  propagate(svBuf[0], pDest, svCurrent);

  return TsosPP(svCurrent.getStateOnSurface(pDest), svCurrent.path());
}

std::pair<TrajectoryStateOnSurface, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const Cylinder& cDest) const {

  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(SteppingHelixStateInfo(ftsStart),svBuf,nPoints);

  StateInfo svCurrent;
  propagate(svBuf[0], cDest, svCurrent);

  return TsosPP(svCurrent.getStateOnSurface(cDest, returnTangentPlane_), svCurrent.path());
}


std::pair<FreeTrajectoryState, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const GlobalPoint& pDest) const {
  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(SteppingHelixStateInfo(ftsStart),svBuf,nPoints);

  StateInfo svCurrent;
  propagate(svBuf[0], pDest,svCurrent);

  FreeTrajectoryState ftsDest;
  svCurrent.getFreeState(ftsDest);

  return FtsPP(ftsDest, svCurrent.path());
}

std::pair<FreeTrajectoryState, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
					   const GlobalPoint& pDest1, const GlobalPoint& pDest2) const {

  if ((pDest1-pDest2).mag() < 1e-10){
    if (sendLogWarning_){
      edm::LogWarning("SteppingHelixPropagator")<<"Can't propagate: the points should be at a bigger distance"
						<<std::endl;
    }
    return FtsPP();
  }
  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(SteppingHelixStateInfo(ftsStart),svBuf,nPoints);
  
  StateInfo svCurrent;
  propagate(svBuf[0], pDest1, pDest2,svCurrent);

  FreeTrajectoryState ftsDest;
  svCurrent.getFreeState(ftsDest);

  return FtsPP(ftsDest, svCurrent.path());
}


std::pair<FreeTrajectoryState, double>  
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart,  
                                           const reco::BeamSpot& beamSpot) const {
  GlobalPoint pDest1(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
  GlobalPoint pDest2(pDest1.x() + beamSpot.dxdz()*1e3, 
		     pDest1.y() + beamSpot.dydz()*1e3,
		     pDest1.z() + 1e3);
  return propagateWithPath(ftsStart, pDest1, pDest2);
}

void
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const Surface& sDest,
				   SteppingHelixStateInfo& out) const {
  
  if (! sStart.isValid()){
    if (sendLogWarning_){
      edm::LogWarning("SteppingHelixPropagator")<<"Can't propagate: invalid input state"
						<<std::endl;
    }
    out=invalidState_;
    return;
  }

  const Plane* pDest = dynamic_cast<const Plane*>(&sDest);
  if (pDest != 0) {
    propagate(sStart, *pDest, out);
    return;
  }

  const Cylinder* cDest = dynamic_cast<const Cylinder*>(&sDest);
  if (cDest != 0) {
    propagate(sStart, *cDest, out);
    return;
  }
      
  throw PropagationException("The surface is neither Cylinder nor Plane");

}

void
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const Plane& pDest,
				   SteppingHelixStateInfo& out) const {
  
  if (! sStart.isValid()){
    if (sendLogWarning_){
      edm::LogWarning("SteppingHelixPropagator")<<"Can't propagate: invalid input state"
						<<std::endl;
    }    
    out = invalidState_; 
    return ;
  }
  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(sStart,svBuf,nPoints);
  
  Point rPlane(pDest.position().x(), pDest.position().y(), pDest.position().z());
  Vector nPlane(pDest.rotation().zx(), pDest.rotation().zy(), pDest.rotation().zz()); nPlane /= nPlane.mag();

  double pars[6] = { rPlane.x(), rPlane.y(), rPlane.z(),
		     nPlane.x(), nPlane.y(), nPlane.z() };

  propagate(svBuf,nPoints,PLANE_DT, pars);
  
  //(re)set it before leaving: dir =1 (-1) if path increased (decreased) and 0 if it didn't change
  //need to implement this somewhere else as a separate function
  double lDir = 0;
  if (sStart.path() < svBuf[cIndex_(nPoints-1)].path()) lDir = 1.;
  if (sStart.path() > svBuf[cIndex_(nPoints-1)].path()) lDir = -1.;
  svBuf[cIndex_(nPoints-1)].dir = lDir;

  out = svBuf[cIndex_(nPoints-1)];
  return;
}

void
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const Cylinder& cDest,
				   SteppingHelixStateInfo& out) const {
  
  if (! sStart.isValid()){
    if (sendLogWarning_){
      edm::LogWarning("SteppingHelixPropagator")<<"Can't propagate: invalid input state"
						<<std::endl;
    }
    out = invalidState_;
    return;
  }
  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(sStart,svBuf,nPoints);
  
  double pars[6] = {0,0,0,0,0,0};
  pars[RADIUS_P] = cDest.radius();

  
  propagate(svBuf,nPoints,RADIUS_DT, pars);
  
  //(re)set it before leaving: dir =1 (-1) if path increased (decreased) and 0 if it didn't change
  //need to implement this somewhere else as a separate function
  double lDir = 0;
  if (sStart.path() < svBuf[cIndex_(nPoints-1)].path()) lDir = 1.;
  if (sStart.path() > svBuf[cIndex_(nPoints-1)].path()) lDir = -1.;
  svBuf[cIndex_(nPoints-1)].dir = lDir;
  out= svBuf[cIndex_(nPoints-1)];
  return;
}

void
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const GlobalPoint& pDest,
				   SteppingHelixStateInfo& out) const {
  
  if (! sStart.isValid()){
    if (sendLogWarning_){
      edm::LogWarning("SteppingHelixPropagator")<<"Can't propagate: invalid input state"
						<<std::endl;
    }    
    out = invalidState_;
    return;
  }
  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(sStart,svBuf,nPoints);
  
  double pars[6] = {pDest.x(), pDest.y(), pDest.z(), 0, 0, 0};

  
  propagate(svBuf,nPoints,POINT_PCA_DT, pars);
  
  out = svBuf[cIndex_(nPoints-1)];
  return;
}

void
SteppingHelixPropagator::propagate(const SteppingHelixStateInfo& sStart, 
				   const GlobalPoint& pDest1, const GlobalPoint& pDest2,
				   SteppingHelixStateInfo& out) const {
  
  if ((pDest1-pDest2).mag() < 1e-10 || !sStart.isValid()){
    if (sendLogWarning_){
      if ((pDest1-pDest2).mag() < 1e-10)
	edm::LogWarning("SteppingHelixPropagator")<<"Can't propagate: points are too close"
						  <<std::endl;
      if (!sStart.isValid())
	edm::LogWarning("SteppingHelixPropagator")<<"Can't propagate: invalid input state"
						  <<std::endl;
    }
    out = invalidState_;
    return;
  }
  StateArray svBuf; initStateArraySHPSpecific(svBuf, true);
  int nPoints = 0;
  setIState(sStart,svBuf,nPoints);
  
  double pars[6] = {pDest1.x(), pDest1.y(), pDest1.z(),
		    pDest2.x(), pDest2.y(), pDest2.z()};
  
  propagate(svBuf,nPoints,LINE_PCA_DT, pars);
  
  out = svBuf[cIndex_(nPoints-1)];
  return;
}

void SteppingHelixPropagator::setIState(const SteppingHelixStateInfo& sStart,
					StateArray& svBuf, int& nPoints) const {
  nPoints = 0;
  svBuf[cIndex_(nPoints)] = sStart; //do this anyways to have a fresh start
  if (sStart.isComplete ) {
    svBuf[cIndex_(nPoints)] = sStart;
    nPoints++;
  } else {
    loadState(svBuf[cIndex_(nPoints)], sStart.p3, sStart.r3, sStart.q,
	      propagationDirection(), sStart.covCurv);
    nPoints++;
  }
  svBuf[cIndex_(0)].hasErrorPropagated_ = sStart.hasErrorPropagated_ & !noErrorPropagation_;
}

SteppingHelixPropagator::Result 
SteppingHelixPropagator::propagate(StateArray& svBuf, int& nPoints,
				   SteppingHelixPropagator::DestType type, 
				   const double pars[6], double epsilon)  const{

  static const std::string metname = "SteppingHelixPropagator";
  StateInfo* svCurrent = &svBuf[cIndex_(nPoints-1)];

  //check if it's going to work at all
  double tanDist = 0;
  double dist = 0;
  PropagationDirection refDirection = anyDirection;
  Result result = refToDest(type, (*svCurrent), pars, dist, tanDist, refDirection);

  if (result != SteppingHelixStateInfo::OK || fabs(dist) > 1e6){
    svCurrent->status_ = result;
    if (fabs(dist) > 1e6) svCurrent->status_ = SteppingHelixStateInfo::INACC;
    svCurrent->isValid_ = false;
    svCurrent->field = field_;
    if (sendLogWarning_){
      edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<" Failed after first refToDest check with status "
			      <<SteppingHelixStateInfo::ResultName[result]
			      <<std::endl;
    }
    return result;
  }

  result = SteppingHelixStateInfo::UNDEFINED;
  bool makeNextStep = true;
  double dStep = defaultStep_;
  PropagationDirection dir,oldDir;
  dir = propagationDirection(); 
  oldDir = dir;
  int nOsc = 0;

  double distMag = 1e12;
  double tanDistMag = 1e12;

  double distMat = 1e12;
  double tanDistMat = 1e12;

  double tanDistNextCheck = -0.1;//just need a negative start val
  double tanDistMagNextCheck = -0.1;
  double tanDistMatNextCheck = -0.1;
  double oldDStep = 0;
  PropagationDirection oldRefDirection = propagationDirection();

  Result resultToMat = SteppingHelixStateInfo::UNDEFINED;
  Result resultToMag = SteppingHelixStateInfo::UNDEFINED;

  bool isFirstStep = true;
  bool expectNewMagVolume = false;

  int loopCount = 0;
  while (makeNextStep){
    dStep = defaultStep_;
    svCurrent = &svBuf[cIndex_(nPoints-1)];
    double curZ = svCurrent->r3.z();
    double curR = svCurrent->r3.perp();
    if ( fabs(curZ) < 440 && curR < 260) dStep = defaultStep_*2;

    //more such ifs might be scattered around
    //even though dStep is large, it will still make stops at each volume boundary
    if (useTuningForL2Speed_){
      dStep = 100.;
      if (! useIsYokeFlag_ && fabs(curZ) < 667 && curR > 380 && curR < 850){
	dStep = 5*(1+0.2*svCurrent->p3.mag());
      }
    }

    //    refDirection = propagationDirection();

    tanDistNextCheck -= oldDStep;
    tanDistMagNextCheck -= oldDStep;
    tanDistMatNextCheck -= oldDStep;
    
    if (tanDistNextCheck < 0){
      //use pre-computed values if it's the first step
      if (! isFirstStep) refToDest(type, (*svCurrent), pars, dist, tanDist, refDirection);
      // constrain allowed path for a tangential approach
      if (fabs(tanDist) > 4.*(fabs(dist)) ) tanDist *= tanDist == 0 ? 0 :fabs(dist/tanDist*4.);

      tanDistNextCheck = fabs(tanDist)*0.5 - 0.5; //need a better guess (to-do)
      //reasonable limit
      if (tanDistNextCheck >  defaultStep_*20. ) tanDistNextCheck = defaultStep_*20.;
      oldRefDirection = refDirection;
    } else {
      tanDist  = tanDist > 0. ? tanDist - oldDStep : tanDist + oldDStep; 
      refDirection = oldRefDirection;
      if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Skipped refToDest: guess tanDist = "<<tanDist
				   <<" next check at "<<tanDistNextCheck<<std::endl;
    }
    //! define a fast-skip distance: should be the shortest of a possible step or distance
    double fastSkipDist = fabs(dist) > fabs(tanDist) ? tanDist : dist;

    if (propagationDirection() == anyDirection){
      dir = refDirection;
    } else {
      dir = propagationDirection();
      if (fabs(tanDist)<0.1 && refDirection != dir ){
	//how did it get here?	nOsc++;
	dir = refDirection;
	if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"NOTE: overstepped last time: switch direction (can do it if within 1 mm)"<<std::endl;
      }
    }    

    if (useMagVolumes_ && ! (fabs(dist) < fabs(epsilon))){//need to know the general direction
      if (tanDistMagNextCheck < 0){
	resultToMag = refToMagVolume((*svCurrent), dir, distMag, tanDistMag, fabs(fastSkipDist), expectNewMagVolume, fabs(tanDist));
	// constrain allowed path for a tangential approach
	if (fabs(tanDistMag) > 4.*(fabs(distMag)) ) tanDistMag *= tanDistMag == 0 ? 0 : fabs(distMag/tanDistMag*4.);

	tanDistMagNextCheck = fabs(tanDistMag)*0.5-0.5; //need a better guess (to-do)
	//reasonable limit; "turn off" checking if bounds are further than the destination
	if (tanDistMagNextCheck >  defaultStep_*20. 
	    || fabs(dist) < fabs(distMag)
	    || resultToMag ==SteppingHelixStateInfo::INACC) 
	  tanDistMagNextCheck  = defaultStep_*20 > fabs(fastSkipDist) ? fabs(fastSkipDist) : defaultStep_*20;	
	if (resultToMag != SteppingHelixStateInfo::INACC 
	    && resultToMag != SteppingHelixStateInfo::OK) tanDistMagNextCheck = -1;
      } else {
	//	resultToMag = SteppingHelixStateInfo::OK;
	tanDistMag  = tanDistMag > 0. ? tanDistMag - oldDStep : tanDistMag + oldDStep; 
	if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Skipped refToMag: guess tanDistMag = "<<tanDistMag
				     <<" next check at "<<tanDistMagNextCheck;
      }
    }

    if (useMatVolumes_ && ! (fabs(dist) < fabs(epsilon))){//need to know the general direction
      if (tanDistMatNextCheck < 0){
	resultToMat = refToMatVolume((*svCurrent), dir, distMat, tanDistMat, fabs(fastSkipDist));
	// constrain allowed path for a tangential approach
	if (fabs(tanDistMat) > 4.*(fabs(distMat)) ) tanDistMat *= tanDistMat == 0 ? 0 : fabs(distMat/tanDistMat*4.);

	tanDistMatNextCheck = fabs(tanDistMat)*0.5-0.5; //need a better guess (to-do)
	//reasonable limit; "turn off" checking if bounds are further than the destination
	if (tanDistMatNextCheck >  defaultStep_*20. 
	    || fabs(dist) < fabs(distMat)
	    || resultToMat ==SteppingHelixStateInfo::INACC ) 
	  tanDistMatNextCheck = defaultStep_*20 > fabs(fastSkipDist) ? fabs(fastSkipDist) : defaultStep_*20;
	if (resultToMat != SteppingHelixStateInfo::INACC 
	    && resultToMat != SteppingHelixStateInfo::OK) tanDistMatNextCheck = -1;
      } else {
	//	resultToMat = SteppingHelixStateInfo::OK;
	tanDistMat  = tanDistMat > 0. ? tanDistMat - oldDStep : tanDistMat + oldDStep; 
	if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Skipped refToMat: guess tanDistMat = "<<tanDistMat
				     <<" next check at "<<tanDistMatNextCheck;
      }
    }

    double rDotP = svCurrent->r3.dot(svCurrent->p3);
    if ((fabs(curZ) > 1.5e3 || curR >800.) 
	&& ((dir == alongMomentum && rDotP > 0) 
	    || (dir == oppositeToMomentum && rDotP < 0) )
	){
      dStep = fabs(tanDist) -1e-12;
    }
    double tanDistMin = fabs(tanDist);
    if (tanDistMin > fabs(tanDistMag)+0.05 && 
	(resultToMag == SteppingHelixStateInfo::OK || resultToMag == SteppingHelixStateInfo::WRONG_VOLUME)){
      tanDistMin = fabs(tanDistMag)+0.05;     //try to step into the next volume
      expectNewMagVolume = true;
    } else expectNewMagVolume = false;

    if (tanDistMin > fabs(tanDistMat)+0.05 && resultToMat == SteppingHelixStateInfo::OK){
      tanDistMin = fabs(tanDistMat)+0.05;     //try to step into the next volume
      if (expectNewMagVolume) expectNewMagVolume = false;
    }

    if (tanDistMin*fabs(svCurrent->dEdx) > 0.5*svCurrent->p3.mag()){
      tanDistMin = 0.5*svCurrent->p3.mag()/fabs(svCurrent->dEdx);
      if (expectNewMagVolume) expectNewMagVolume = false;
    }



    double tanDistMinLazy = fabs(tanDistMin);
    if ((type == POINT_PCA_DT || type == LINE_PCA_DT)
	&& fabs(tanDist) < 2.*fabs(tanDistMin) ){
      //being lazy here; the best is to take into account the curvature
      tanDistMinLazy = fabs(tanDistMin)*0.5;
    }
 
    if (fabs(tanDistMinLazy) < dStep){
      dStep = fabs(tanDistMinLazy); 
    }

    //keep this path length for the next step
    oldDStep = dStep;

    if (dStep > 1e-10 && ! (fabs(dist) < fabs(epsilon))){
      StateInfo* svNext = &svBuf[cIndex_(nPoints)];
      makeAtomStep((*svCurrent), (*svNext), dStep, dir, HEL_AS_F);
//       if (useMatVolumes_ && expectNewMagVolume 
// 	  && svCurrent->magVol == svNext->magVol){
// 	double tmpDist=0;
// 	double tmpDistMag = 0;
// 	if (refToMagVolume((*svNext), dir, tmpDist, tmpDistMag, fabs(dist)) != SteppingHelixStateInfo::OK){
// 	//the point appears to be outside, but findVolume claims the opposite
// 	  dStep += 0.05*fabs(tanDistMag/distMag); oldDStep = dStep; //do it again with a bigger step
// 	  if (debug_) LogTrace(metname)
// 	    <<"Failed to get into new mag volume: will try with new bigger step "<<dStep<<std::endl;
// 	  makeAtomStep((*svCurrent), (*svNext), dStep, dir, HEL_AS_F);	  
// 	}
//       }
      nPoints++;    svCurrent = &svBuf[cIndex_(nPoints-1)];
      if (oldDir != dir){
	nOsc++;
	tanDistNextCheck = -1;//check dist after osc
	tanDistMagNextCheck = -1;
	tanDistMatNextCheck = -1;
      }
      oldDir = dir;
    }

    if (nOsc>1 && fabs(dStep)>epsilon){
      if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Ooops"<<std::endl;
    }

    if (fabs(dist) < fabs(epsilon)  ) result = SteppingHelixStateInfo::OK;

    if ((type == POINT_PCA_DT || type == LINE_PCA_DT )
	&& fabs(dStep) < fabs(epsilon)  ){
      //now check if it's not a branch point (peek ahead at 1 cm)
      double nextDist = 0;
      double nextTanDist = 0;
      PropagationDirection nextRefDirection = anyDirection;
      StateInfo* svNext = &svBuf[cIndex_(nPoints)];
      makeAtomStep((*svCurrent), (*svNext), 1., dir, HEL_AS_F);
      nPoints++;     svCurrent = &svBuf[cIndex_(nPoints-1)];
      refToDest(type, (*svCurrent), pars, nextDist, nextTanDist, nextRefDirection);
      if ( fabs(nextDist) > fabs(dist)){
	nPoints--;      svCurrent = &svBuf[cIndex_(nPoints-1)];
	result = SteppingHelixStateInfo::OK;
	if (debug_){
	  LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Found real local minimum in PCA"<<std::endl;
	}
      } else {
	//keep this trial point and continue
	dStep = defaultStep_;
	if (debug_){
	  LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Found branch point in PCA"<<std::endl;
	}
      }
    }

    if (nPoints > MAX_STEPS*1./defaultStep_  || loopCount > MAX_STEPS*100
	|| nOsc > 6 ) result = SteppingHelixStateInfo::FAULT;

    if (svCurrent->p3.mag() < 0.1 ) result = SteppingHelixStateInfo::RANGEOUT;

    curZ = svCurrent->r3.z();
    curR = svCurrent->r3.perp();
    if ( curR > 20000 || fabs(curZ) > 20000 ) result = SteppingHelixStateInfo::INACC;

    makeNextStep = result == SteppingHelixStateInfo::UNDEFINED;
    svCurrent->status_ = result;
    svCurrent->isValid_ = (result == SteppingHelixStateInfo::OK || makeNextStep );
    svCurrent->field = field_;

    isFirstStep = false;
    loopCount++;
  }

  if (sendLogWarning_ && result != SteppingHelixStateInfo::OK){
    edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<" Propagation failed with status "
			    <<SteppingHelixStateInfo::ResultName[result]
			    <<std::endl;
    if (result == SteppingHelixStateInfo::RANGEOUT)
      edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<" Momentum at last point is too low (<0.1) p_last = "
			      <<svCurrent->p3.mag()
			      <<std::endl;
    if (result == SteppingHelixStateInfo::INACC)
      edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<" Went too far: the last point is at "<<svCurrent->r3
			      <<std::endl;
    if (result == SteppingHelixStateInfo::FAULT && nOsc > 6)
      edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<" Infinite loop condidtion detected: going in cycles. Break after 6 cycles"
			      <<std::endl;
    if (result == SteppingHelixStateInfo::FAULT && nPoints > MAX_STEPS*1./defaultStep_)
      edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<" Tired to go farther. Made too many steps: more than "
			      <<MAX_STEPS*1./defaultStep_
			      <<std::endl;
    
  }

  if (debug_){
    switch (type) {
    case RADIUS_DT:
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"going to radius "<<pars[RADIUS_P]<<std::endl;
      break;
    case Z_DT:
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"going to z "<<pars[Z_P]<<std::endl;
      break;
    case PATHL_DT:
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"going to pathL "<<pars[PATHL_P]<<std::endl;
      break;
    case PLANE_DT:
      {
	Point rPlane(pars[0], pars[1], pars[2]);
	Vector nPlane(pars[3], pars[4], pars[5]);
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"going to plane r0:"<<rPlane<<" n:"<<nPlane<<std::endl;
      }
      break;
    case POINT_PCA_DT:
      {
	Point rDest(pars[0], pars[1], pars[2]);
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"going to PCA to point "<<rDest<<std::endl;
      }
      break;
    case LINE_PCA_DT:
      {
	Point rDest1(pars[0], pars[1], pars[2]);
	Point rDest2(pars[3], pars[4], pars[5]);
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"going to PCA to line "<<rDest1<<" - "<<rDest2<<std::endl;
      }
      break;
    default:
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"going to NOT IMPLEMENTED"<<std::endl;
      break;
    }
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Made "<<nPoints-1<<" steps and stopped at(cur step) "<<svCurrent->r3<<" nOsc "<<nOsc<<std::endl;
  }
  
  return result;
}
  
void SteppingHelixPropagator::loadState(SteppingHelixPropagator::StateInfo& svCurrent, 
					const SteppingHelixPropagator::Vector& p3, 
					const SteppingHelixPropagator::Point& r3, int charge,
					PropagationDirection dir,
					const AlgebraicSymMatrix55& covCurv) const{
  static const std::string metname = "SteppingHelixPropagator";

  svCurrent.q = charge;
  svCurrent.p3 = p3;
  svCurrent.r3 = r3;
  svCurrent.dir = dir == alongMomentum ? 1.: -1.;

  svCurrent.path_ = 0; // this could've held the initial path
  svCurrent.radPath_ = 0;

  GlobalPoint gPointNorZ(svCurrent.r3.x(), svCurrent.r3.y(), svCurrent.r3.z());

  float gpmag = gPointNorZ.mag2();
  float pmag2 = p3.mag2();
  if (gpmag > 1e20f ) {
    LogTrace(metname)<<"Initial point is too far";
    svCurrent.isValid_ = false;
    return;
  }
  if (pmag2 < 1e-18f ) {
    LogTrace(metname)<<"Initial momentum is too low";
    svCurrent.isValid_ = false;
    return;
  }
  if (! (gpmag == gpmag) ) {
    LogTrace(metname)<<"Initial point is a nan";
    edm::LogWarning("SteppingHelixPropagatorNAN")<<"Initial point is a nan";
    svCurrent.isValid_ = false;
    return;
  }
  if (! (pmag2 == pmag2) ) {
    LogTrace(metname)<<"Initial momentum is a nan";
    edm::LogWarning("SteppingHelixPropagatorNAN")<<"Initial momentum is a nan";
    svCurrent.isValid_ = false;
    return;
  }

  GlobalVector bf(0,0,0);
  // = field_->inTesla(gPoint);
  if (useMagVolumes_){
    if (vbField_ ){
      svCurrent.magVol = vbField_->findVolume(gPointNorZ);
      if (useIsYokeFlag_){
	double curRad = svCurrent.r3.perp();
	if (curRad > 380 && curRad < 850 && fabs(svCurrent.r3.z()) < 667){
	  svCurrent.isYokeVol = isYokeVolume(svCurrent.magVol);
	} else {
	  svCurrent.isYokeVol = false;
	}
      }
    } else {
      edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Failed to cast into VolumeBasedMagneticField: fall back to the default behavior"<<std::endl;
      svCurrent.magVol = 0;
    }
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Got volume at "<<svCurrent.magVol<<std::endl;
    }
  }
  
  if (useMagVolumes_ && svCurrent.magVol != 0 && ! useInTeslaFromMagField_){
    bf = svCurrent.magVol->inTesla(gPointNorZ);
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Loaded bfield float: "<<bf
		       <<" at global float "<< gPointNorZ<<" double "<< svCurrent.r3<<std::endl;
      LocalPoint lPoint(svCurrent.magVol->toLocal(gPointNorZ));
      LocalVector lbf = svCurrent.magVol->fieldInTesla(lPoint);
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"\t cf in local locF: "<<lbf<<" at "<<lPoint<<std::endl;
    }
    svCurrent.bf.set(bf.x(), bf.y(), bf.z());
  } else {
    GlobalPoint gPoint(r3.x(), r3.y(), r3.z());
    bf = field_->inTesla(gPoint);
    svCurrent.bf.set(bf.x(), bf.y(), bf.z());
  }
  if (svCurrent.bf.mag2() < 1e-32) svCurrent.bf.set(0., 0., 1e-16);
  if (debug_){
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific
		     <<"Loaded bfield double: "<<svCurrent.bf<<"  from float: "<<bf
		     <<" at float "<< gPointNorZ<<" double "<< svCurrent.r3<<std::endl;
  }



  double dEdXPrime = 0;
  double dEdx = 0;
  double radX0 = 0;
  MatBounds rzLims;
  dEdx = getDeDx(svCurrent, dEdXPrime, radX0, rzLims);
  svCurrent.dEdx = dEdx;    svCurrent.dEdXPrime = dEdXPrime;
  svCurrent.radX0 = radX0;
  svCurrent.rzLims = rzLims;

  svCurrent.covCurv =covCurv;

  svCurrent.isComplete = true;
  svCurrent.isValid_ = true;

  if (debug_){
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Loaded at  path: "<<svCurrent.path()<<" radPath: "<<svCurrent.radPath()
		     <<" p3 "<<" pt: "<<svCurrent.p3.perp()<<" phi: "<<svCurrent.p3.phi()
		     <<" eta: "<<svCurrent.p3.eta()
		     <<" "<<svCurrent.p3
		     <<" r3: "<<svCurrent.r3
		     <<" bField: "<<svCurrent.bf.mag()
		     <<std::endl;
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Input Covariance in Curvilinear RF "<<covCurv<<std::endl;
  }
}

void SteppingHelixPropagator::getNextState(const SteppingHelixPropagator::StateInfo& svPrevious, 
					   SteppingHelixPropagator::StateInfo& svNext,
					   double dP, const SteppingHelixPropagator::Vector& tau,
					   const SteppingHelixPropagator::Vector& drVec, double dS, double dX0,
					   const AlgebraicMatrix55& dCovCurvTransform) const{
  static const std::string metname = "SteppingHelixPropagator";
  svNext.q = svPrevious.q;
  svNext.dir = dS > 0.0 ? 1.: -1.; 
  svNext.p3 = tau;  svNext.p3*=(svPrevious.p3.mag() - svNext.dir*fabs(dP));

  svNext.r3 = svPrevious.r3; svNext.r3 += drVec;

  svNext.path_ = svPrevious.path() + dS;
  svNext.radPath_ = svPrevious.radPath() + dX0;

  GlobalPoint gPointNorZ(svNext.r3.x(), svNext.r3.y(), svNext.r3.z());

  GlobalVector bf(0,0,0); 

  if (useMagVolumes_){
    if (vbField_ != 0){
      svNext.magVol = vbField_->findVolume(gPointNorZ);
      if (useIsYokeFlag_){
	double curRad = svNext.r3.perp();
	if (curRad > 380 && curRad < 850 && fabs(svNext.r3.z()) < 667){
	  svNext.isYokeVol = isYokeVolume(svNext.magVol);
	} else {
	  svNext.isYokeVol = false;
	}
      }
    } else {
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Failed to cast into VolumeBasedMagneticField"<<std::endl;
      svNext.magVol = 0;
    }
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Got volume at "<<svNext.magVol<<std::endl;
    }
  }

  if (useMagVolumes_ && svNext.magVol != 0 && ! useInTeslaFromMagField_){
    bf = svNext.magVol->inTesla(gPointNorZ);
    svNext.bf.set(bf.x(), bf.y(), bf.z());
  } else {
    GlobalPoint gPoint(svNext.r3.x(), svNext.r3.y(), svNext.r3.z());
    bf = field_->inTesla(gPoint);
    svNext.bf.set(bf.x(), bf.y(), bf.z());
  }
  if (svNext.bf.mag2() < 1e-32) svNext.bf.set(0., 0., 1e-16);
  
  
  double dEdXPrime = 0;
  double dEdx = 0;
  double radX0 = 0;
  MatBounds rzLims;
  dEdx = getDeDx(svNext, dEdXPrime, radX0, rzLims);
  svNext.dEdx = dEdx;    svNext.dEdXPrime = dEdXPrime;
  svNext.radX0 = radX0;
  svNext.rzLims = rzLims;

  //update Emat only if it's valid
  svNext.hasErrorPropagated_ = svPrevious.hasErrorPropagated_;
  if (svPrevious.hasErrorPropagated_){
    {
      AlgebraicMatrix55 tmp = dCovCurvTransform*svPrevious.covCurv;
      ROOT::Math::AssignSym::Evaluate(svNext.covCurv, tmp*ROOT::Math::Transpose(dCovCurvTransform));
      
      svNext.covCurv += svPrevious.matDCovCurv;
    }
  } else {
    //could skip dragging along the unprop. cov later .. now
    // svNext.cov = svPrevious.cov;
  }

  if (debug_){
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Now at  path: "<<svNext.path()<<" radPath: "<<svNext.radPath()
		     <<" p3 "<<" pt: "<<svNext.p3.perp()<<" phi: "<<svNext.p3.phi()
		     <<" eta: "<<svNext.p3.eta()
		     <<" "<<svNext.p3
		     <<" r3: "<<svNext.r3
		     <<" dPhi: "<<acos(svNext.p3.unit().dot(svPrevious.p3.unit()))
		     <<" bField: "<<svNext.bf.mag()
		     <<" dedx: "<<svNext.dEdx
		     <<std::endl;
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"New Covariance "<<svNext.covCurv<<std::endl;
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Transf by dCovTransform "<<dCovCurvTransform<<std::endl;
  }
}

void SteppingHelixPropagator::setRep(SteppingHelixPropagator::Basis& rep, 
				     const SteppingHelixPropagator::Vector& tau) const{
  Vector zRep(0., 0., 1.);
  rep.lX = tau;
  rep.lY = zRep.cross(tau); rep.lY *= 1./tau.perp();
  rep.lZ = rep.lX.cross(rep.lY);
}

bool SteppingHelixPropagator::makeAtomStep(SteppingHelixPropagator::StateInfo& svCurrent,
					   SteppingHelixPropagator::StateInfo& svNext,
					   double dS, 
					   PropagationDirection dir, 
					   SteppingHelixPropagator::Fancy fancy) const{
  static const std::string metname = "SteppingHelixPropagator";
  if (debug_){
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Make atom step "<<svCurrent.path()<<" with step "<<dS<<" in direction "<<dir<<std::endl;
  }

  AlgebraicMatrix55 dCCurvTransform(unit55_);

  double dP = 0;
  double curP = svCurrent.p3.mag();
  Vector tau = svCurrent.p3; tau *= 1./curP;
  Vector tauNext(tau);
  Vector drVec(0,0,0);

  dS = dir == alongMomentum ? fabs(dS) : -fabs(dS);


  double radX0 = 1e24;

  switch (fancy){
  case HEL_AS_F:
  case HEL_ALL_F:{
    double p0 = curP;// see above = svCurrent.p3.mag();
    double p0Inv = 1./p0;
    double b0 = svCurrent.bf.mag();

    //get to the mid-point first
    double phi = (2.99792458e-3*svCurrent.q*b0*p0Inv)*dS/2.;
    bool phiSmall = fabs(phi) < 1e-4;

    double cosPhi = 0;
    double sinPhi = 0;

    double oneLessCosPhi=0;
    double oneLessCosPhiOPhi=0;
    double sinPhiOPhi=0;
    double phiLessSinPhiOPhi=0;

    if (phiSmall){
      double phi2 = phi*phi;
      double phi3 = phi2*phi;
      double phi4 = phi3*phi;
      sinPhi = phi - phi3/6. + phi4*phi/120.;
      cosPhi = 1. -phi2/2. + phi4/24.;
      oneLessCosPhi = phi2/2. - phi4/24. + phi2*phi4/720.; // 0.5*phi*phi;//*(1.- phi*phi/12.);
      oneLessCosPhiOPhi = 0.5*phi - phi3/24. + phi2*phi3/720.;//*(1.- phi*phi/12.);
      sinPhiOPhi = 1. - phi*phi/6. + phi4/120.;
      phiLessSinPhiOPhi = phi*phi/6. - phi4/120. + phi4*phi2/5040.;//*(1. - phi*phi/20.);
    } else {
      cosPhi = cos(phi);
      sinPhi = sin(phi);
      oneLessCosPhi = 1.-cosPhi;
      oneLessCosPhiOPhi = oneLessCosPhi/phi;
      sinPhiOPhi = sinPhi/phi;
      phiLessSinPhiOPhi = 1 - sinPhiOPhi;
    }

    Vector bHat = svCurrent.bf; bHat *= 1./b0; //bHat.mag();
    //    bool bAlongZ = fabs(bHat.z()) > 0.9999;

    Vector btVec(bHat.cross(tau)); // for balong z btVec.z()==0
    double tauB =  tau.dot(bHat);
    Vector bbtVec(bHat*tauB - tau); // (-tau.x(), -tau.y(), 0)

    //don't need it here    tauNext = tau + bbtVec*oneLessCosPhi - btVec*sinPhi;
    drVec = bbtVec*phiLessSinPhiOPhi; drVec -= btVec*oneLessCosPhiOPhi; drVec += tau; 
    drVec *= dS/2.;

    double dEdx = svCurrent.dEdx;
    double dEdXPrime = svCurrent.dEdXPrime;
    radX0 = svCurrent.radX0;
    dP = dEdx*dS;

    //improve with above values:
    drVec += svCurrent.r3;
    GlobalVector bfGV(0,0,0);
    Vector bf(0,0,0); 
    if (useMagVolumes_ && svCurrent.magVol != 0 && ! useInTeslaFromMagField_){
      bfGV = svCurrent.magVol->inTesla(GlobalPoint(drVec.x(), drVec.y(), drVec.z()));
      bf.set(bfGV.x(), bfGV.y(), bfGV.z());
    } else {
      bfGV = field_->inTesla(GlobalPoint(drVec.x(), drVec.y(), drVec.z()));
      bf.set(bfGV.x(), bfGV.y(), bfGV.z());
    }
    double b0Init = b0;
    b0 = bf.mag();
    if (b0 < 1e-16) {
      b0 = 1e-16;
      bf.set(0., 0., 1e-16);
    }
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Improved b "<<b0
		       <<" at r3 "<<drVec<<std::endl;
    }

    if (fabs((b0-b0Init)*dS) > 1){
      //missed the mag volume boundary?
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Large bf*dS change "<<fabs((b0-svCurrent.bf.mag())*dS)
			 <<" --> recalc dedx"<<std::endl;
      }
      svNext.r3 = drVec;
      svNext.bf = bf;
      svNext.p3 = svCurrent.p3;
      svNext.isYokeVol = svCurrent.isYokeVol;
      svNext.magVol = svCurrent.magVol;
      MatBounds rzTmp;
      dEdx = getDeDx(svNext, dEdXPrime, radX0, rzTmp);
      dP = dEdx*dS;      
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"New dEdX= "<<dEdx
			 <<" dP= "<<dP
			 <<" for p0 "<<p0<<std::endl;
      }
    }
    //p0 is mid-way and b0 from mid-point
    p0 += dP/2.; if (p0 < 1e-2) p0 = 1e-2;
    p0Inv = 1./p0;

    phi = (2.99792458e-3*svCurrent.q*b0*p0Inv)*dS;
    phiSmall = fabs(phi) < 1e-4;

    if (phiSmall){
      double phi2 = phi*phi;
      double phi3 = phi2*phi;
      double phi4 = phi3*phi;
      sinPhi = phi - phi3/6. + phi4*phi/120.;
      cosPhi = 1. -phi2/2. + phi4/24.;
      oneLessCosPhi = phi2/2. - phi4/24. + phi2*phi4/720.; // 0.5*phi*phi;//*(1.- phi*phi/12.);
      oneLessCosPhiOPhi = 0.5*phi - phi3/24. + phi2*phi3/720.;//*(1.- phi*phi/12.);
      sinPhiOPhi = 1. - phi*phi/6. + phi4/120.;
      phiLessSinPhiOPhi = phi*phi/6. - phi4/120. + phi4*phi2/5040.;//*(1. - phi*phi/20.);
    }else {
      cosPhi = cos(phi); 
      sinPhi = sin(phi);
      oneLessCosPhi = 1.-cosPhi;
      oneLessCosPhiOPhi = oneLessCosPhi/phi;
      sinPhiOPhi = sinPhi/phi;
      phiLessSinPhiOPhi = 1. - sinPhiOPhi;
    }

    bHat = bf; bHat *= 1./b0;// as above =1./bHat.mag();
    //    bAlongZ = fabs(bHat.z()) > 0.9999;
    btVec = bHat.cross(tau); // for b||z (-tau.y(), tau.x() ,0)
    tauB = tau.dot(bHat);
    bbtVec = bHat*tauB - tau; //bHat.cross(btVec); for b||z: (-tau.x(), -tau.y(), 0)

    tauNext = bbtVec*oneLessCosPhi; tauNext -= btVec*sinPhi;     tauNext += tau; //for b||z tauNext.z() == tau.z()
    double tauNextPerpInv = 1./tauNext.perp();
    drVec   = bbtVec*phiLessSinPhiOPhi; drVec -= btVec*oneLessCosPhiOPhi;  drVec += tau;
    drVec *= dS;
    
    
    if (svCurrent.hasErrorPropagated_){
      double theta02 = 0;
      double dX0 = fabs(dS)/radX0;

      if (applyRadX0Correction_){
	// this provides the integrand for theta^2
	// if summed up along the path, should result in 
	// theta_total^2 = Int_0^x0{ f(x)dX} = (13.6/p0)^2*x0*(1+0.036*ln(x0+1))
	// x0+1 above is to make the result infrared safe.
	double x0 = fabs(svCurrent.radPath());
	double alphaX0 = 13.6e-3*p0Inv; alphaX0 *= alphaX0;
	double betaX0 = 0.038;
	double logx0p1 = log(x0+1);
	theta02 = dX0*alphaX0*(1+betaX0*logx0p1)*(1 + betaX0*logx0p1 + 2.*betaX0*x0/(x0+1) );
      } else {
	theta02 = 196e-6* p0Inv * p0Inv * dX0; //14.e-3/p0*sqrt(fabs(dS)/radX0); // .. drop log term (this is non-additive)
      }
      
      double epsilonP0 = 1.+ dP/(p0-0.5*dP);
      //      double omegaP0 = -dP/(p0-0.5*dP) + dS*dEdXPrime;      
      //      double dsp = dS/(p0-0.5*dP); //use the initial p0 (not the mid-point) to keep the transport properly additive
      
      Vector tbtVec(bHat - tauB*tau); // for b||z tau.z()*(-tau.x(), -tau.y(), 1.-tau.z())
      
      {
	//Slightly modified copy of the curvilinear jacobian (don't use the original just because it's in float precision
	// and seems to have some assumptions about the field values
	// notation changes: p1--> tau, p2-->tauNext
	// theta --> phi
	//	Vector p1 = tau;
	//	Vector p2 = tauNext;
	Point xStart = svCurrent.r3;
	Vector dx = drVec;
	//GlobalVector h  = MagneticField::inInverseGeV(xStart);
	// Martijn: field is now given as parameter.. GlobalVector h  = globalParameters.magneticFieldInInverseGeV(xStart);

	//double qbp = fts.signedInverseMomentum();
	double qbp = svCurrent.q*p0Inv;
	//	double absS = dS;
  
	// calculate transport matrix
	// Origin: TRPRFN
	double t11 = tau.x(); double t12 = tau.y(); double t13 = tau.z();
	double t21 = tauNext.x(); double t22 = tauNext.y(); double t23 = tauNext.z();
	double cosl0 = tau.perp(); 
	//	double cosl1 = 1./tauNext.perp(); //not quite a cos .. it's a cosec--> change to cosecl1 below
	double cosecl1 = tauNextPerpInv;
	//AlgebraicMatrix a(5,5,1);
	// define average magnetic field and gradient 
	// at initial point - inlike TRPRFN
	Vector hn = bHat;
	//	double qp = -2.99792458e-3*b0;
	//   double q = -h.mag()*qbp;

	double q = -phi/dS; //qp*qbp; // -phi/dS
	//	double theta = -phi; 
	double sint = -sinPhi; double cost = cosPhi;
	double hn1 = hn.x(); double hn2 = hn.y(); double hn3 = hn.z();
	double dx1 = dx.x(); double dx2 = dx.y(); double dx3 = dx.z();
	//	double hnDt1 = hn1*t11 + hn2*t12 + hn3*t13;

	double gamma =  hn1*t21 + hn2*t22 + hn3*t23;
	double an1 =  hn2*t23 - hn3*t22;
	double an2 =  hn3*t21 - hn1*t23;
	double an3 =  hn1*t22 - hn2*t21;
	//	  double auInv = sqrt(1.- t13*t13); double au = auInv>0 ? 1./auInv : 1e24;
	double auInv = cosl0; double au = auInv>0 ? 1./auInv : 1e24;
	//	  double auInv = sqrt(t11*t11 + t12*t12); double au = auInv>0 ? 1./auInv : 1e24;
	double u11 = -au*t12; double u12 = au*t11;
	double v11 = -t13*u12; double v12 = t13*u11; double v13 = auInv;//t11*u12 - t12*u11;
	auInv = sqrt(1. - t23*t23); au = auInv>0 ? 1./auInv : 1e24;
	//	  auInv = sqrt(t21*t21 + t22*t22); au = auInv>0 ? 1./auInv : 1e24;
	double u21 = -au*t22; double u22 = au*t21;
	double v21 = -t23*u22; double v22 = t23*u21; double v23 = auInv;//t21*u22 - t22*u21;
	// now prepare the transport matrix
	// pp only needed in high-p case (WA)
	//   double pp = 1./qbp;
	////    double pp = fts.momentum().mag();
	// moved up (where -h.mag() is needed()
	//   double qp = q*pp;
	double anv =  -(hn1*u21 + hn2*u22          );
	double anu =   (hn1*v21 + hn2*v22 + hn3*v23); 
	double omcost = oneLessCosPhi; double tmsint = -phi*phiLessSinPhiOPhi;
	
	double hu1 =         - hn3*u12;
	double hu2 =  hn3*u11;
	double hu3 =  hn1*u12 - hn2*u11;
	
	double hv1 =  hn2*v13 - hn3*v12;
	double hv2 =  hn3*v11 - hn1*v13;
	double hv3 =  hn1*v12 - hn2*v11;
	
	//   1/p - doesn't change since |tau| = |tauNext| ... not. It changes now
	dCCurvTransform(0,0) = 1./(epsilonP0*epsilonP0)*(1. + dS*dEdXPrime);
	
	//   lambda
	
	dCCurvTransform(1,0) = phi*p0/svCurrent.q*cosecl1*
	  (sinPhi*bbtVec.z() - cosPhi*btVec.z());
	//was dCCurvTransform(1,0) = -qp*anv*(t21*dx1 + t22*dx2 + t23*dx3); //NOTE (SK) this was found to have an opposite sign
	//from independent re-calculation ... in fact the tauNext.dot.dR piece isnt reproduced 
	
	dCCurvTransform(1,1) = cost*(v11*v21 + v12*v22 + v13*v23) +
	  sint*(hv1*v21 + hv2*v22 + hv3*v23) +
	  omcost*(hn1*v11 + hn2*v12 + hn3*v13) * (hn1*v21 + hn2*v22 + hn3*v23) +
	  anv*(-sint*(v11*t21 + v12*t22 + v13*t23) +
	       omcost*(v11*an1 + v12*an2 + v13*an3) -
	       tmsint*gamma*(hn1*v11 + hn2*v12 + hn3*v13) );
	
	dCCurvTransform(1,2) = cost*(u11*v21 + u12*v22          ) +
	  sint*(hu1*v21 + hu2*v22 + hu3*v23) +
	  omcost*(hn1*u11 + hn2*u12          ) * (hn1*v21 + hn2*v22 + hn3*v23) +
	  anv*(-sint*(u11*t21 + u12*t22          ) +
	       omcost*(u11*an1 + u12*an2          ) -
	       tmsint*gamma*(hn1*u11 + hn2*u12          ) );
	dCCurvTransform(1,2) *= cosl0;
	
	// Commented out in part for reproducibility purposes: these terms are zero in cart->curv 
	//	dCCurvTransform(1,3) = -q*anv*(u11*t21 + u12*t22          ); //don't show up in cartesian setup-->curv
	//why would lambdaNext depend explicitely on initial position ? any arbitrary init point can be chosen not 
	// affecting the final state's momentum direction ... is this the field gradient in curvilinear coord?
	//	dCCurvTransform(1,4) = -q*anv*(v11*t21 + v12*t22 + v13*t23); //don't show up in cartesian setup-->curv
	
	//   phi
	
	dCCurvTransform(2,0) = - phi*p0/svCurrent.q*cosecl1*cosecl1*
	  (oneLessCosPhi*bHat.z()*btVec.mag2() + sinPhi*btVec.z() + cosPhi*tbtVec.z()) ;
	//was 	dCCurvTransform(2,0) = -qp*anu*(t21*dx1 + t22*dx2 + t23*dx3)*cosecl1;
	
	dCCurvTransform(2,1) = cost*(v11*u21 + v12*u22          ) +
	  sint*(hv1*u21 + hv2*u22          ) +
	  omcost*(hn1*v11 + hn2*v12 + hn3*v13) *
	  (hn1*u21 + hn2*u22          ) +
	  anu*(-sint*(v11*t21 + v12*t22 + v13*t23) +
	       omcost*(v11*an1 + v12*an2 + v13*an3) -
	       tmsint*gamma*(hn1*v11 + hn2*v12 + hn3*v13) );
	dCCurvTransform(2,1) *= cosecl1;
	
	dCCurvTransform(2,2) = cost*(u11*u21 + u12*u22          ) +
	  sint*(hu1*u21 + hu2*u22          ) +
	  omcost*(hn1*u11 + hn2*u12          ) *
	  (hn1*u21 + hn2*u22          ) +
	  anu*(-sint*(u11*t21 + u12*t22          ) +
	       omcost*(u11*an1 + u12*an2          ) -
	       tmsint*gamma*(hn1*u11 + hn2*u12          ) );
	dCCurvTransform(2,2) *= cosecl1*cosl0;
	
	// Commented out in part for reproducibility purposes: these terms are zero in cart->curv 
	// dCCurvTransform(2,3) = -q*anu*(u11*t21 + u12*t22          )*cosecl1;
	//why would lambdaNext depend explicitely on initial position ? any arbitrary init point can be chosen not 
	// affecting the final state's momentum direction ... is this the field gradient in curvilinear coord?
	// dCCurvTransform(2,4) = -q*anu*(v11*t21 + v12*t22 + v13*t23)*cosecl1;
	
	//   yt
	
	double pp = 1./qbp;
	// (SK) these terms seem to consistently have a sign opp from private derivation
	dCCurvTransform(3,0) = - pp*(u21*dx1 + u22*dx2            ); //NB: modified from the original: changed the sign
	dCCurvTransform(4,0) = - pp*(v21*dx1 + v22*dx2 + v23*dx3);  
	
	
	dCCurvTransform(3,1) = (sint*(v11*u21 + v12*u22          ) +
				 omcost*(hv1*u21 + hv2*u22          ) +
				 tmsint*(hn1*u21 + hn2*u22          ) *
				 (hn1*v11 + hn2*v12 + hn3*v13))/q;
	
	dCCurvTransform(3,2) = (sint*(u11*u21 + u12*u22          ) +
				 omcost*(hu1*u21 + hu2*u22          ) +
				 tmsint*(hn1*u21 + hn2*u22          ) *
				 (hn1*u11 + hn2*u12          ))*cosl0/q;
	
	dCCurvTransform(3,3) = (u11*u21 + u12*u22          );
	
	dCCurvTransform(3,4) = (v11*u21 + v12*u22          );
	
	//   zt
	
	dCCurvTransform(4,1) = (sint*(v11*v21 + v12*v22 + v13*v23) +
				 omcost*(hv1*v21 + hv2*v22 + hv3*v23) +
				 tmsint*(hn1*v21 + hn2*v22 + hn3*v23) *
				 (hn1*v11 + hn2*v12 + hn3*v13))/q;
	
	dCCurvTransform(4,2) = (sint*(u11*v21 + u12*v22          ) +
				 omcost*(hu1*v21 + hu2*v22 + hu3*v23) +
				 tmsint*(hn1*v21 + hn2*v22 + hn3*v23) *
				 (hn1*u11 + hn2*u12          ))*cosl0/q;
	
	dCCurvTransform(4,3) = (u11*v21 + u12*v22          );
	
	dCCurvTransform(4,4) = (v11*v21 + v12*v22 + v13*v23);
	// end of TRPRFN
      }
    
      if (debug_){
	Basis rep; setRep(rep, tauNext);
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"rep X: "<<rep.lX<<" "<<rep.lX.mag()
			 <<"\t Y: "<<rep.lY<<" "<<rep.lY.mag()
			 <<"\t Z: "<<rep.lZ<<" "<<rep.lZ.mag();
      }
      //mind the sign of dS and dP (dS*dP < 0 allways)
      //covariance should grow no matter which direction you propagate
      //==> take abs values.
      //reset not needed: fill all below  svCurrent.matDCov *= 0.;
      double mulRR = theta02*dS*dS/3.;
      double mulRP = theta02*fabs(dS)*p0/2.;
      double mulPP = theta02*p0*p0;
      double losPP = dP*dP*1.6/fabs(dS)*(1.0 + p0*1e-3);
      //another guess .. makes sense for 1 cm steps 2./dS == 2 [cm] / dS [cm] at low pt
      //double it by 1TeV
      //not gaussian anyways
      // derived from the fact that sigma_p/eLoss ~ 0.08 after ~ 200 steps

      //curvilinear
      double sinThetaInv = tauNextPerpInv;
      double p0Mat = p0+ 0.5*dP; // FIXME change this to p0 after it's clear that there's agreement in everything else
      double p0Mat2 = p0Mat*p0Mat;
      // with 6x6 formulation
      svCurrent.matDCovCurv*=0;
      
      svCurrent.matDCovCurv(0,0) = losPP/(p0Mat2*p0Mat2);
      //      svCurrent.matDCovCurv(0,1) = 0;
      //      svCurrent.matDCovCurv(0,2) = 0;
      //      svCurrent.matDCovCurv(0,3) = 0;
      //      svCurrent.matDCovCurv(0,4) = 0;

      //      svCurrent.matDCovCurv(1,0) = 0;
      svCurrent.matDCovCurv(1,1) = mulPP/p0Mat2;
      //      svCurrent.matDCovCurv(1,2) = 0;
      //      svCurrent.matDCovCurv(1,3) = 0;
      svCurrent.matDCovCurv(1,4) = mulRP/p0Mat;

      //      svCurrent.matDCovCurv(2,0) = 0;
      //      svCurrent.matDCovCurv(2,1) = 0;
      svCurrent.matDCovCurv(2,2) = mulPP/p0Mat2*(sinThetaInv*sinThetaInv);
      svCurrent.matDCovCurv(2,3) = mulRP/p0Mat*sinThetaInv;
      //      svCurrent.matDCovCurv(2,4) = 0;

      //      svCurrent.matDCovCurv(3,0) = 0;
      //      svCurrent.matDCovCurv(3,1) = 0;
      svCurrent.matDCovCurv(3,2) = mulRP/p0Mat*sinThetaInv;
      //      svCurrent.matDCovCurv(3,0) = 0;
      svCurrent.matDCovCurv(3,3) = mulRR;
      //      svCurrent.matDCovCurv(3,4) = 0;

      //      svCurrent.matDCovCurv(4,0) = 0;
      svCurrent.matDCovCurv(4,1) = mulRP/p0Mat;
      //      svCurrent.matDCovCurv(4,2) = 0;
      //      svCurrent.matDCovCurv(4,3) = 0;
      svCurrent.matDCovCurv(4,4) = mulRR;
    }
    break;
  }
  default:
    break;
  }
  
  double pMag = curP;//svCurrent.p3.mag();
  
  if (dir == oppositeToMomentum) dP = fabs(dP);
  else if( dP < 0) { //the case of negative dP
    dP = -dP > pMag ? -pMag+1e-5 : dP;
  }
  
  getNextState(svCurrent, svNext, dP, tauNext, drVec, dS, dS/radX0,
	       dCCurvTransform);
  return true;
}

double SteppingHelixPropagator::getDeDx(const SteppingHelixPropagator::StateInfo& sv, 
					double& dEdXPrime, double& radX0,
					MatBounds& rzLims) const{
  radX0 = 1.e24;
  dEdXPrime = 0.;
  rzLims = MatBounds();
  if (noMaterialMode_) return 0;

  double dEdx = 0.;

  double lR = sv.r3.perp();
  double lZ = fabs(sv.r3.z());

  //assume "Iron" .. seems to be quite the same for brass/iron/PbW04
  //good for Fe within 3% for 0.2 GeV to 10PeV

  double dEdX_HCal = 0.95; //extracted from sim
  double dEdX_ECal = 0.45;
  double dEdX_coil = 0.35; //extracted from sim .. closer to 40% in fact
  double dEdX_Fe =   1;
  double dEdX_MCh =  0.053; //chambers on average
  double dEdX_Trk =  0.0114;
  double dEdX_Air =  2E-4;
  double dEdX_Vac =  0.0;

  double radX0_HCal = 1.44/0.8; //guessing
  double radX0_ECal = 0.89/0.7;
  double radX0_coil = 4.; //
  double radX0_Fe =   1.76;
  double radX0_MCh =  1e3; //
  double radX0_Trk =  320.;
  double radX0_Air =  3.e4;
  double radX0_Vac =  3.e9; //"big" number for vacuum


  //not all the boundaries are set below: this will be a default
  if (! (lR < 380 && lZ < 785)){
    if (lZ > 785 ) rzLims = MatBounds(0, 1e4, 785, 1e4);
    if (lZ < 785 ) rzLims = MatBounds(380, 1e4, 0, 785);
  }

  //this def makes sense assuming we don't jump into endcap volume from the other z-side in one step
  //also, it is a positive shift only (at least for now): can't move ec further into the detector
  double ecShift = sv.r3.z() > 0 ? fabs(ecShiftPos_) : fabs(ecShiftNeg_); 

  //this should roughly figure out where things are 
  //(numbers taken from Fig1.1.2 TDR and from geom xmls)
  if (lR < 2.9){ //inside beampipe
    dEdx = dEdX_Vac; radX0 = radX0_Vac;
    rzLims = MatBounds(0, 2.9, 0, 1E4);
  }
  else if (lR < 129){
    if (lZ < 294){ 
      rzLims = MatBounds(2.9, 129, 0, 294); //tracker boundaries
      dEdx = dEdX_Trk; radX0 = radX0_Trk; 
      //somewhat empirical formula that ~ matches the average if going from 0,0,0
      //assuming "uniform" tracker material
      //doesn't really track material layer to layer
      double lEtaDet = fabs(sv.r3.eta());
      double scaleRadX = lEtaDet > 1.5 ? 0.7724 : 1./cosh(0.5*lEtaDet);//sin(2.*atan(exp(-0.5*lEtaDet)));
      scaleRadX *= scaleRadX;
      if (lEtaDet > 2 && lZ > 20) scaleRadX *= (lEtaDet-1.);
      if (lEtaDet > 2.5 && lZ > 20) scaleRadX *= (lEtaDet-1.);
      radX0 *= scaleRadX;
    }
    //endcap part begins here
    else if ( lZ < 294 + ecShift ){
      //gap in front of EE here, piece inside 2.9<R<129
      rzLims = MatBounds(2.9, 129, 294, 294 + ecShift); 
      dEdx = dEdX_Air; radX0 = radX0_Air;
    }
    else if (lZ < 372 + ecShift){ 
      rzLims = MatBounds(2.9, 129, 294 + ecShift, 372 + ecShift); //EE here, piece inside 2.9<R<129
      dEdx = dEdX_ECal; radX0 = radX0_ECal; 
    }//EE averaged out over a larger space
    else if (lZ < 398 + ecShift){
      rzLims = MatBounds(2.9, 129, 372 + ecShift, 398 + ecShift); //whatever goes behind EE 2.9<R<129 is air up to Z=398
      dEdx = dEdX_HCal*0.05; radX0 = radX0_Air; 
    }//betw EE and HE
    else if (lZ < 555 + ecShift){ 
      rzLims = MatBounds(2.9, 129, 398 + ecShift, 555 + ecShift); //HE piece 2.9<R<129; 
      dEdx = dEdX_HCal*0.96; radX0 = radX0_HCal/0.96; 
    } //HE calor abit less dense
    else {
      //      rzLims = MatBounds(2.9, 129, 555, 785);
      // set the boundaries first: they serve as stop-points here
      // the material is set below
      if (lZ < 568  + ecShift) rzLims = MatBounds(2.9, 129, 555 + ecShift, 568 + ecShift); //a piece of HE support R<129, 555<Z<568
      else if (lZ < 625 + ecShift){
	if (lR < 85 + ecShift) rzLims = MatBounds(2.9, 85, 568 + ecShift, 625 + ecShift); 
	else rzLims = MatBounds(85, 129, 568 + ecShift, 625 + ecShift);
      } else if (lZ < 785 + ecShift) rzLims = MatBounds(2.9, 129, 625 + ecShift, 785 + ecShift);
      else if (lZ < 1500 + ecShift)  rzLims = MatBounds(2.9, 129, 785 + ecShift, 1500 + ecShift);
      else rzLims = MatBounds(2.9, 129, 1500 + ecShift, 1E4);

      //iron .. don't care about no material in front of HF (too forward)
      if (! (lZ > 568 + ecShift && lZ < 625 + ecShift && lR > 85 ) // HE support 
	  && ! (lZ > 785 + ecShift && lZ < 850 + ecShift && lR > 118)) {dEdx = dEdX_Fe; radX0 = radX0_Fe; }
      else  { dEdx = dEdX_MCh; radX0 = radX0_MCh; } //ME at eta > 2.2
    }
  }
  else if (lR < 287){
    if (lZ < 372 + ecShift && lR < 177){ // 129<<R<177
      if (lZ < 304) rzLims = MatBounds(129, 177, 0, 304); //EB  129<<R<177 0<Z<304
      else if (lZ < 343){ // 129<<R<177 304<Z<343
	if (lR < 135 ) rzLims = MatBounds(129, 135, 304, 343);// tk piece 129<<R<135 304<Z<343
	else if (lR < 172 ){ //
	  if (lZ < 311 ) rzLims = MatBounds(135, 172, 304, 311);
	  else rzLims = MatBounds(135, 172, 311, 343);
	} else {
	  if (lZ < 328) rzLims = MatBounds(172, 177, 304, 328);
	  else rzLims = MatBounds(172, 177, 328, 343);
	}
      }
      else if ( lZ < 343 + ecShift){
	rzLims = MatBounds(129, 177, 343, 343 + ecShift); //gap
      }
      else {
	if (lR < 156 ) rzLims = MatBounds(129, 156, 343 + ecShift, 372 + ecShift);
	else if ( (lZ - 343 - ecShift) > (lR - 156)*1.38 ) 
	  rzLims = MatBounds(156, 177, 127.72 + ecShift, 372 + ecShift, atan(1.38), 0);
	else rzLims = MatBounds(156, 177, 343 + ecShift, 127.72 + ecShift, 0, atan(1.38));
      }

      if (!(lR > 135 && lZ <343 + ecShift && lZ > 304 )
	  && ! (lR > 156 && lZ < 372 + ecShift  && lZ > 343 + ecShift  && ((lZ-343. - ecShift )< (lR-156.)*1.38)))
	{
	  //the crystals are the same length, but they are not 100% of material
	  double cosThetaEquiv = 0.8/sqrt(1.+lZ*lZ/lR/lR) + 0.2;
	  if (lZ > 343) cosThetaEquiv = 1.;
	  dEdx = dEdX_ECal*cosThetaEquiv; radX0 = radX0_ECal/cosThetaEquiv; 
	} //EB
      else { 
	if ( (lZ > 304 && lZ < 328 && lR < 177 && lR > 135) 
	     && ! (lZ > 311 && lR < 172) ) {dEdx = dEdX_Fe; radX0 = radX0_Fe; } //Tk_Support
	else if ( lZ > 343 && lZ < 343 + ecShift) { dEdx = dEdX_Air; radX0 = radX0_Air; }
	else {dEdx = dEdX_ECal*0.2; radX0 = radX0_Air;} //cables go here <-- will be abit too dense for ecShift > 0
      }
    }
    else if (lZ < 554 + ecShift){ // 129<R<177 372<Z<554  AND 177<R<287 0<Z<554
      if (lR < 177){ //  129<R<177 372<Z<554
	if ( lZ > 372 + ecShift && lZ < 398 + ecShift )rzLims = MatBounds(129, 177, 372 + ecShift, 398 + ecShift); // EE 129<R<177 372<Z<398
	else if (lZ < 548 + ecShift) rzLims = MatBounds(129, 177, 398 + ecShift, 548 + ecShift); // HE 129<R<177 398<Z<548
	else rzLims = MatBounds(129, 177, 548 + ecShift, 554 + ecShift); // HE gap 129<R<177 548<Z<554
      }
      else if (lR < 193){ // 177<R<193 0<Z<554
	if ((lZ - 307) < (lR - 177.)*1.739) rzLims = MatBounds(177, 193, 0, -0.803, 0, atan(1.739));
	else if ( lZ < 389)  rzLims = MatBounds(177, 193, -0.803, 389, atan(1.739), 0.);
	else if ( lZ < 389 + ecShift)  rzLims = MatBounds(177, 193, 389, 389 + ecShift); // air insert
	else if ( lZ < 548 + ecShift ) rzLims = MatBounds(177, 193, 389 + ecShift, 548 + ecShift);// HE 177<R<193 389<Z<548
	else rzLims = MatBounds(177, 193, 548 + ecShift, 554 + ecShift); // HE gap 177<R<193 548<Z<554
      }
      else if (lR < 264){ // 193<R<264 0<Z<554
	double anApex = 375.7278 - 193./1.327; // 230.28695599096
	if ( (lZ - 375.7278) < (lR - 193.)/1.327) rzLims = MatBounds(193, 264, 0, anApex, 0, atan(1./1.327)); //HB
	else if ( (lZ - 392.7278 ) < (lR - 193.)/1.327) 
	  rzLims = MatBounds(193, 264, anApex, anApex+17., atan(1./1.327), atan(1./1.327)); // HB-HE gap
	else if ( (lZ - 392.7278 - ecShift ) < (lR - 193.)/1.327) 
	  rzLims = MatBounds(193, 264, anApex+17., anApex+17. + ecShift, atan(1./1.327), atan(1./1.327)); // HB-HE gap air insert
	// HE (372,193)-(517,193)-(517,264)-(417.8,264)
	else if ( lZ < 517 + ecShift ) rzLims = MatBounds(193, 264, anApex+17. + ecShift, 517 + ecShift, atan(1./1.327), 0);
	else if (lZ < 548 + ecShift){ // HE+gap 193<R<264 517<Z<548
	  if (lR < 246 ) rzLims = MatBounds(193, 246, 517 + ecShift, 548 + ecShift); // HE 193<R<246 517<Z<548
	  else rzLims = MatBounds(246, 264, 517 + ecShift, 548 + ecShift); // HE gap 246<R<264 517<Z<548
	} 
	else rzLims = MatBounds(193, 264, 548 + ecShift, 554 + ecShift); // HE gap  193<R<246 548<Z<554
      }
      else if ( lR < 275){ // 264<R<275 0<Z<554
	if (lZ < 433) rzLims = MatBounds(264, 275, 0, 433); //HB
	else if (lZ < 554 ) rzLims = MatBounds(264, 275, 433, 554); // HE gap 264<R<275 433<Z<554
	else rzLims = MatBounds(264, 275, 554, 554 + ecShift); // HE gap 264<R<275 554<Z<554 air insert
      }
      else { // 275<R<287 0<Z<554
	if (lZ < 402) rzLims = MatBounds(275, 287, 0, 402);// HB 275<R<287 0<Z<402
	else if (lZ < 554) rzLims = MatBounds(275, 287, 402, 554);//  //HE gap 275<R<287 402<Z<554
	else rzLims = MatBounds(275, 287, 554, 554 + ecShift); //HE gap 275<R<287 554<Z<554 air insert
      }

      if ((lZ < 433 || lR < 264) && (lZ < 402 || lR < 275) && (lZ < 517 + ecShift || lR < 246) //notches
	  //I should've made HE and HF different .. now need to shorten HE to match
	  && lZ < 548 + ecShift
	  && ! (lZ < 389 + ecShift && lZ > 335 && lR < 193 ) //not a gap EB-EE 129<R<193
	  && ! (lZ > 307 && lZ < 335 && lR < 193 && ((lZ - 307) > (lR - 177.)*1.739)) //not a gap 
	  && ! (lR < 177 && lZ < 398 + ecShift) //under the HE nose
	  && ! (lR < 264 && lR > 193 && fabs(441.5+0.5*ecShift - lZ + (lR - 269.)/1.327) < (8.5 + ecShift*0.5)) ) //not a gap
	{ dEdx = dEdX_HCal; radX0 = radX0_HCal; //hcal
	}
      else if( (lR < 193 && lZ > 389 && lZ < 389 + ecShift ) || (lR > 264 && lR < 287 && lZ > 554 && lZ < 554 + ecShift)
	       ||(lR < 264 && lR > 193 && fabs(441.5+8.5+0.5*ecShift - lZ + (lR - 269.)/1.327) < ecShift*0.5) )  {
	dEdx = dEdX_Air; radX0 = radX0_Air; 
      }
      else {dEdx = dEdX_HCal*0.05; radX0 = radX0_Air; }//endcap gap
    }
    //the rest is a tube of endcap volume  129<R<251 554<Z<largeValue
    else if (lZ < 564 + ecShift){ // 129<R<287  554<Z<564
      if (lR < 251) {
	rzLims = MatBounds(129, 251, 554 + ecShift, 564 + ecShift);  // HE support 129<R<251  554<Z<564    
	dEdx = dEdX_Fe; radX0 = radX0_Fe; 
      }//HE support
      else { 
	rzLims = MatBounds(251, 287, 554 + ecShift, 564 + ecShift); //HE/ME gap 251<R<287  554<Z<564    
	dEdx = dEdX_MCh; radX0 = radX0_MCh; 
      }
    }
    else if (lZ < 625 + ecShift){ // ME/1/1 129<R<287  564<Z<625
      rzLims = MatBounds(129, 287, 564 + ecShift, 625 + ecShift);      
      dEdx = dEdX_MCh; radX0 = radX0_MCh; 
    }
    else if (lZ < 785 + ecShift){ //129<R<287  625<Z<785
      if (lR < 275) rzLims = MatBounds(129, 275, 625 + ecShift, 785 + ecShift); //YE/1 129<R<275 625<Z<785
      else { // 275<R<287  625<Z<785
	if (lZ < 720 + ecShift) rzLims = MatBounds(275, 287, 625 + ecShift, 720 + ecShift); // ME/1/2 275<R<287  625<Z<720
	else rzLims = MatBounds(275, 287, 720 + ecShift, 785 + ecShift); // YE/1 275<R<287  720<Z<785
      }
      if (! (lR > 275 && lZ < 720 + ecShift)) { dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
      else { dEdx = dEdX_MCh; radX0 = radX0_MCh; }
    }
    else if (lZ < 850 + ecShift){
      rzLims = MatBounds(129, 287, 785 + ecShift, 850 + ecShift);
      dEdx = dEdX_MCh; radX0 = radX0_MCh; 
    }
    else if (lZ < 910 + ecShift){
      rzLims = MatBounds(129, 287, 850 + ecShift, 910 + ecShift);
      dEdx = dEdX_Fe; radX0 = radX0_Fe; 
    }//iron
    else if (lZ < 975 + ecShift){
      rzLims = MatBounds(129, 287, 910 + ecShift, 975 + ecShift);
      dEdx = dEdX_MCh; radX0 = radX0_MCh; 
    }
    else if (lZ < 1000 + ecShift){
      rzLims = MatBounds(129, 287, 975 + ecShift, 1000 + ecShift);
      dEdx = dEdX_Fe; radX0 = radX0_Fe; 
    }//iron
    else if (lZ < 1063 + ecShift){
      rzLims = MatBounds(129, 287, 1000 + ecShift, 1063 + ecShift);
      dEdx = dEdX_MCh; radX0 = radX0_MCh; 
    }
    else if ( lZ < 1073 + ecShift){
      rzLims = MatBounds(129, 287, 1063 + ecShift, 1073 + ecShift);
      dEdx = dEdX_Fe; radX0 = radX0_Fe; 
    }
    else if (lZ < 1E4 )  { 
      rzLims = MatBounds(129, 287, 1073 + ecShift, 1E4);
      dEdx = dEdX_Air; radX0 = radX0_Air;
    }
    else { 
      
      dEdx = dEdX_Air; radX0 = radX0_Air;
    }
  }
  else if (lR <380 && lZ < 667){
    if (lZ < 630){
      if (lR < 315) rzLims = MatBounds(287, 315, 0, 630); 
      else if (lR < 341 ) rzLims = MatBounds(315, 341, 0, 630); //b-field ~linear rapid fall here
      else rzLims = MatBounds(341, 380, 0, 630);      
    } else rzLims = MatBounds(287, 380, 630, 667);  

    if (lZ < 630) { dEdx = dEdX_coil; radX0 = radX0_coil; }//a guess for the solenoid average
    else {dEdx = dEdX_Air; radX0 = radX0_Air; }//endcap gap
  }
  else {
    if (lZ < 667) {
      if (lR < 850){
	bool isIron = false;
	//sanity check in addition to flags
	if (useIsYokeFlag_ && useMagVolumes_ && sv.magVol != 0){
	  isIron = sv.isYokeVol;
	} else {
	  double bMag = sv.bf.mag();
	  isIron = (bMag > 0.75 && ! (lZ > 500 && lR <500 && bMag < 1.15)
		    && ! (lZ < 450 && lR > 420 && bMag < 1.15 ) );
	}
	//tell the material stepper where mat bounds are
	rzLims = MatBounds(380, 850, 0, 667);
	if (isIron) { dEdx = dEdX_Fe; radX0 = radX0_Fe; }//iron
	else { dEdx = dEdX_MCh; radX0 = radX0_MCh; }
      } else {
	rzLims = MatBounds(850, 1E4, 0, 667);
	dEdx = dEdX_Air; radX0 = radX0_Air; 
      }
    } 
    else if (lR > 750 ){
      rzLims = MatBounds(750, 1E4, 667, 1E4);
      dEdx = dEdX_Air; radX0 = radX0_Air; 
    }
    else if (lZ < 667 + ecShift){
      rzLims = MatBounds(287, 750, 667, 667 + ecShift);
      dEdx = dEdX_Air; radX0 = radX0_Air;       
    }
    //the rest is endcap piece with 287<R<750 Z>667
    else if (lZ < 724 + ecShift){
      if (lR < 380 ) rzLims = MatBounds(287, 380, 667 + ecShift, 724 + ecShift); 
      else rzLims = MatBounds(380, 750, 667 + ecShift, 724 + ecShift); 
      dEdx = dEdX_MCh; radX0 = radX0_MCh; 
    }
    else if (lZ < 785 + ecShift){
      if (lR < 380 ) rzLims = MatBounds(287, 380, 724 + ecShift, 785 + ecShift); 
      else rzLims = MatBounds(380, 750, 724 + ecShift, 785 + ecShift); 
      dEdx = dEdX_Fe; radX0 = radX0_Fe; 
    }//iron
    else if (lZ < 850 + ecShift){
      rzLims = MatBounds(287, 750, 785 + ecShift, 850 + ecShift); 
      dEdx = dEdX_MCh; radX0 = radX0_MCh; 
    }
    else if (lZ < 910 + ecShift){
      rzLims = MatBounds(287, 750, 850 + ecShift, 910 + ecShift); 
      dEdx = dEdX_Fe; radX0 = radX0_Fe; 
    }//iron
    else if (lZ < 975 + ecShift){
      rzLims = MatBounds(287, 750, 910 + ecShift, 975 + ecShift); 
      dEdx = dEdX_MCh; radX0 = radX0_MCh; 
    }
    else if (lZ < 1000 + ecShift){
      rzLims = MatBounds(287, 750, 975 + ecShift, 1000 + ecShift); 
      dEdx = dEdX_Fe; radX0 = radX0_Fe; 
    }//iron
    else if (lZ < 1063 + ecShift){
      if (lR < 360){
	rzLims = MatBounds(287, 360, 1000 + ecShift, 1063 + ecShift);
	dEdx = dEdX_MCh; radX0 = radX0_MCh; 
      } 
      //put empty air where me4/2 should be (future)
      else {
        rzLims = MatBounds(360, 750, 1000 + ecShift, 1063 + ecShift);
        dEdx = dEdX_Air; radX0 = radX0_Air;
      }
    }
    else if ( lZ < 1073 + ecShift){
      rzLims = MatBounds(287, 750, 1063 + ecShift, 1073 + ecShift);
      //this plate does not exist: air
      dEdx = dEdX_Air; radX0 = radX0_Air; 
    }
    else if (lZ < 1E4 )  { 
      rzLims = MatBounds(287, 750, 1073 + ecShift, 1E4);
      dEdx = dEdX_Air; radX0 = radX0_Air;
    }
    else {dEdx = dEdX_Air; radX0 = radX0_Air; }//air
  }
  
  //dEdx so far is a relative number (relative to iron)
  //scale by what's expected for iron (the function was fit from pdg table)
  //0.065 (PDG) --> 0.044 to better match with MPV
  double p0 = sv.p3.mag();
  double logp0 = log(p0);
  double p0powN33 = 0; 
  if (p0>3.) {
    // p0powN33 = exp(-0.33*logp0); //calculate for p>3GeV
    double xx=1./p0; xx=sqrt(sqrt(sqrt(sqrt(xx)))); xx=xx*xx*xx*xx*xx; // this is (p0)**(-5/16), close enough to -0.33
    p0powN33 = xx;
  }
  double dEdX_mat = -(11.4 + 0.96*fabs(logp0+log(2.8)) + 0.033*p0*(1.0 - p0powN33) )*1e-3; 
  //in GeV/cm .. 0.8 to get closer to the median or MPV

  dEdXPrime = dEdx == 0 ? 0 : -dEdx*(0.96/p0 + 0.033 - 0.022*p0powN33)*1e-3; //== d(dEdX)/dp
  dEdx *= dEdX_mat;

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
				   PropagationDirection& refDirection,
				   double fastSkipDist) const{
  static const std::string metname = "SteppingHelixPropagator";
  Result result = SteppingHelixStateInfo::NOT_IMPLEMENTED;

  switch (dest){
  case RADIUS_DT:
    {
      double curR = sv.r3.perp();
      dist = pars[RADIUS_P] - curR;
      if (fabs(dist) > fastSkipDist){
	result = SteppingHelixStateInfo::INACC;
	break;
      }
      double curP2 = sv.p3.mag2();
      double curPtPos2 = sv.p3.perp2(); if(curPtPos2< 1e-16) curPtPos2=1e-16;

      double cosDPhiPR = (sv.r3.x()*sv.p3.x()+sv.r3.y()*sv.p3.y());//only the sign is needed cos((sv.r3.deltaPhi(sv.p3)));
      refDirection = dist*cosDPhiPR > 0 ?
	alongMomentum : oppositeToMomentum;
      tanDist = dist*sqrt(curP2/curPtPos2);      
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case Z_DT:
    {
      double curZ = sv.r3.z();
      dist = pars[Z_P] - curZ;
      if (fabs(dist) > fastSkipDist){
	result = SteppingHelixStateInfo::INACC;
	break;
      }
      double curP = sv.p3.mag();
      refDirection = sv.p3.z()*dist > 0. ?
	alongMomentum : oppositeToMomentum;
      tanDist = dist/sv.p3.z()*curP;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case PLANE_DT:
    {
      Point rPlane(pars[0], pars[1], pars[2]);
      Vector nPlane(pars[3], pars[4], pars[5]);
      

      // unfortunately this doesn't work: the numbers are too large
      //      bool pVertical = fabs(pars[5])>0.9999;
      //      double dRDotN = pVertical? (sv.r3.z() - rPlane.z())*nPlane.z() :(sv.r3 - rPlane).dot(nPlane);
      double dRDotN = (sv.r3.x()-rPlane.x())*nPlane.x() + (sv.r3.y()-rPlane.y())*nPlane.y() + (sv.r3.z()-rPlane.z())*nPlane.z();//(sv.r3 - rPlane).dot(nPlane);

      dist = fabs(dRDotN);
      if (dist > fastSkipDist){
	result = SteppingHelixStateInfo::INACC;
	break;
      }
      double curP = sv.p3.mag();
      double p0 = curP;
      double p0Inv = 1./p0;
      Vector tau(sv.p3); tau *=p0Inv; 
      double tN =  tau.dot(nPlane);
      refDirection = tN*dRDotN < 0. ?
	alongMomentum : oppositeToMomentum;
      double b0 = sv.bf.mag();
      if (fabs(tN)>1e-24){
	tanDist = -dRDotN/tN;
      } else {
	tN = 1e-24;
	if (fabs(dRDotN)>1e-24) tanDist = 1e6;
	else tanDist = 1;
      }
      if (fabs(tanDist) > 1e4) tanDist = 1e4;
      if (b0>1.5e-6){
	double b0Inv = 1./b0;
	double tNInv = 1./tN;
	Vector bHat(sv.bf); bHat *=b0Inv;
	double bHatN = bHat.dot(nPlane);
	double cosPB = bHat.dot(tau);
	double kVal = 2.99792458e-3*sv.q*p0Inv*b0;
	double aVal = tanDist*kVal;
	Vector lVec = bHat.cross(tau);
	double bVal = lVec.dot(nPlane)*tNInv;
	if (fabs(aVal*bVal)< 0.3){
	  double cVal = 1. - bHatN*cosPB*tNInv; // - sv.bf.cross(lVec).dot(nPlane)*b0Inv*tNInv; //1- bHat_n*bHat_tau/tau_n;
          double aacVal = cVal*aVal*aVal;
          if (fabs(aacVal)<1){
            double tanDCorr = bVal/2. + (bVal*bVal/2. + cVal/6)*aVal;
            tanDCorr *= aVal;
            //+ (-bVal/24. + 0.625*bVal*bVal*bVal + 5./12.*bVal*cVal)*aVal*aVal*aVal
            if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<tanDist<<" vs "
					 <<tanDist*(1.+tanDCorr)<<" corr "<<tanDist*tanDCorr<<std::endl;
            tanDist *= (1.+tanDCorr);
          } else {
            if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"AACVal "<< fabs(aacVal)
                                         <<" = "<<aVal<<"**2 * "<<cVal<<" too large:: will not converge"<<std::endl;
          }
	} else {
	  if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"ABVal "<< fabs(aVal*bVal)
				       <<" = "<<aVal<<" * "<<bVal<<" too large:: will not converge"<<std::endl;
	}
      }
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case CONE_DT:
    {
      Point cVertex(pars[0], pars[1], pars[2]);
      if (cVertex.perp2() < 1e-10){
	//assumes the cone axis/vertex is along z
	Vector relV3 = sv.r3 - cVertex;
	double relV3mag = relV3.mag();
	//	double relV3Theta = relV3.theta();
	double theta(pars[3]);
	//	double dTheta = theta-relV3Theta;
	double sinTheta = sin(theta);
	double cosTheta = cos(theta);
	double cosV3Theta = relV3.z()/relV3mag;
	if (cosV3Theta>1) cosV3Theta=1;
	if (cosV3Theta<-1) cosV3Theta=-1;
	double sinV3Theta = sqrt(1.-cosV3Theta*cosV3Theta);

	double sinDTheta = sinTheta*cosV3Theta - cosTheta*sinV3Theta;//sin(dTheta);
	double cosDTheta = cosTheta*cosV3Theta + sinTheta*sinV3Theta;//cos(dTheta);
	bool isInside = sinTheta > sinV3Theta  && cosTheta*cosV3Theta > 0;
	dist = isInside || cosDTheta > 0 ? relV3mag*sinDTheta : relV3mag;
	if (fabs(dist) > fastSkipDist){
	  result = SteppingHelixStateInfo::INACC;
	  break;
	}

	double relV3phi=relV3.phi();
	double normPhi = isInside ? 
	  Geom::pi() + relV3phi : relV3phi;
	double normTheta = theta > Geom::pi()/2. ?
	  ( isInside ? 1.5*Geom::pi()  - theta : theta - Geom::pi()/2. )
	  : ( isInside ? Geom::pi()/2. - theta : theta + Geom::pi()/2 );
	//this is a normVector from the cone to the point
	Vector norm; norm.setRThetaPhi(fabs(dist), normTheta, normPhi);
	double curP = sv.p3.mag(); double cosp3theta = sv.p3.z()/curP; 
	if (cosp3theta>1) cosp3theta=1;
	if (cosp3theta<-1) cosp3theta=-1;
	double sineConeP = sinTheta*cosp3theta - cosTheta*sqrt(1.-cosp3theta*cosp3theta);

	double sinSolid = norm.dot(sv.p3)/(fabs(dist)*curP);
	tanDist = fabs(sinSolid) > fabs(sineConeP) ? dist/fabs(sinSolid) : dist/fabs(sineConeP);


	refDirection = sinSolid > 0 ?
	  oppositeToMomentum : alongMomentum;
	if (debug_){
	  LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Cone pars: theta "<<theta
			   <<" normTheta "<<norm.theta()
			   <<" p3Theta "<<sv.p3.theta()
			   <<" sinD: "<< sineConeP
			   <<" sinSolid "<<sinSolid;
	}
	if (debug_){
	  LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"refToDest:toCone the point is "
			   <<(isInside? "in" : "out")<<"side the cone"
			   <<std::endl;
	}
      }
      result = SteppingHelixStateInfo::OK;
    }
    break;
    //   case CYLINDER_DT:
    //     break;
  case PATHL_DT:
    {
      double curS = fabs(sv.path());
      dist = pars[PATHL_P] - curS;
      if (fabs(dist) > fastSkipDist){
	result = SteppingHelixStateInfo::INACC;
	break;
      }
      refDirection = pars[PATHL_P] > 0 ? 
	alongMomentum : oppositeToMomentum;
      tanDist = dist;
      result = SteppingHelixStateInfo::OK;
    }
    break;
  case POINT_PCA_DT:
    {
      Point pDest(pars[0], pars[1], pars[2]);
      double curP = sv.p3.mag();
      dist = (sv.r3 - pDest).mag()+ 1e-24;//add a small number to avoid 1/0
      tanDist = (sv.r3.dot(sv.p3) - pDest.dot(sv.p3))/curP;
      //account for bending in magnetic field (quite approximate)
      double b0 = sv.bf.mag();
      if (b0>1.5e-6){
	double p0 = curP;
        double kVal = 2.99792458e-3*sv.q/p0*b0;
        double aVal = fabs(dist*kVal);
        tanDist *= 1./(1.+ aVal);
	if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"corrected by aVal "<<aVal<<" to "<<tanDist;
      }
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
      dLine *= 1./dLine.mag(); if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"dLine "<<dLine;

      Vector dR = sv.r3 - rLine;
      double curP = sv.p3.mag();
      Vector dRPerp = dR - dLine*(dR.dot(dLine)); 
      if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"dRperp "<<dRPerp;

      dist = dRPerp.mag() + 1e-24;//add a small number to avoid 1/0
      tanDist = dRPerp.dot(sv.p3)/curP;
      //angle wrt line
      double cosAlpha2 = dLine.dot(sv.p3)/curP; cosAlpha2 *= cosAlpha2;
      tanDist *= 1./sqrt(fabs(1.-cosAlpha2)+1e-96);
      //correct for dPhi in magnetic field: this isn't made quite right here 
      //(the angle between the line and the trajectory plane is neglected .. conservative)
      double b0 = sv.bf.mag();
      if (b0>1.5e-6){
	double p0 = curP;
        double kVal = 2.99792458e-3*sv.q/p0*b0;
        double aVal = fabs(dist*kVal);
	tanDist *= 1./(1.+ aVal);
	if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"corrected by aVal "<<aVal<<" to "<<tanDist;
      }
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

  double tanDistConstrained = tanDist;
  if (fabs(tanDist) > 4.*fabs(dist) ) tanDistConstrained *= tanDist == 0 ? 0 : fabs(dist/tanDist*4.);

  if (debug_){
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"refToDest input: dest"<<dest<<" pars[]: ";
    for (int i = 0; i < 6; i++){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<", "<<i<<" "<<pars[i];
    }
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<std::endl;
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"refToDest output: "
		     <<"\t dist" << dist
		     <<"\t tanDist "<< tanDist      
		     <<"\t tanDistConstr "<< tanDistConstrained      
		     <<"\t refDirection "<< refDirection
		     <<std::endl;
  }
  tanDist = tanDistConstrained;

  return result;
}

SteppingHelixPropagator::Result
SteppingHelixPropagator::refToMagVolume(const SteppingHelixPropagator::StateInfo& sv,
					PropagationDirection dir,
					double& dist, double& tanDist,
					double fastSkipDist, bool expectNewMagVolume, double maxStep) const{

  static const std::string metname = "SteppingHelixPropagator";
  Result result = SteppingHelixStateInfo::NOT_IMPLEMENTED;
  const MagVolume* cVol = sv.magVol;

  if (cVol == 0) return result;
  const std::vector<VolumeSide>& cVolFaces(cVol->faces());

  double distToFace[6] = {0,0,0,0,0,0};
  double tanDistToFace[6] = {0,0,0,0,0,0};
  PropagationDirection refDirectionToFace[6] = {anyDirection, anyDirection, anyDirection, anyDirection, anyDirection, anyDirection};
  Result resultToFace[6] = {result, result, result, result, result, result};
  int iFDest = -1;
  int iDistMin = -1;
  
  unsigned int iFDestSorted[6] = {0,0,0,0,0,0};
  int nDestSorted =0;
  unsigned int nearParallels = 0;

  double curP = sv.p3.mag();

  if (debug_){
    LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Trying volume "<<DDSolidShapesName::name(cVol->shapeType())
		     <<" with "<<cVolFaces.size()<<" faces"<<std::endl;
  }

  unsigned int nFaces = cVolFaces.size();
  for (unsigned int iFace = 0; iFace < nFaces; ++iFace){
    if (iFace > 5){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Too many faces"<<std::endl;
    }
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Start with face "<<iFace<<std::endl;
    }
//     const Plane* cPlane = dynamic_cast<const Plane*>(&cVolFaces[iFace].surface());
//     const Cylinder* cCyl = dynamic_cast<const Cylinder*>(&cVolFaces[iFace].surface());
//     const Cone* cCone = dynamic_cast<const Cone*>(&cVolFaces[iFace].surface());
    const Surface* cPlane = 0; //only need to know the loc->glob transform
    const Cylinder* cCyl = 0;
    const Cone* cCone = 0;
    if (typeid(cVolFaces[iFace].surface()) == typeid(const Plane&)){
      cPlane = &cVolFaces[iFace].surface();
    } else if (typeid(cVolFaces[iFace].surface()) == typeid(const Cylinder&)){
      cCyl = dynamic_cast<const Cylinder*>(&cVolFaces[iFace].surface());
    } else if (typeid(cVolFaces[iFace].surface()) == typeid(const Cone&)){
      cCone = dynamic_cast<const Cone*>(&cVolFaces[iFace].surface());
    } else {
      edm::LogWarning(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Could not cast a volume side surface to a known type"<<std::endl;
    }
    
    if (debug_){
      if (cPlane!=0) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"The face is a plane at "<<cPlane<<std::endl;
      if (cCyl!=0) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"The face is a cylinder at "<<cCyl<<std::endl;
    }

    double pars[6] = {0,0,0,0,0,0};
    DestType dType = UNDEFINED_DT;
    if (cPlane != 0){
      Point rPlane(cPlane->position().x(),cPlane->position().y(),cPlane->position().z());
      // = cPlane->toGlobal(LocalVector(0,0,1.)); nPlane = nPlane.unit();
      Vector nPlane(cPlane->rotation().zx(), cPlane->rotation().zy(), cPlane->rotation().zz()); nPlane /= nPlane.mag();
      
      pars[0] = rPlane.x(); pars[1] = rPlane.y(); pars[2] = rPlane.z();
      pars[3] = nPlane.x(); pars[4] = nPlane.y(); pars[5] = nPlane.z();
      dType = PLANE_DT;
    } else if (cCyl != 0){
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Cylinder at "<<cCyl->position()
			 <<" rorated by "<<cCyl->rotation()
			 <<std::endl;
      }
      pars[RADIUS_P] = cCyl->radius();
      dType = RADIUS_DT;
    } else if (cCone != 0){
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Cone at "<<cCone->position()
			 <<" rorated by "<<cCone->rotation()
			 <<" vertex at "<<cCone->vertex()
			 <<" angle of "<<cCone->openingAngle()
			 <<std::endl;
      }
      pars[0] = cCone->vertex().x(); pars[1] = cCone->vertex().y(); 
      pars[2] = cCone->vertex().z();
      pars[3] = cCone->openingAngle();
      dType = CONE_DT;
    } else {
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Unknown surface"<<std::endl;
      resultToFace[iFace] = SteppingHelixStateInfo::UNDEFINED;
      continue;
    }
    resultToFace[iFace] = 
      refToDest(dType, sv, pars, 
		distToFace[iFace], tanDistToFace[iFace], refDirectionToFace[iFace], fastSkipDist);    
    
    
    if (resultToFace[iFace] != SteppingHelixStateInfo::OK){
      if (resultToFace[iFace] == SteppingHelixStateInfo::INACC) result = SteppingHelixStateInfo::INACC;
    }


      
    //keep those in right direction for later use
    if (resultToFace[iFace] == SteppingHelixStateInfo::OK){
      double invDTFPosiF = 1./(1e-32+fabs(tanDistToFace[iFace]));
      double dSlope = fabs(distToFace[iFace])*invDTFPosiF;
      double maxStepL = maxStep> 100 ? 100 : maxStep; if (maxStepL < 10) maxStepL = 10.;
      bool isNearParallel = fabs(tanDistToFace[iFace]) + 100.*curP*dSlope < maxStepL //
	//a better choice is to use distance to next check of mag volume instead of 100cm; the last is for ~1.5arcLength(4T)+tandistance< maxStep
	&& dSlope < 0.15 ; //
      if (refDirectionToFace[iFace] == dir || isNearParallel){
	if (isNearParallel) nearParallels++;
	iFDestSorted[nDestSorted] = iFace;
	nDestSorted++;
      }
    }
    
    //pick a shortest distance here (right dir only for now)
    if (refDirectionToFace[iFace] == dir){
      if (iDistMin == -1) iDistMin = iFace;
      else if (fabs(distToFace[iFace]) < fabs(distToFace[iDistMin])) iDistMin = iFace;
    }
    if (debug_) 
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<cVol<<" "<<iFace<<" "
		       <<tanDistToFace[iFace]<<" "<<distToFace[iFace]<<" "<<refDirectionToFace[iFace]<<" "<<dir<<std::endl;
    
  }
   
  for (int i = 0;i<nDestSorted; ++i){
    int iMax = nDestSorted-i-1;
    for (int j=0;j<nDestSorted-i; ++j){
      if (fabs(tanDistToFace[iFDestSorted[j]]) > fabs(tanDistToFace[iFDestSorted[iMax]]) ){
	iMax = j;
      }
    }
    int iTmp = iFDestSorted[nDestSorted-i-1];
    iFDestSorted[nDestSorted-i-1] = iFDestSorted[iMax];
    iFDestSorted[iMax] = iTmp;
  }

  if (debug_){
    for (int i=0;i<nDestSorted;++i){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<cVol<<" "<<i<<" "<<iFDestSorted[i]<<" "<<tanDistToFace[iFDestSorted[i]]<<std::endl;
    }
  }

  //now go from the shortest to the largest distance hoping to get a point in the volume.
  //other than in case of a near-parallel travel this should stop after the first try
  
  for (int i=0; i<nDestSorted;++i){
    iFDest = iFDestSorted[i];

    double sign = dir == alongMomentum ? 1. : -1.;
    Point gPointEst(sv.r3);
    Vector lDelta(sv.p3); lDelta *= sign/curP*fabs(distToFace[iFDest]);
    gPointEst += lDelta;
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Linear est point "<<gPointEst
		       <<" for iFace "<<iFDest<<std::endl;
    }
    GlobalPoint gPointEstNorZ(gPointEst.x(), gPointEst.y(), gPointEst.z() );
    if (  cVol->inside(gPointEstNorZ)  ){
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"The point is inside the volume"<<std::endl;
      }
      
      result = SteppingHelixStateInfo::OK;
      dist = distToFace[iFDest];
      tanDist = tanDistToFace[iFDest];
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Got a point near closest boundary -- face "<<iFDest<<std::endl;
      }
      break;
    }
  }
  
  if (result != SteppingHelixStateInfo::OK && expectNewMagVolume){
    double sign = dir == alongMomentum ? 1. : -1.;

    //check if it's a wrong volume situation
    if (nDestSorted-nearParallels > 0 ) result = SteppingHelixStateInfo::WRONG_VOLUME;
    else {
      //get here if all faces in the corr direction were skipped
      Point gPointEst(sv.r3);
      double lDist = iDistMin == -1 ? fastSkipDist : fabs(distToFace[iDistMin]);
      if (lDist > fastSkipDist) lDist = fastSkipDist;
      Vector lDelta(sv.p3); lDelta *= sign/curP*lDist;
      gPointEst += lDelta;
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Linear est point to shortest dist "<<gPointEst
			 <<" for iFace "<<iDistMin<<" at distance "<<lDist*sign<<std::endl;
      }
      GlobalPoint gPointEstNorZ(gPointEst.x(), gPointEst.y(), gPointEst.z() );
      if ( cVol->inside(gPointEstNorZ) ){
	if (debug_){
	  LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"The point is inside the volume"<<std::endl;
	}
	
      }else {
	result = SteppingHelixStateInfo::WRONG_VOLUME;
      }
    }
    
    if (result == SteppingHelixStateInfo::WRONG_VOLUME){
      dist = sign*0.05;
      tanDist = dist*1.01;
      if( debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Wrong volume located: return small dist, tandist"<<std::endl;
      }
    }
  }

  if (result == SteppingHelixStateInfo::INACC){
    if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"All faces are too far"<<std::endl;
  } else if (result == SteppingHelixStateInfo::WRONG_VOLUME){
    if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Appear to be in a wrong volume"<<std::endl;
  } else if (result != SteppingHelixStateInfo::OK){
    if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Something else went wrong"<<std::endl;
  }

  
  return result;
}


SteppingHelixPropagator::Result
SteppingHelixPropagator::refToMatVolume(const SteppingHelixPropagator::StateInfo& sv,
					PropagationDirection dir,
					double& dist, double& tanDist,
					double fastSkipDist) const{

  static const std::string metname = "SteppingHelixPropagator";
  Result result = SteppingHelixStateInfo::NOT_IMPLEMENTED;

  double parLim[6] = {sv.rzLims.rMin, sv.rzLims.rMax, 
		      sv.rzLims.zMin, sv.rzLims.zMax, 
		      sv.rzLims.th1, sv.rzLims.th2 };

  double distToFace[4] = {0,0,0,0};
  double tanDistToFace[4] = {0,0,0,0};
  PropagationDirection refDirectionToFace[4] = {anyDirection, anyDirection, anyDirection, anyDirection};
  Result resultToFace[4] = {result, result, result, result};
  int iFDest = -1;
  
  double curP = sv.p3.mag();

  for (unsigned int iFace = 0; iFace < 4; iFace++){
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Start with mat face "<<iFace<<std::endl;
    }

    double pars[6] = {0,0,0,0,0,0};
    DestType dType = UNDEFINED_DT;
    if (iFace > 1){
      if (fabs(parLim[iFace+2])< 1e-6){//plane
	if (sv.r3.z() < 0){
	  pars[0] = 0; pars[1] = 0; pars[2] = -parLim[iFace];
	  pars[3] = 0; pars[4] = 0; pars[5] = 1;
	} else {
	  pars[0] = 0; pars[1] = 0; pars[2] = parLim[iFace];
	  pars[3] = 0; pars[4] = 0; pars[5] = 1;
	}
	dType = PLANE_DT;
      } else {
	if (sv.r3.z() > 0){
	  pars[0] = 0; pars[1] = 0; 
	  pars[2] = parLim[iFace];
	  pars[3] = Geom::pi()/2. - parLim[iFace+2];
	} else {
	  pars[0] = 0; pars[1] = 0; 
	  pars[2] = - parLim[iFace];
	  pars[3] = Geom::pi()/2. + parLim[iFace+2];
	}
	dType = CONE_DT;	
      }
    } else {
      pars[RADIUS_P] = parLim[iFace];
      dType = RADIUS_DT;
    }

    resultToFace[iFace] = 
      refToDest(dType, sv, pars, 
		distToFace[iFace], tanDistToFace[iFace], refDirectionToFace[iFace], fastSkipDist);
    
    if (resultToFace[iFace] != SteppingHelixStateInfo::OK){
      if (resultToFace[iFace] == SteppingHelixStateInfo::INACC) result = SteppingHelixStateInfo::INACC;
      continue;
    }
    if (refDirectionToFace[iFace] == dir || fabs(distToFace[iFace]) < 2e-2*fabs(tanDistToFace[iFace]) ){
      double sign = dir == alongMomentum ? 1. : -1.;
      Point gPointEst(sv.r3);
      Vector lDelta(sv.p3); lDelta *= sign*fabs(distToFace[iFace])/curP;
      gPointEst += lDelta;
      if (debug_){
	LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Linear est point "<<gPointEst
			 <<std::endl;
      }
      double lZ = fabs(gPointEst.z());
      double lR = gPointEst.perp();
      double tan4 = parLim[4] == 0 ? 0 : tan(parLim[4]);
      double tan5 = parLim[5] == 0 ? 0 : tan(parLim[5]);
      if ( (lZ - parLim[2]) > lR*tan4 
	   && (lZ - parLim[3]) < lR*tan5  
	   && lR > parLim[0] && lR < parLim[1]){
	if (debug_){
	  LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"The point is inside the volume"<<std::endl;
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
	  LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"The point is NOT inside the volume"<<std::endl;
	}
      }
    }

  }
  if (iFDest != -1){
    result = SteppingHelixStateInfo::OK;
    dist = distToFace[iFDest];
    tanDist = tanDistToFace[iFDest];
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Got a point near closest boundary -- face "<<iFDest<<std::endl;
    }
  } else {
    if (debug_){
      LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Failed to find a dest point inside the volume"<<std::endl;
    }
  }

  return result;
}


bool SteppingHelixPropagator::isYokeVolume(const MagVolume* vol) const {
  static const std::string metname = "SteppingHelixPropagator";
  if (vol == 0) return false;
  /*
  const MFGrid* mGrid = reinterpret_cast<const MFGrid*>(vol->provider());
  std::vector<int> dims(mGrid->dimensions());
  
  LocalVector lVCen(mGrid->nodeValue(dims[0]/2, dims[1]/2, dims[2]/2));
  LocalVector lVZLeft(mGrid->nodeValue(dims[0]/2, dims[1]/2, dims[2]/5));
  LocalVector lVZRight(mGrid->nodeValue(dims[0]/2, dims[1]/2, (dims[2]*4)/5));

  double mag2VCen = lVCen.mag2();
  double mag2VZLeft = lVZLeft.mag2();
  double mag2VZRight = lVZRight.mag2();

  bool result = false;
  if (mag2VCen > 0.6 && mag2VZLeft > 0.6 && mag2VZRight > 0.6){
    if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Volume is magnetic, located at "<<vol->position()<<std::endl;    
    result = true;
  } else {
    if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Volume is not magnetic, located at "<<vol->position()<<std::endl;
  }

  */
  bool result = vol->isIron();
  if (result){
    if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Volume is magnetic, located at "<<vol->position()<<std::endl;
  } else {
    if (debug_) LogTrace(metname)<<std::setprecision(17)<<std::setw(20)<<std::scientific<<"Volume is not magnetic, located at "<<vol->position()<<std::endl;
  }

  return result;
}
