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

SteppingHelixPropagator::SteppingHelixPropagator(const MagneticField* field, PropagationDirection dir):
  Propagator(dir),
  unit66_(6,1)
{
  field_ = field;
  covRot_ = HepMatrix(6,6,0);
  dCTransform_ = unit66_;
  debug_ = false;
  noMaterialMode_ = false;
  for (int i = 0; i <= MAX_POINTS; i++){
    covLoc_[i] = HepSymMatrix(6,0);
    cov_[i] = HepSymMatrix(6,0);
  }
}


std::pair<TrajectoryStateOnSurface, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, const Plane& pDest) const {
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

  setIState(p3, r3, charge, ftsStart.cartesianError().matrix(), propagationDirection());
  Result result = propagateToPlane(pars);

  Vector p3F;
  Point  r3F;
  HepSymMatrix covF;

  
  getFState(p3F, r3F, covF); 
  GlobalVector p3FGV(p3F.x(), p3F.y(), p3F.z());
  GlobalPoint r3FGP(r3F.x(), r3F.y(), r3F.z());
  //how do I tell that I didn't reach the surface without throuwing an exception?
  SurfaceSide side = result == OK ? atCenterOfSurface : beforeSurface;
  GlobalTrajectoryParameters tParsDest(r3FGP, p3FGV, charge, field_);
  CartesianTrajectoryError tCovDest(covF);

  TrajectoryStateOnSurface tsosDest(tParsDest, tCovDest, pDest, side);
  int cInd = cIndex_(nPoints_-1);
  
  return TsosPP(tsosDest, path_[cInd]);
}

std::pair<TrajectoryStateOnSurface, double> 
SteppingHelixPropagator::propagateWithPath(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const {
  //need to get rid of these conversions .. later
  GlobalVector p3GV = ftsStart.momentum();
  GlobalPoint r3GP = ftsStart.position();
  Vector p3(p3GV.x(), p3GV.y(), p3GV.z());
  Point  r3(r3GP.x(), r3GP.y(), r3GP.z());

  int charge = ftsStart.charge();

  setIState(p3, r3, charge, ftsStart.cartesianError().matrix(), propagationDirection());
  Result result = propagateToR(cDest.radius());

  Vector p3F;
  Point  r3F;
  HepSymMatrix covF;

  
  getFState(p3F, r3F, covF); 
  GlobalVector p3FGV(p3F.x(), p3F.y(), p3F.z());
  GlobalPoint r3FGP(r3F.x(), r3F.y(), r3F.z());
  //how do I tell that I didn't reach the surface without throuwing an exception?
  SurfaceSide side = result == OK ? atCenterOfSurface : beforeSurface;
  GlobalTrajectoryParameters tParsDest(r3FGP, p3FGV, charge, field_);
  CartesianTrajectoryError tCovDest(covF);

  TrajectoryStateOnSurface tsosDest(tParsDest, tCovDest, cDest, side);
  int cInd = cIndex_(nPoints_-1);

  return TsosPP(tsosDest, path_[cInd]);
}

TrajectoryStateOnSurface 
SteppingHelixPropagator::propagate(const FreeTrajectoryState& ftsStart, const Plane& pDest) const {
  return propagateWithPath(ftsStart, pDest).first;
}

TrajectoryStateOnSurface 
SteppingHelixPropagator::propagate(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const {
  return propagateWithPath(ftsStart, cDest).first;
}


void SteppingHelixPropagator::setIState(const SteppingHelixPropagator::Vector& p3, 
					const SteppingHelixPropagator::Point& r3, int charge, 
					const HepSymMatrix& cov, PropagationDirection dir) const {
  nPoints_ = 0;
  loadState(0, p3, r3, charge, cov, dir);
  nPoints_++;
}

void SteppingHelixPropagator::getFState(SteppingHelixPropagator::Vector& p3, SteppingHelixPropagator::Point& r3, 
					HepSymMatrix& cov) const{
  int cInd = cIndex_(nPoints_-1);
  p3 = p3_[cInd];
  r3 = r3_[cInd];
  cov.assign(cov_[cInd]);
}


SteppingHelixPropagator::Result SteppingHelixPropagator::propagate(SteppingHelixPropagator::DestType type, const double pars[6])  const{
  Result result = NOT_IMPLEMENTED;
  switch (type) {
  case RADIUS_DT:
    result = propagateToR(pars[RADIUS_P]);
    break;
  case Z_DT:
    result = propagateToZ(pars[Z_P]);
    break;
  case PATHL_DT:
    result = propagateByPathLength(pars[PATHL_P]);
    break;
  case PLANE_DT:
    result = propagateToPlane(pars);
    break;
  default:
    break;
  }

  return FAULT;
}
  
SteppingHelixPropagator::Result SteppingHelixPropagator::propagateToR(double rDest, double epsilon) const{
  Result result = OK;
  bool makeNextStep = true;
  double dStep = 1.;
  PropagationDirection dir,oldDir=alongMomentum;
  int nOsc = 0;
  while (makeNextStep){
    int cInd = cIndex_(nPoints_-1);
    double curZ = r3_[cInd].z();
    double curR = r3_[cInd].perp(); 
    double cosDPhiPR = cos((r3_[cInd].deltaPhi(p3_[cInd])));
    double cosecTheta = 1./p3_[cInd].perp()*p3_[cInd].mag();

    dir = ((rDest-curR)*cosDPhiPR > 0 || curR < 2e-1 )? alongMomentum : oppositeToMomentum;
    if (oldDir != dir ) nOsc++;
    if ((fabs(curZ) > 1.5e3 || curR >800.) && dir == alongMomentum ) dStep = fabs((rDest-curR)*cosecTheta) -1e-9;
    if (fabs((rDest-curR)*cosecTheta) < dStep){ //change this to "line distance" at some point
      dStep = fabs((rDest-curR)*cosecTheta); 
    }
    makeAtomStep(nPoints_-1, dStep, dir, HEL_AS_F);
    nPoints_++;  cInd = cIndex_(nPoints_-1);
    oldDir = dir;
    if (nPoints_ > MAX_STEPS || nOsc > 6  || curR > 20000 || fabs(rDest-curR)<fabs(epsilon)  || p3_[cInd].mag() < 0.1){
      makeNextStep = false ;
      if ( nPoints_ > MAX_STEPS || nOsc > 6 ) result = FAULT;
      if (debug_){
	std::cout<<"going to radius "<<rDest<<std::endl;
	std::cout<<"Made "<<nPoints_-1<<" steps and stopped at(prev step) "<<curR<<" now at"<<r3_[cInd].perp()<<std::endl;
      }
    }
  }
  return result;
}

SteppingHelixPropagator::Result SteppingHelixPropagator::propagateToZ(double zDest, double epsilon)  const{
  Result result = OK;
  bool makeNextStep = true;
  double dStep = 1.;
  PropagationDirection dir,oldDir=alongMomentum;
  int nOsc = 0;
  while (makeNextStep){
    int cInd = cIndex_(nPoints_-1);
    double curZ = r3_[cInd].z();
    double curR = r3_[cInd].perp();
    double secTheta = 1./p3_[cInd].z()*p3_[cInd].mag();
    if (debug_){
      std::cout<<" current z "<<curZ<<" and p_z "<<p3_[cInd].z()<<std::endl;
    }
    dir = (zDest - curZ)*p3_[cInd].z() > 0. ? alongMomentum : oppositeToMomentum;    
    if (oldDir != dir ) nOsc++;
    if ((fabs(curZ) > 1.5e3 || curR >800.) && dir == alongMomentum) dStep = fabs((zDest-curZ)*secTheta) -1e-9;
    if (fabs((zDest-curZ)*secTheta) < dStep){
      dStep = fabs((zDest-curZ)*secTheta); 
    }
    makeAtomStep(nPoints_-1, dStep, dir, HEL_AS_F);
    nPoints_++;  cInd = cIndex_(nPoints_-1);
    oldDir = dir;
    if (nPoints_ > MAX_STEPS || nOsc > 6 ||  fabs(curZ) > 20000 || fabs(zDest-curZ)<fabs(epsilon)  || p3_[cInd].mag() < 0.1){
      makeNextStep = false ;
      if ( nPoints_ > MAX_STEPS || nOsc > 6) result = FAULT;
      if (debug_){
	std::cout<<"going to z "<<zDest<<std::endl;
	std::cout<<"Made "<<nPoints_-1<<" steps and stopped at(prev step) "<<curZ<<" now at"<<r3_[cInd].z()<<std::endl;
      }
    }
  }
  return result;
}

SteppingHelixPropagator::Result SteppingHelixPropagator::propagateByPathLength(double sDest, double epsilon) const{
  Result result = OK;
  bool makeNextStep = true;
  double dStep = 1.;
  PropagationDirection dir = sDest > 0 ? alongMomentum : oppositeToMomentum;
  sDest = fabs(sDest);
  double curS = 0;
  while (makeNextStep){
    int cInd = cIndex_(nPoints_-1);
    double curZ = r3_[cInd].z();
    double curR = r3_[cInd].perp();
    if ((fabs(curZ) > 1.5e3 || curR >800.)&& dir == alongMomentum) dStep = fabs(sDest-curS) -1e-9;
    if (fabs(sDest-curS) < dStep){
      dStep = sDest-curS; 
    }
    makeAtomStep(nPoints_-1, dStep, dir, HEL_AS_F);
    curS += dStep;
    nPoints_++;  cInd = cIndex_(nPoints_-1);
    if (nPoints_ > MAX_STEPS || fabs(curS) > 20000 || fabs(sDest-curS)<fabs(epsilon) || p3_[cInd].mag() < 0.1){
      makeNextStep = false ;
      if ( nPoints_ > MAX_STEPS ) result = FAULT;
      if (debug_){
	std::cout<<"going to pathL "<<sDest<<std::endl;
	std::cout<<"Made "<<nPoints_-1<<" steps and stopped at "<<curS<<std::endl;
      }
    }
  }
  return result;
}



SteppingHelixPropagator::Result SteppingHelixPropagator::propagateToPlane(const double pars[6], double epsilon) const{
  Result result = OK;
  bool makeNextStep = true;
  double dStep = 1.;
  PropagationDirection dir,oldDir=alongMomentum;
  int nOsc = 0;
  Point rPlane(pars[0], pars[1], pars[2]);
  Vector nPlane(pars[3], pars[4], pars[5]);
  while (makeNextStep){
    int cInd = cIndex_(nPoints_-1);
    double curZ = r3_[cInd].z();
    double curR = r3_[cInd].perp();
    double dist;
    bool isIncoming;
    refToPlane(nPoints_-1, pars, dist, isIncoming);
    dir = isIncoming ? alongMomentum : oppositeToMomentum;
    if (oldDir != dir ) nOsc++;
    if ((fabs(curZ) > 1.5e3 || curR >800.) && dir == alongMomentum) dStep = dist -1e-9;
    if (dist < dStep){
      dStep = dist; 
    }
    makeAtomStep(nPoints_-1, dStep, dir, HEL_AS_F);
    nPoints_++;   cInd = cIndex_(nPoints_-1);
    oldDir = dir;
    if (nPoints_ > MAX_STEPS || nOsc > 6 || fabs(curZ) > 20000 || dist<fabs(epsilon)  || p3_[cInd].mag() < 0.1){
      makeNextStep = false ;
      if ( nPoints_ > MAX_STEPS || nOsc > 6 ) result = FAULT;
      if (debug_){
	std::cout<<"going to plane r0:"<<rPlane<<" n:"<<nPlane<<std::endl;
	std::cout<<"Made "<<nPoints_-1<<" steps and stopped at(cur step) "<<r3_[cInd]<<std::endl;
      }
    }
  }
  return result;
}


void SteppingHelixPropagator::loadState(int ind, 
					const SteppingHelixPropagator::Vector& p3, const SteppingHelixPropagator::Point& r3, int charge,
					const HepSymMatrix& cov, PropagationDirection dir) const{
  int cInd = cIndex_(ind);
  q_[cInd] = charge;
  p3_[cInd] = p3;
  r3_[cInd] = r3;
  cov_[cInd].assign(cov);
  dir_[cInd] = dir == alongMomentum ? 1.: -1.;

  GlobalVector bf = field_->inTesla(GlobalPoint(r3.x(), r3.y(), r3.z()));
  
  bf_[cInd].set(bf.x(), bf.y(), bf.z());

  setReps(ind);
  getLocBGrad(ind, 1e-1);

  covLoc_[cInd].assign(cov_[cInd]);

  Vector xRep(1., 0., 0.);
  Vector yRep(0., 1., 0.);
  Vector zRep(0., 0., 1.);
  const Vector* repI[3] = {&xRep, &yRep, &zRep};
  const Vector* repF[3] = {&reps_[cInd].lX, &reps_[cInd].lY, &reps_[cInd].lZ};
  initCovRotation(repI, repF, covRot_);
  covLoc_[cInd] = covLoc_[cInd].similarity(covRot_);

  //   std::cout<<"Load at "<<ind<<" path: "<<path_[cInd]
  //  	   <<" p3 "<<" pt: "<<p3_[cInd].perp()<<" phi: "<<p3_[cInd].phi()<<" eta: "<<p3_[cInd].eta()
  // 	   <<" "<<p3_[cInd]
  //  	   <<" r3: "<<r3_[cInd]<<std::endl;
}

void SteppingHelixPropagator::incrementState(int ind, 
					     double dP, SteppingHelixPropagator::Vector tau,
					     double dX, double dY, double dZ, double dS,
					     const HepMatrix& dCovTransform) const{
  //  TimeMe locTimer("SteppingHelixPropagator::incrementState");
  if (ind ==0) return;
  int iPrev = ind-1;
  int cInd = cIndex_(ind);
  int cPrev = cIndex_(iPrev);
  q_[cInd] = q_[cPrev];
  dir_[cInd] = dS > 0. ? 1.: -1.; 
  //  std::cout<<tau.deltaPhi(p3_[cPrev])<<std::endl;
  p3_[cInd] = tau;  p3_[cInd]*=(p3_[cPrev].mag() - dir_[cInd]*fabs(dP));

  r3_[cInd] = r3_[cPrev];
  Vector tmpR3 = reps_[cPrev].lX; tmpR3*=dX;
  r3_[cInd]+= tmpR3;
  tmpR3 = reps_[cPrev].lY; tmpR3*=dY;
  r3_[cInd]+= tmpR3;
  tmpR3 = reps_[cPrev].lZ; tmpR3*=dZ;
  r3_[cInd]+= tmpR3;
  //  cov_[cInd].assign(cov_[cPrev]); // do I need it here?
  path_[cInd] = path_[cPrev] + dS;
  
  GlobalVector bf = field_->inTesla(GlobalPoint(r3_[cInd].x(), r3_[cInd].y(), r3_[cInd].z()));
  
  bf_[cInd].set(bf.x(), bf.y(), bf.z());
  
  setReps(ind);
  getLocBGrad(ind, 1e-1);
  
  const Vector* repI[3] = {&reps_[cPrev].lX, &reps_[cPrev].lY, &reps_[cPrev].lZ};
  const Vector* repF[3] = {&reps_[cInd].lX, &reps_[cInd].lY, &reps_[cInd].lZ};
  initCovRotation(repI, repF, covRot_);
  covRot_ = covRot_*dCovTransform;
  covLoc_[cInd] = covLoc_[cPrev].similarity(covRot_);
  
  if (debug_){
    std::cout<<"Now at "<<ind<<" path: "<<path_[cInd]
	     <<" p3 "<<" pt: "<<p3_[cInd].perp()<<" phi: "<<p3_[cInd].phi()<<" eta: "<<p3_[cInd].eta()
	     <<" "<<p3_[cInd]
	     <<" r3: "<<r3_[cInd]
	     <<" dPhi: "<<acos(p3_[cInd].unit().dot(p3_[cPrev].unit()))
	     <<" bField: "<<bf_[cInd].mag()
	     <<std::endl;
  }
}

void SteppingHelixPropagator::setReps(int ind) const{
  int cInd = cIndex_(ind);
  Vector tau = p3_[cInd]/(p3_[cInd].mag());
  Vector bHat;
  double bMag = bf_[cInd].mag();
  if (bMag < 1e-3){
    bHat.set(0., 0., 1.);
  } else {
    bHat = bf_[cInd]/bMag;
  }
  reps_[cInd].lZ = bHat;
  reps_[cInd].lX = tau.cross(bHat); reps_[cInd].lX/=reps_[cInd].lX.mag();
  reps_[cInd].lY = bHat.cross(reps_[cInd].lX);  reps_[cInd].lY/=reps_[cInd].lY.mag();
}

bool SteppingHelixPropagator::makeAtomStep(int iIn, double dS, PropagationDirection dir, SteppingHelixPropagator::Fancy fancy) const{
  //  TimeMe locTimer("SteppingHelixPropagator::makeAtomStep");
  int cInd = cIndex_(iIn);
  if (debug_){
    std::cout<<"Make atom step "<<iIn<<" with step "<<dS<<" in direction "<<dir<<std::endl;
  }

  //  HepMatrix dCTr(HepDiagMatrix(6,1));//unit transform is the default
  double dP = 0;
  Vector tau = p3_[cInd]; tau/=tau.mag();

  dS = dir == alongMomentum ? fabs(dS) : -fabs(dS);
  double dX =0.;
  double dY =0.;
  double dZ =0.;
  

  double p0 = p3_[cInd].mag();
  double b0 = bf_[cInd].mag();
  double kappa0 = 0.0029979*q_[cInd]*b0/p0;
  if (fabs(kappa0) < 1e-12) kappa0 = 1e-12;

  double cosTheta = reps_[cInd].lZ.dot(tau);
  double sinTheta = sin(acos(cosTheta));
  double cotTheta = fabs(sinTheta) > 1e-12 ? cosTheta/sinTheta : 1e12;
  double tanTheta = fabs(cosTheta) > 1e-12 ? sinTheta/cosTheta : 1e12;
  double phi = kappa0*dS;
  double cosPhi = cos(phi);
  double sinPhi = sin(phi);

  double bfLGL[3];// grad(log(B))
  for (int i = 0; i < 3; i++){
    if (b0 < 1e-6){
      bfLGL[i] = 0.;
    } else {
      bfLGL[i] = bfGradLoc_[cInd][i]; bfLGL[i]/=b0;
    }
  }

  double dEdXPrime = 0;
  double dEdx = getDeDx(iIn, dEdXPrime);
  Vector tmpR3;
  
  switch (fancy){
  case HEL_AS_F:
  case HEL_ALL_F:
    tmpR3 = reps_[cInd].lX; tmpR3*=sinPhi; tmpR3*=sinTheta;
    tau = tmpR3;
    tmpR3 = reps_[cInd].lY;  tmpR3*=cosPhi; tmpR3*=sinTheta;
    tau+=tmpR3;
    tmpR3 = reps_[cInd].lZ;  tmpR3*=cosTheta;
    tau+=tmpR3;
    //the stuff above is
    //    tau = reps_[cInd].lX*sinPhi*sinTheta + reps_[cInd].lY*cosPhi*sinTheta + reps_[cInd].lZ*cosTheta;

    dP = dEdx*dS;
    dX = (1. - cosPhi)/kappa0*sinTheta;
    dY = sinPhi/kappa0*sinTheta;
    dZ = dS*cosTheta;

    dCTransform_ = unit66_;
    //     //yuck
    //     dCTr(1,1) = 0.;
    //     dCTr(1,2) = sinPhi;
    //     dCTr(1,3) = -cotTheta*sinPhi;
    //     dCTr(1,4) = sinTheta*phi*cosPhi*bfLGL[0];
    //     dCTr(1,5) = sinTheta*phi*cosPhi*bfLGL[1];
    //     dCTr(1,6) = sinTheta*phi*cosPhi*bfLGL[2];
    //     //    dCTr(1,7) = -sinTheta*phi*cosPhi/p0;
    
    //     dCTr(2,1) = 0.;
    //     dCTr(2,2) = cosPhi;
    //     dCTr(2,3) = -cotTheta*cosPhi;
    //     dCTr(2,4) = -sinTheta*phi*sinPhi*bfLGL[0];
    //     dCTr(2,5) = -sinTheta*phi*sinPhi*bfLGL[1];
    //     dCTr(2,6) = -sinTheta*phi*sinPhi*bfLGL[2];
    //     //    dCTr(2,7) =  sinTheta*phi*sinPhi/p0;
    
    //     dCTr(3,1) = 0.;
    //     dCTr(3,2) = -tanTheta;
    //     dCTr(3,3) = 1.;
    //     dCTr(3,4) = 0.;
    //     dCTr(3,5) = 0.;
    //     dCTr(3,6) = 0.;
    //     //    dCTr(3,7) = 0.;
    
    //     dCTr(4,1) = 0.;
    //     dCTr(4,2) = (1.-cosPhi)/kappa0;
    //     dCTr(4,3) = -cotTheta*(1.-cosPhi)/kappa0;
    //     dCTr(4,4) = 1.+sinTheta*(phi*sinPhi - 1. + cosPhi)/kappa0*bfLGL[0];
    //     dCTr(4,5) =    sinTheta*(phi*sinPhi - 1. + cosPhi)/kappa0*bfLGL[1];
    //     dCTr(4,6) =    sinTheta*(phi*sinPhi - 1. + cosPhi)/kappa0*bfLGL[2];
    //     //    dCTr(4,7) =   -sinTheta*(phi*sinPhi - 1. + cosPhi)/kappa0/p0;
    
    //     dCTr(5,1) = 0.;
    //     dCTr(5,2) = sinPhi/kappa0;
    //     dCTr(5,3) = -cotTheta*sinPhi/kappa0;
    //     dCTr(5,4) =    sinTheta*(phi*cosPhi - sinPhi)/kappa0*bfLGL[0];
    //     dCTr(5,5) = 1+ sinTheta*(phi*cosPhi - sinPhi)/kappa0*bfLGL[1];
    //     dCTr(5,6) =    sinTheta*(phi*cosPhi - sinPhi)/kappa0*bfLGL[2];
    //     //    dCTr(5,7) =   -sinTheta*(phi*cosPhi - sinPhi)/kappa0/p0;
    
    //     dCTr(6,1) = 0.;
    //     dCTr(6,2) = -dS*tanTheta;
    //     dCTr(6,3) = dS;
    //     dCTr(6,4) = 0.;
    //     dCTr(6,5) = 0.;
    //     dCTr(6,6) = 1.;
    //     //    dCTr(6,7) = 0.;
    
    break;
  case POL_1_F:
  case POL_2_F:
  case POL_M_F:
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
  incrementState(iIn+1, dP, tau, dX, dY, dZ, dS, dCTransform_);
  return true;
}

double SteppingHelixPropagator::getDeDx(int iIn, double& dEdXPrime) const{
  if (noMaterialMode_) return 0;
  int cInd = cIndex_(iIn);

  double dEdx = 0.;
  double lR = r3_[cInd].perp();
  double lZ = fabs(r3_[cInd].z());

  //assume "Iron" .. seems to be quite the same for brass/iron/PbW04
  //will do proper Bethe-Bloch later
  double p0 = p3_[cInd].mag();
  double dEdX_mat = -(11.4 + 2.4*fabs(log(p0*2.8)))*1e-3*0.935; //in GeV/cm .. 0.98 to get closer to the median or MPV
  double dEdX_HCal = 0.805*dEdX_mat; //extracted from sim
  double dEdX_ECal = 0.39*dEdX_mat;
  double dEdX_coil = 0.298*dEdX_mat; //extracted from sim
  double dEdX_Fe =   dEdX_mat;
  double dEdX_MCh =  0.053*dEdX_mat; //chambers on average
  double dEdX_Trk =  0.0097*dEdX_mat;


  //this should roughly figure out where things are (numbers taken from Fig1.1.2 TDR)
  if (lR < 129){
    if (lZ < 294) dEdx = dEdx = dEdX_Trk;
    else if (lZ < 390 ) dEdx = dEdX_ECal*0.8; //averaged out over a larger space
    else if (lZ < 568 ) dEdx = dEdX_HCal; //endcap calor
    else dEdx = dEdX_Fe; //iron .. don't care about no material in front of HF (too forward)
  }
  else if (lR < 285 ){
    if (lZ < 390 && lR < 181 ) dEdx = dEdX_ECal;
    else if (lZ < 568 ) dEdx = dEdX_HCal; //hcal
    else if (lZ < 625) dEdx = dEdX_MCh;
    else if (lZ < 785) dEdx = dEdX_Fe;//iron
    else if (lZ < 850) dEdx = dEdX_MCh;
    else if (lZ < 910) dEdx = dEdX_Fe; //iron
    else if (lZ < 975) dEdx = dEdX_MCh;
    else if (lZ < 1000) dEdx = dEdX_Fe; //iron
    else dEdx = 0;
  }
  else if (lR <380 && lZ < 645 ) dEdx = dEdX_coil;//a guess for the solenoid average
  else {
    if (lZ < 667) {
      if (bf_[cInd].mag()> 0.75) dEdx = dEdX_Fe; //iron
      else dEdx = dEdX_MCh;
    } 
    else if (lZ < 724) dEdx = dEdX_MCh;
    else if (lZ < 785) dEdx = dEdX_Fe;//iron
    else if (lZ < 850) dEdx = dEdX_MCh;
    else if (lZ < 910) dEdx = dEdX_Fe; //iron
    else if (lZ < 975) dEdx = dEdX_MCh;
    else if (lZ < 1000) dEdx = dEdX_Fe; //iron
    else dEdx = 0; //air
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

void SteppingHelixPropagator::refToPlane(int ind, const double pars[6], double& dist, bool& isIncoming) const{
  int cInd = cIndex_(ind);
  Point rPlane(pars[0], pars[1], pars[2]);
  Vector nPlane(pars[3], pars[4], pars[5]);

  double dRDotN = (r3_[cInd] - rPlane).dot(nPlane);

  dist = fabs(dRDotN);
  isIncoming = (p3_[cInd].dot(nPlane))*dRDotN < 0.;

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

