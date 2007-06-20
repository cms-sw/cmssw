/** \class SteppingHelixStateInfo
 *  Implementation part of the stepping helix propagator state data structure
 *
 *  $Date: 2007/05/11 04:51:11 $
 *  $Revision: 1.8 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Wed Jan  3 16:01:24 CST 2007
// $Id: SteppingHelixStateInfo.cc,v 1.8 2007/05/11 04:51:11 slava77 Exp $
//
//

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"

#include "DataFormats/GeometrySurface/interface/TangentPlane.h"

const std::string SteppingHelixStateInfo::ResultName[MAX_RESULT] = {
  "RESULT_OK",
  "RESULT_FAULT",
  "RESULT_RANGEOUT",
  "RESULT_INACC",
  "RESULT_NOT_IMPLEMENTED",
  "RESULT_UNDEFINED"
};

SteppingHelixStateInfo::SteppingHelixStateInfo(const FreeTrajectoryState& fts): 
  path_(0), radPath(0), dir(0), magVol(0), field(0), dEdx(0), dEdXPrime(0), radX0(1e12),
  status_(UNDEFINED)
{
  p3.set(fts.momentum().x(), fts.momentum().y(), fts.momentum().z());
  r3.set(fts.position().x(), fts.position().y(), fts.position().z());
  q = fts.charge();

  if (fts.hasError()){
    cov = fts.cartesianError().matrix();
    hasErrorPropagated_ = true;
  } else {
    cov = AlgebraicSymMatrix66();
    hasErrorPropagated_ = false;
  }

  isComplete = false;
  isValid_ = true;
}

TrajectoryStateOnSurface SteppingHelixStateInfo::getStateOnSurface(const Surface& surf, bool returnTangentPlane) const {
  if (! isValid()) return TrajectoryStateOnSurface();
  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, q, field);
  CartesianTrajectoryError tCov(cov);

  CurvilinearTrajectoryError tCCov(ROOT::Math::Similarity(JacobianCartesianToCurvilinear(tPars).jacobian(), cov));

  FreeTrajectoryState fts(tPars, tCov, tCCov);
  if (cov.kRows != 6) fts = FreeTrajectoryState(tPars);
  //  if (fts.hasError()) fts.curvilinearError(); //call it so it gets created
  //equivalent to  getFreeState(fts);

  return TrajectoryStateOnSurface(fts, returnTangentPlane ? *surf.tangentPlane(fts.position()) : surf);
}


void SteppingHelixStateInfo::getFreeState(FreeTrajectoryState& fts) const {
  if (isValid()){
    GlobalVector p3GV(p3.x(), p3.y(), p3.z());
    GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
    GlobalTrajectoryParameters tPars(r3GP, p3GV, q, field);
    CartesianTrajectoryError tCov(cov);
    
    fts = (cov.kRows == 6 ) 
      ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars);
    if (fts.hasError()) fts.curvilinearError(); //call it so it gets created
  }
}
