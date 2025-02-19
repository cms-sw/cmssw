/** \class SteppingHelixStateInfo
 *  Implementation part of the stepping helix propagator state data structure
 *
 *  $Date: 2009/09/08 20:44:32 $
 *  $Revision: 1.13 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Wed Jan  3 16:01:24 CST 2007
// $Id: SteppingHelixStateInfo.cc,v 1.13 2009/09/08 20:44:32 slava77 Exp $
//
//

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  path_(0), radPath_(0), dir(0), magVol(0), field(0), dEdx(0), dEdXPrime(0), radX0(1e12),
  status_(UNDEFINED)
{
  p3.set(fts.momentum().x(), fts.momentum().y(), fts.momentum().z());
  r3.set(fts.position().x(), fts.position().y(), fts.position().z());
  q = fts.charge();

  if (fts.hasError()){
    covCurv = fts.curvilinearError().matrix();
    hasErrorPropagated_ = true;
  } else {
    covCurv = AlgebraicSymMatrix55();
    hasErrorPropagated_ = false;
  }
  static const std::string metname = "SteppingHelixPropagator";
  if (fts.hasError()){ 
    LogTrace(metname)<<"Created SHPStateInfo from FTS\n"<<fts;
    //    LogTrace(metname)<<"and cartesian error of\n"<<fts.cartesianError().matrix();
  }
  else LogTrace(metname)<<"Created SHPStateInfo from FTS without errors";

  isComplete = false;
  isValid_ = true;
}

TrajectoryStateOnSurface SteppingHelixStateInfo::getStateOnSurface(const Surface& surf, bool returnTangentPlane) const {
  static const std::string metname = "SteppingHelixPropagator";
  if (! isValid()) LogTrace(metname)<<"Return TSOS is invalid";
  else LogTrace(metname)<<"Return TSOS is valid";
  if (! isValid()) return TrajectoryStateOnSurface();
  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, q, field);
  //  CartesianTrajectoryError tCov(cov);

  //  CurvilinearTrajectoryError tCCov(ROOT::Math::Similarity(JacobianCartesianToCurvilinear(tPars).jacobian(), cov));
  CurvilinearTrajectoryError tCCov(covCurv);

  FreeTrajectoryState fts(tPars, tCCov);
  if (! hasErrorPropagated_) fts = FreeTrajectoryState(tPars);


  SurfaceSideDefinition::SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface;
  if ( dir > 0 ) side =  SurfaceSideDefinition::beforeSurface;
  if ( dir < 0 ) side =  SurfaceSideDefinition::afterSurface;
  return TrajectoryStateOnSurface(fts, returnTangentPlane ? *surf.tangentPlane(fts.position()) : surf, side);
}


void SteppingHelixStateInfo::getFreeState(FreeTrajectoryState& fts) const {
  if (isValid()){
    GlobalVector p3GV(p3.x(), p3.y(), p3.z());
    GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
    GlobalTrajectoryParameters tPars(r3GP, p3GV, q, field);
    //    CartesianTrajectoryError tCov(cov);
    //    CurvilinearTrajectoryError tCCov(ROOT::Math::Similarity(JacobianCartesianToCurvilinear(tPars).jacobian(), cov));
    CurvilinearTrajectoryError tCCov(covCurv);
    
    fts = (hasErrorPropagated_ ) 
      ? FreeTrajectoryState(tPars, tCCov) : FreeTrajectoryState(tPars);
    //      ? FreeTrajectoryState(tPars, tCov, tCCov) : FreeTrajectoryState(tPars);
    //    if (fts.hasError()) fts.curvilinearError(); //call it so it gets created
  }
}
