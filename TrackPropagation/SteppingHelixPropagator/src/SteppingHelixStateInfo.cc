/** \class SteppingHelixStateInfo
 *  Implementation part of the stepping helix propagator state data structure
 *
 *  $Date: 2007/02/14 10:19:36 $
 *  $Revision: 1.4 $
 *  \author Vyacheslav Krutelyov (slava77)
 */

//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Wed Jan  3 16:01:24 CST 2007
// $Id: SteppingHelixStateInfo.cc,v 1.4 2007/02/14 10:19:36 slava77 Exp $
//
//

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "DataFormats/GeometrySurface/interface/TangentPlane.h"

const std::string SteppingHelixStateInfo::ResultName[MAX_RESULT] = {
  "RESULT_OK",
  "RESULT_FAULT",
  "RESULT_RANGEOUT",
  "RESULT_INACC",
  "RESULT_NOT_IMPLEMENTED",
  "RESULT_UNDEFINED"
};

SteppingHelixStateInfo::SteppingHelixStateInfo(const FreeTrajectoryState& fts){
  p3.set(fts.momentum().x(), fts.momentum().y(), fts.momentum().z());
  r3.set(fts.position().x(), fts.position().y(), fts.position().z());
  q = fts.charge();

  if (fts.hasError()){
    cov = fts.cartesianError().matrix();
    hasErrorPropagated_ = true;
  }else{
    cov = HepSymMatrix(1, 0);
    hasErrorPropagated_ = false;
  }

  isComplete = false;
  isValid_ = true;
}

TrajectoryStateOnSurface SteppingHelixStateInfo::getStateOnSurface(const Surface& surf, bool returnTangentPlane) const {
  if (! isValid()) return TrajectoryStateOnSurface();
  FreeTrajectoryState fts;
  getFreeState(fts);

  return TrajectoryStateOnSurface(fts, surf);
}


void SteppingHelixStateInfo::getFreeState(FreeTrajectoryState& fts) const {
  if (isValid()){
    GlobalVector p3GV(p3.x(), p3.y(), p3.z());
    GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
    GlobalTrajectoryParameters tPars(r3GP, p3GV, q, field);
    CartesianTrajectoryError tCov(cov);
    
    fts = (cov.num_row() == 6 ) 
      ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars);
    if (fts.hasError()) fts.curvilinearError(); //call it so it gets created
  }
}
