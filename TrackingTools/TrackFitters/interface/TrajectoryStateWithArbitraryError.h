#ifndef CD_TrajectoryStateWithArbitraryError_H_
#define CD_TrajectoryStateWithArbitraryError_H_

/** \class TrajectoryStateWithArbitraryError
 *  Creates a TrajectoryState with the same parameters as the inlut one,
 *  but with "infinite" errors, i.e. errors so big that they don't
 *  bias a fit starting with this state. Ported from ORCA
 *
 *  $Date: 2007/05/09 14:17:57 $
 *  $Revision: 1.2 $
 *  \author todorov, cerati
 */

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class TrajectoryStateWithArbitraryError {

private:

  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;

public:
  
  TSOS operator()(const TSOS& aTsos) const;

};

#endif //CD_TrajectoryStateWithArbitraryError_H_
