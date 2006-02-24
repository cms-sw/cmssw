#ifndef TrajectoryStateCombiner_H
#define TrajectoryStateCombiner_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

/** Combines the information from two trajectory states via a weighted mean.
 *  The input states should not be correlated.
 */

class TrajectoryStateCombiner {
public:

  typedef TrajectoryStateOnSurface    TSOS;

  TSOS combine(const TSOS& pTsos1, const TSOS& pTsos2) const;

  TSOS operator()(const TSOS& pTsos1, const TSOS& pTsos2) const {
    return combine( pTsos1, pTsos2);
  }
  
};

#endif
