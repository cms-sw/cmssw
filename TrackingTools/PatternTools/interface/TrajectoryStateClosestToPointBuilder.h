#ifndef TrajectoryStateClosestToPointBuilder_H
#define TrajectoryStateClosestToPointBuilder_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

/**
 * This class builds a TrajectoryStateClosestToPoint given an original 
 * TrajectoryStateOnSurface or FreeTrajectoryState. This new state is then 
 * defined at the point of closest approach to the reference point.
 * In case the propagation was not successful, this state can be invalid.
 */

class TrajectoryStateClosestToPointBuilder
{
public: 
  typedef TrajectoryStateOnSurface	TSOS;
  typedef FreeTrajectoryState		FTS;

  virtual ~TrajectoryStateClosestToPointBuilder(){}

  virtual TrajectoryStateClosestToPoint operator() (const FTS& originalFTS, 
    const GlobalPoint& referencePoint) const = 0;

  virtual TrajectoryStateClosestToPoint operator() (const TSOS& originalTSOS, 
    const GlobalPoint& referencePoint) const = 0;

  static bool positionEqual(const GlobalPoint& ptB, const GlobalPoint& ptA) {
    return ptA==ptB;
  }

protected:

  static TrajectoryStateClosestToPoint constructTSCP(const FTS& originalFTS, 
    const GlobalPoint& referencePoint)
    {return TrajectoryStateClosestToPoint(originalFTS, referencePoint);}


};
#endif
