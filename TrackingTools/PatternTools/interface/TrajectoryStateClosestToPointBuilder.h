#ifndef TrajectoryStateClosestToPointBuilder_H
#define TrajectoryStateClosestToPointBuilder_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

/**
 * This class builds a TrajectoryStateClosestToPoint given an original 
 * TrajectoryStateOnSurface or FreeTrajectoryState. This new state is then 
 * defined at the point of closest approach to the reference point.
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

  bool positionEqual(const GlobalPoint& ptB, const GlobalPoint& ptA) const 
  {
    if ((ptA.x() == ptB.x()) && (ptA.y() == ptB.y()) && (ptA.z() == ptB.z()))
      return true;
    else return false;
  }

protected:

  TrajectoryStateClosestToPoint constructTSCP(const FTS& originalFTS, 
    const GlobalPoint& referencePoint) const
    {return TrajectoryStateClosestToPoint(originalFTS, referencePoint);}


};
#endif
