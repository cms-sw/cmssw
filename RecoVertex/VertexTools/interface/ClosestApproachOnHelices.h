#ifndef _ClosestApproachOnHelices_H_
#define _ClosestApproachOnHelices_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <utility>

using namespace std;

/** \class ClosestApproachOnHelices
 *  Abstract interface for classes which compute the points of closest 
 *  approach of 2 helices. 
 */

class FreeTrajectoryState;
class TrajectoryStateOnSurface;

class ClosestApproachOnHelices {

public:

  ClosestApproachOnHelices() {}

  virtual ~ClosestApproachOnHelices() {}

  /** Points of closest approach on the 2 helices */
  virtual pair<GlobalPoint, GlobalPoint> 
  points(const TrajectoryStateOnSurface & sta, 
	 const TrajectoryStateOnSurface & stb) 
  const = 0;

  virtual pair<GlobalPoint, GlobalPoint> 
  points(const FreeTrajectoryState & sta, const FreeTrajectoryState & stb) 
  const = 0;

  /** Crossing point of the 2 helices, computed as an average 
   *  of the points of closest approach. 
   *  The average can be weighted or not, depending on the implementation. 
   */
  virtual GlobalPoint crossingPoint(const TrajectoryStateOnSurface & sta, 
				    const TrajectoryStateOnSurface & stb) 
    const = 0;

  virtual GlobalPoint crossingPoint(const FreeTrajectoryState & sta, 
				    const FreeTrajectoryState & stb) const = 0;

  /** Distance between the points of closest approach */
  virtual float distance(const TrajectoryStateOnSurface & sta, 
			 const TrajectoryStateOnSurface & stb) const = 0;

  virtual float distance(const FreeTrajectoryState & sta, 
			 const FreeTrajectoryState & stb) const = 0;

  virtual ClosestApproachOnHelices * clone() const = 0;

};

#endif
