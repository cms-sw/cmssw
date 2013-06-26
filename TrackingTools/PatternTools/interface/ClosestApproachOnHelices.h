#ifndef _ClosestApproachOnHelices_H_
#define _ClosestApproachOnHelices_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include <utility>

/** \class ClosestApproachOnHelices
 *  Abstract interface for classes which compute the points of closest 
 *  approach of 2 helices. <br>
 * For a pair of states, the calculate methods have to be called before any of
 * the other methods. This will do all calculations needed to get the result.
 * It returns a status which says whether the calculation was successful.
 * This should be checked before getting the result. If the status is false 
 * and the results querried, an exception will be thrown. 
 */

class FreeTrajectoryState;
class TrajectoryStateOnSurface;

class ClosestApproachOnHelices {

public:

  ClosestApproachOnHelices() {}

  virtual ~ClosestApproachOnHelices() {}


  virtual bool calculate(const TrajectoryStateOnSurface & sta, 
	 const TrajectoryStateOnSurface & stb) = 0;

  virtual bool calculate(const FreeTrajectoryState & sta,
	const FreeTrajectoryState & stb) = 0;

  virtual bool status() const = 0;

  /** Points of closest approach on the 2 helices */
  virtual std::pair<GlobalPoint, GlobalPoint> points() const = 0;

  /** Crossing point of the 2 helices, computed as an average 
   *  of the points of closest approach. 
   *  The average can be weighted or not, depending on the implementation. 
   */
  virtual GlobalPoint crossingPoint() const = 0;

  /** Distance between the points of closest approach */
  virtual float distance() const = 0;

  virtual ClosestApproachOnHelices * clone() const = 0;

};

#endif
