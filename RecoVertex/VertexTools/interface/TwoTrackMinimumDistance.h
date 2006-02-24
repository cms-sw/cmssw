#ifndef _TwoTrackMinimumDistance_H_
#define _TwoTrackMinimumDistance_H_

#include "RecoVertex/VertexTools/interface/ClosestApproachOnHelices.h"
#include "RecoVertex/VertexTools/interface/ClosestApproachInRPhi.h"
#include "RecoVertex/VertexTools/src/TwoTrackMinimumDistanceHelixHelix.h"
#include "RecoVertex/VertexTools/src/TwoTrackMinimumDistanceLineLine.h"
#include "RecoVertex/VertexTools/src/TwoTrackMinimumDistanceHelixLine.h"

using namespace std;

  /**
   * General interface to calculate the PCA of two tracks. 
   * According to the charge of the tracks, the correct algorithm is used:<ul>
   * <li> charged-charged: TwoTrackMinimumDistanceHelixHelix
   * <li> charged-neutral: TwoTrackMinimumDistanceHelixLine
   * <li> neutral-neutral: TwoTrackMinimumDistanceLineLine
   * </ul>
   */

class TwoTrackMinimumDistance : public ClosestApproachOnHelices {

public:

  enum Mode { FastMode=0, SlowMode=1 };

  TwoTrackMinimumDistance( const Mode m=FastMode ) { theModus=m; };

  /**
   * Returns the two PCA on the trajectories.
   */
  virtual pair<GlobalPoint, GlobalPoint>
  points(const TrajectoryStateOnSurface & sta,
         const TrajectoryStateOnSurface & stb) const;

  /**
   * Returns the two PCA on the trajectories.
   */
  virtual pair<GlobalPoint, GlobalPoint>
  points(const FreeTrajectoryState & sta,
         const FreeTrajectoryState & stb) const;



  /** arithmetic mean of the two points of closest approach */
  virtual GlobalPoint crossingPoint(const TrajectoryStateOnSurface & sta,
                                    const TrajectoryStateOnSurface & stb) const;

  /** arithmetic mean of the two points of closest approach */
  virtual GlobalPoint crossingPoint(const FreeTrajectoryState & sta,
                                    const FreeTrajectoryState & stb) const;

  /** distance between the two points of closest approach in 3D.
   */
  virtual float distance(const TrajectoryStateOnSurface & sta,
                         const TrajectoryStateOnSurface & stb) const;

  virtual float distance(const FreeTrajectoryState & sta,
                         const FreeTrajectoryState & stb) const;

  /**
   *  Clone method
   */
  virtual TwoTrackMinimumDistance * clone() const {
    return new TwoTrackMinimumDistance(* this);
  }

  double firstAngle() const;
  double secondAngle() const;

private:
  enum Charge { hh, hl, ll };
  Mode theModus;
  mutable Charge theCharge;
  ClosestApproachInRPhi theIniAlgo;
  mutable TwoTrackMinimumDistanceHelixHelix theTTMDhh;
  mutable TwoTrackMinimumDistanceLineLine theTTMDll;
  mutable TwoTrackMinimumDistanceHelixLine theTTMDhl;

  pair<GlobalPoint, GlobalPoint> points(const GlobalTrajectoryParameters & sta,
                                const GlobalTrajectoryParameters & stb) const;
  pair<GlobalPoint, GlobalPoint> pointsLineLine(const GlobalTrajectoryParameters & sta,
                                const GlobalTrajectoryParameters & stb) const;
  pair<GlobalPoint, GlobalPoint> pointsHelixLine(const GlobalTrajectoryParameters & sta,
                                const GlobalTrajectoryParameters & stb) const;
  pair<GlobalPoint, GlobalPoint> pointsHelixHelix(const GlobalTrajectoryParameters & sta,
                                const GlobalTrajectoryParameters & stb) const;
};

#endif
