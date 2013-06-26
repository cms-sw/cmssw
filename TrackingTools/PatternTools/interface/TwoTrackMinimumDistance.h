#ifndef _TwoTrackMinimumDistance_H_
#define _TwoTrackMinimumDistance_H_

#include "TrackingTools/PatternTools/interface/ClosestApproachOnHelices.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixHelix.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceLineLine.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixLine.h"

  /**
   * General interface to calculate the PCA of two tracks. 
   * According to the charge of the tracks, the correct algorithm is used:<ul>
   * <li> charged-charged: TwoTrackMinimumDistanceHelixHelix
   * <li> charged-neutral: TwoTrackMinimumDistanceHelixLine
   * <li> neutral-neutral: TwoTrackMinimumDistanceLineLine
   * </ul>
   */

class TwoTrackMinimumDistance GCC11_FINAL : public ClosestApproachOnHelices {

public:

  enum Mode { FastMode=0, SlowMode=1 };

  TwoTrackMinimumDistance( const Mode m=FastMode ) { theModus=m; status_ = false;};
  ~TwoTrackMinimumDistance(){}

  virtual bool calculate(const TrajectoryStateOnSurface & sta, 
	 const TrajectoryStateOnSurface & stb);

  virtual bool calculate(const FreeTrajectoryState & sta,
	const FreeTrajectoryState & stb);

  virtual bool calculate(const GlobalTrajectoryParameters & sta,
	const GlobalTrajectoryParameters & stb);

  virtual bool status() const {return status_;}

  /**
   * Returns the two PCA on the trajectories.
   */

  virtual std::pair<GlobalPoint, GlobalPoint> points() const;

  /** arithmetic mean of the two points of closest approach */
  virtual GlobalPoint crossingPoint() const;

  /** distance between the two points of closest approach in 3D */
  virtual float distance() const;


  /**
   *  Clone method
   */
  virtual TwoTrackMinimumDistance * clone() const {
    return new TwoTrackMinimumDistance(* this);
  }

  double firstAngle() const;
  double secondAngle() const;
  std::pair <double, double> pathLength() const;

private:
  enum Charge { hh, hl, ll };
  Mode theModus;
  mutable Charge theCharge;
  ClosestApproachInRPhi theIniAlgo;
  mutable TwoTrackMinimumDistanceHelixHelix theTTMDhh;
  mutable TwoTrackMinimumDistanceLineLine theTTMDll;
  mutable TwoTrackMinimumDistanceHelixLine theTTMDhl;
  bool status_;
  std::pair<GlobalPoint, GlobalPoint> points_;

  bool pointsLineLine(const GlobalTrajectoryParameters & sta,
	const GlobalTrajectoryParameters & stb)  dso_internal;
  bool pointsHelixLine(const GlobalTrajectoryParameters & sta,
	const GlobalTrajectoryParameters & stb)  dso_internal;
  bool pointsHelixHelix(const GlobalTrajectoryParameters & sta,
	const GlobalTrajectoryParameters & stb)  dso_internal;
};

#endif
