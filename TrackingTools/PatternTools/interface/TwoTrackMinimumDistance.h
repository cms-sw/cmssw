#ifndef _TwoTrackMinimumDistance_H_
#define _TwoTrackMinimumDistance_H_

#include "TrackingTools/PatternTools/interface/ClosestApproachOnHelices.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixHelix.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceLineLine.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixLine.h"
#include "FWCore/Utilities/interface/Visibility.h"

/**
   * General interface to calculate the PCA of two tracks. 
   * According to the charge of the tracks, the correct algorithm is used:<ul>
   * <li> charged-charged: TwoTrackMinimumDistanceHelixHelix
   * <li> charged-neutral: TwoTrackMinimumDistanceHelixLine
   * <li> neutral-neutral: TwoTrackMinimumDistanceLineLine
   * </ul>
   */

class TwoTrackMinimumDistance final : public ClosestApproachOnHelices {
public:
  enum Mode { FastMode = 0, SlowMode = 1 };

  TwoTrackMinimumDistance(const Mode m = FastMode) {
    theModus = m;
    status_ = false;
  };
  ~TwoTrackMinimumDistance() override {}

  bool calculate(const TrajectoryStateOnSurface& sta, const TrajectoryStateOnSurface& stb) override;

  bool calculate(const FreeTrajectoryState& sta, const FreeTrajectoryState& stb) override;

  virtual bool calculate(const GlobalTrajectoryParameters& sta, const GlobalTrajectoryParameters& stb);

  bool status() const override { return status_; }

  /**
   * Returns the two PCA on the trajectories.
   */

  std::pair<GlobalPoint, GlobalPoint> points() const override;

  /** arithmetic mean of the two points of closest approach */
  GlobalPoint crossingPoint() const override;

  /** distance between the two points of closest approach in 3D */
  float distance() const override;

  /**
   *  Clone method
   */
  TwoTrackMinimumDistance* clone() const override { return new TwoTrackMinimumDistance(*this); }

  double firstAngle() const;
  double secondAngle() const;
  std::pair<double, double> pathLength() const;

private:
  enum Charge { hh, hl, ll };
  Mode theModus;
  Charge theCharge;
  ClosestApproachInRPhi theIniAlgo;
  TwoTrackMinimumDistanceHelixHelix theTTMDhh;
  TwoTrackMinimumDistanceLineLine theTTMDll;
  TwoTrackMinimumDistanceHelixLine theTTMDhl;
  bool status_;
  std::pair<GlobalPoint, GlobalPoint> points_;

  bool pointsLineLine(const GlobalTrajectoryParameters& sta, const GlobalTrajectoryParameters& stb) dso_internal;
  bool pointsHelixLine(const GlobalTrajectoryParameters& sta, const GlobalTrajectoryParameters& stb) dso_internal;
  bool pointsHelixHelix(const GlobalTrajectoryParameters& sta, const GlobalTrajectoryParameters& stb) dso_internal;
};

#endif
