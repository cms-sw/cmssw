#ifndef ThresholdPtTrajectoryFilter_H
#define ThresholdPtTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"

/** A TrajectoryFilter that stops reconstruction if P_t goes 
 *  above some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the conditional p_T cut
 */

class ThresholdPtTrajectoryFilter : public TrajectoryFilter {
public:

  explicit ThresholdPtTrajectoryFilter( double ptThreshold, float nSigma = 5.F, int nH=3): thePtThreshold( ptThreshold), theNSigma(nSigma), theMinHits(nH) {}

  explicit ThresholdPtTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC) :
    thePtThreshold(pset.getParameter<double>("thresholdPt")),
    theNSigma(pset.getParameter<double>("nSigmaThresholdPt")),
    theMinHits(pset.getParameter<int>("minHitsThresholdPt"))
      {}

  bool qualityFilter( const Trajectory& traj) const override { return !test(traj.lastMeasurement(),traj.foundHits());}
  bool qualityFilter( const TempTrajectory& traj) const override { return !test(traj.lastMeasurement(),traj.foundHits());}
   
  bool toBeContinued( Trajectory& traj) const override { return test(traj.lastMeasurement(),traj.foundHits()); }
  bool toBeContinued( TempTrajectory& traj) const override { return test(traj.lastMeasurement(),traj.foundHits()); }
  
  std::string name() const override {return "ThresholdPtTrajectoryFilter";}

 protected:

  bool test( const TrajectoryMeasurement & tm, int foundHits) const;

  double thePtThreshold;
  double theNSigma;
  int theMinHits;

};

#endif
