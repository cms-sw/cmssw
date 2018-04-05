#ifndef MinPtTrajectoryFilter_H
#define MinPtTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"

/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */

class MinPtTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit MinPtTrajectoryFilter( float ptMin, float nSigma = 5.F, int nH=3): 
    thePtMin2(ptMin*ptMin),theInvPtMin(1.f/ptMin), theNSigma(nSigma), theMinHits(nH)  {}


  explicit MinPtTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC) :
    thePtMin2(pset.getParameter<double>("minPt")),
    theInvPtMin(1.f/thePtMin2),
    theNSigma(pset.getParameter<double>("nSigmaMinPt")),
    theMinHits(pset.getParameter<int>("minHitsMinPt")){thePtMin2*=thePtMin2;}
    

  bool qualityFilter( const Trajectory& traj)const override { return test(traj.lastMeasurement(),traj.foundHits()); }
  bool qualityFilter( const TempTrajectory& traj) const override { return test(traj.lastMeasurement(),traj.foundHits()); }
    
  bool toBeContinued( Trajectory& traj) const override {
    bool ret = test(traj.lastMeasurement(),traj.foundHits());
    if (!ret) traj.setStopReason(StopReason::MIN_PT);
    return  ret;
  }
  bool toBeContinued( TempTrajectory& traj) const override {
    bool ret = test(traj.lastMeasurement(),traj.foundHits());
    if (!ret) traj.setStopReason(StopReason::MIN_PT);
    return ret;
  }
  
  std::string name() const override {return "MinPtTrajectoryFilter";}

 protected:

  bool test( const TrajectoryMeasurement & tm, int foundHits) const; 

  float thePtMin2; 
  float theInvPtMin;
  float theNSigma;
  int theMinHits;

};

#endif
