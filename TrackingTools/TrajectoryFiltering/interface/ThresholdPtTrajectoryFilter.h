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

  virtual bool qualityFilter( const Trajectory& traj) const { return !test(traj.lastMeasurement(),traj.foundHits());}
  virtual bool qualityFilter( const TempTrajectory& traj) const { return !test(traj.lastMeasurement(),traj.foundHits());}
   
  virtual bool toBeContinued( Trajectory& traj) const { return test(traj.lastMeasurement(),traj.foundHits()); }
  virtual bool toBeContinued( TempTrajectory& traj) const { return test(traj.lastMeasurement(),traj.foundHits()); }
  
  virtual std::string name() const {return "ThresholdPtTrajectoryFilter";}

 protected:

  bool test( const TrajectoryMeasurement & tm, int foundHits) const 
  {
    //first check min number of hits 
    if (foundHits < theMinHits ){ return true;}

    // check for momentum below limit
    const FreeTrajectoryState& fts = *tm.updatedState().freeTrajectoryState();

    //avoid doing twice the check in TBC and QF
    // We make it thread local so that we avoid race conditions between
    // threads, and we make sure there is no cache contention between them.
    static thread_local bool answerMemory=false;
    static thread_local FreeTrajectoryState ftsMemory;
    if (ftsMemory.parameters().vector() == fts.parameters().vector()) { return answerMemory;}
    ftsMemory=fts;

    //if p_T is way too small: stop
    double pT = fts.momentum().perp();
    if (pT<0.010) {answerMemory=false; return false;}
    //if error is way too big: stop
    double invError = TrajectoryStateAccessor(fts).inversePtError();
    if (invError > 1.e10) {answerMemory=false;return false;}

    //calculate the actual pT cut: 
    if ((1/pT + theNSigma*invError ) < 1/thePtThreshold ) {answerMemory=false; return false;}
    //    first term is the minimal value of pT (pT-N*sigma(pT))
    //    secon term is the cut

    answerMemory=true; return true;
  }

  double thePtThreshold;
  double theNSigma;
  int theMinHits;

};

#endif
