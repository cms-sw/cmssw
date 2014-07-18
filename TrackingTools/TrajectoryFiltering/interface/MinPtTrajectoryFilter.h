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
    

  virtual bool qualityFilter( const Trajectory& traj)const { return test(traj.lastMeasurement(),traj.foundHits()); }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return test(traj.lastMeasurement(),traj.foundHits()); }
    
  virtual bool toBeContinued( Trajectory& traj) const {return test(traj.lastMeasurement(),traj.foundHits()); }
  virtual bool toBeContinued( TempTrajectory& traj) const { return test(traj.lastMeasurement(),traj.foundHits()); }
  
  virtual std::string name() const {return "MinPtTrajectoryFilter";}

 protected:

  bool test( const TrajectoryMeasurement & tm, int foundHits) const 
  {
    //first check min number of hits 
    if (foundHits < theMinHits ){ return true;}

    // check for momentum below limit
    //    const FreeTrajectoryState& fts = *tm.updatedState().freeTrajectoryState();

    auto const & tsos = tm.updatedState();
    GlobalVector gtp = tsos.globalMomentum();
    
    //avoid doing twice the check in TBC and QF
    static thread_local bool answerMemory=false;
    static thread_local GlobalVector ftsMemory;
    

    if ( gtp == ftsMemory) { return answerMemory;}
    ftsMemory= gtp;

    auto pT2 =  gtp.perp2();

    //if p_T is way too small: stop
    if (pT2<0.0010f) {answerMemory=false; return false;}

    // if large enouth go
    if (pT2> thePtMin2) { answerMemory=true; return true;}

    //if error is way too big: stop
    float invError = TrajectoryStateAccessor(*tsos.freeTrajectoryState()).inversePtError();
    if (invError > 1.e10f) {answerMemory=false;return false;}

    //calculate the actual pT cut: 
    if ((1.f/std::sqrt(pT2) - theNSigma*invError) > theInvPtMin ) {answerMemory=false; return false;}
    //    first term if the max value of pT (pT+N*sigma(pT))
    //    second tern is the cut

    answerMemory=true; return true;
  }

  float thePtMin2; 
  float theInvPtMin;
  float theNSigma;
  int theMinHits;

};

#endif
