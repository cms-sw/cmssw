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

class MinPtTrajectoryFilter : public TrajectoryFilter {
public:

  explicit MinPtTrajectoryFilter( double ptMin, float nSigma = 5.F, int nH=3): thePtMin( ptMin), theNSigma(nSigma), theMinHits(nH)  {}

  explicit MinPtTrajectoryFilter( const edm::ParameterSet & pset) :
    thePtMin(pset.getParameter<double>("minPt")),
    theNSigma(pset.getParameter<double>("nSigmaMinPt")),
    theMinHits(pset.getParameter<int>("minHitsMinPt")){}
    

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
    const FreeTrajectoryState& fts = *tm.updatedState().freeTrajectoryState();

    //avoid doing twice the check in TBC and QF
    static bool answerMemory=false;
    static FreeTrajectoryState ftsMemory;
    if (ftsMemory.parameters().vector() == fts.parameters().vector()) { return answerMemory;}
    ftsMemory=fts;

    //if p_T is way too small: stop
    double pT = fts.momentum().perp();
    if (pT<0.010) {answerMemory=false; return false;}
    //if error is way too big: stop
    double invError = TrajectoryStateAccessor(fts).inversePtError();
    if (invError > 1.e10) {answerMemory=false;return false;}

    //calculate the actual pT cut: 
    if ((1/pT - theNSigma*invError) > 1/thePtMin) {answerMemory=false; return false;}
    //    first term if the max value of pT (pT+N*sigma(pT))
    //    second tern is the cut

    answerMemory=true; return true;
  }

  double thePtMin;
  double theNSigma;
  int theMinHits;

};

#endif
