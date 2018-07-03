#include "TrackingTools/TrajectoryFiltering/interface/MinPtTrajectoryFilter.h"

namespace {
struct TLS {
   bool answerMemory=false;
   GlobalVector ftsMemory;

};

thread_local TLS tls;
}


  bool MinPtTrajectoryFilter::test( const TrajectoryMeasurement & tm, int foundHits) const
  {
    //first check min number of hits
    if (foundHits < theMinHits ){ return true;}

    // check for momentum below limit
    //    const FreeTrajectoryState& fts = *tm.updatedState().freeTrajectoryState();

    auto const & tsos = tm.updatedState();
    if (!tsos.isValid()) return false;
    GlobalVector gtp = tsos.globalMomentum();

    //avoid doing twice the check in TBC and QF


    if ( gtp == tls.ftsMemory) { return tls.answerMemory;}
    tls.ftsMemory= gtp;

    auto pT2 =  gtp.perp2();

    //if p_T is way too small: stop
    if (pT2<0.0010f) {tls.answerMemory=false; return false;}

    // if large enouth go
    if (pT2> thePtMin2) { tls.answerMemory=true; return true;}

    //if error is way too big: stop
    float invError = TrajectoryStateAccessor(*tsos.freeTrajectoryState()).inversePtError();
    if (invError > 1.e10f) {tls.answerMemory=false;return false;}

    //calculate the actual pT cut:
    if ((1.f/std::sqrt(pT2) - theNSigma*invError) > theInvPtMin ) {tls.answerMemory=false; return false;}
    //    first term if the max value of pT (pT+N*sigma(pT))
    //    second tern is the cut

    tls.answerMemory=true; return true;
  }

