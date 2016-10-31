#include "TrackingTools/TrajectoryFiltering/interface/ThresholdPtTrajectoryFilter.h"

namespace {
struct TLS {
    bool answerMemory=false;
   FreeTrajectoryState ftsMemory;

};

thread_local TLS tls;
}

  bool ThresholdPtTrajectoryFilter::test( const TrajectoryMeasurement & tm, int foundHits) const
  {
    //first check min number of hits
    if (foundHits < theMinHits ){ return true;}

    // check for momentum below limit
    if (!tm.updatedState().isValid()) return	false;
    const FreeTrajectoryState& fts = *tm.updatedState().freeTrajectoryState();

    //avoid doing twice the check in TBC and QF
    // We make it thread local so that we avoid race conditions between
    // threads, and we make sure there is no cache contention between them.
    if (tls.ftsMemory.parameters().vector() == fts.parameters().vector()) { return tls.answerMemory;}
    tls.ftsMemory=fts;

    //if p_T is way too small: stop
    double pT = fts.momentum().perp();
    if (pT<0.010) {tls.answerMemory=false; return false;}
    //if error is way too big: stop
    double invError = TrajectoryStateAccessor(fts).inversePtError();
    if (invError > 1.e10) {tls.answerMemory=false;return false;}

    //calculate the actual pT cut:
    if ((1/pT + theNSigma*invError ) < 1/thePtThreshold ) {tls.answerMemory=false; return false;}
    //    first term is the minimal value of pT (pT-N*sigma(pT))
    //    secon term is the cut

    tls.answerMemory=true; return true;
  }

