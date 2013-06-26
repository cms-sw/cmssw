#include "TrackingTools/TrackFitters/interface/KFSplittingFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"



void KFSplittingFitter::sorter(const RecHitContainer& hits, PropagationDirection dir, RecHitContainer & result) const {
  result.reserve(hits.size());
  for(RecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end();
      ihit++) {
    if(!(**ihit).isValid()) result.push_back(*ihit);
    else if((**ihit).transientHits().size() == 1) result.push_back(*ihit);
    else {      
      RecHitContainer splitted = RecHitSplitter().split((**ihit).transientHits());
      RecHitContainer sorted = 
	RecHitSorter().sortHits(splitted, dir);
      for (RecHitContainer::iterator srt = sorted.begin(); srt != sorted.end(); srt++) {
	result.push_back(*srt);
	//      result.insert(result.end(), sorted.begin(), sorted.end());
      }
    }
  }
}


Trajectory KFSplittingFitter::fitOne(const Trajectory& aTraj, fitType type) const {

  if(aTraj.empty()) return Trajectory();
  
  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = 
    TrajectoryStateWithArbitraryError()(firstTM.predictedState());
  
  RecHitContainer const & hits = aTraj.recHits();
  RecHitContainer result; 
  sorter(hits,aTraj.direction(),result);
  

  return fitter.fitOne(aTraj.seed(), result, firstTsos,type);
  
}

Trajectory KFSplittingFitter::fitOne(const TrajectorySeed& aSeed,
				     const RecHitContainer& hits,
				     fitType type) const {
  
  RecHitContainer result;
  sorter(hits,aSeed.direction(),result);
  
  
  return fitter.fitOne(aSeed, result, type);
  
  
}

Trajectory KFSplittingFitter::fitOne(const TrajectorySeed& aSeed,
				     const RecHitContainer& hits, 
				     const TSOS& firstPredTsos, fitType type) const {
  
  RecHitContainer result;
  sorter(hits,aSeed.direction(),result);
  
  
  return fitter.fitOne(aSeed, result, firstPredTsos,type);
  
}

