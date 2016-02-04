#include "TrackingTools/TrackFitters/interface/KFSplittingFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/RecHitSplitter.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

std::vector<Trajectory> KFSplittingFitter::fit(const Trajectory& aTraj) const {
  
  typedef RecHitSplitter::RecHitContainer        RecHitContainer;
  
  if(aTraj.empty()) return std::vector<Trajectory>();
  
  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = 
    TrajectoryStateWithArbitraryError()(firstTM.predictedState());
  
  RecHitContainer hits = aTraj.recHits();
  RecHitContainer result; 
  result.reserve(hits.size());
  for(RecHitContainer::iterator ihit = hits.begin(); ihit != hits.end();
      ihit++) {
    if(!(**ihit).isValid()) result.push_back(*ihit);
    else if((**ihit).transientHits().size() == 1) result.push_back(*ihit);
    else {
      RecHitContainer splitted = RecHitSplitter().split((**ihit).transientHits());
      RecHitContainer sorted = 
	RecHitSorter().sortHits(splitted, aTraj.direction());
      for (RecHitContainer::iterator srt = sorted.begin(); srt != sorted.end(); srt++) {
	result.push_back(*srt);
	//      result.insert(result.end(), sorted.begin(), sorted.end());
      }
    }
  }
  

  return KFTrajectoryFitter::fit(aTraj.seed(), result, firstTsos);
  
}

std::vector<Trajectory> KFSplittingFitter::fit(const TrajectorySeed& aSeed,
					       const RecHitContainer& hits, 
					       const TSOS& firstPredTsos) const {

  RecHitContainer result;
  result.reserve(hits.size());
  for(RecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end();
      ihit++) {
    if(!(**ihit).isValid()) result.push_back(*ihit);
    else if((**ihit).transientHits().size() == 1) result.push_back(*ihit);
    else {      
      RecHitContainer splitted = RecHitSplitter().split((**ihit).transientHits());
      RecHitContainer sorted = 
	RecHitSorter().sortHits(splitted, aSeed.direction());
      for (RecHitContainer::iterator srt = sorted.begin(); srt != sorted.end(); srt++) {
	result.push_back(*srt);
	//      result.insert(result.end(), sorted.begin(), sorted.end());
      }
    }
  }
  
  return KFTrajectoryFitter::fit(aSeed, result, firstPredTsos);
  
}

