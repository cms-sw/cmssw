#include "TrackingTools/TrackFitters/interface/KFSplittingFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/RecHitSplitter.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
//#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
// #include "CommonDet/BasicDet/interface/InvalidRecHit.h"
// #include "CommonDet/BasicDet/interface/Det.h"
// #include "CommonDet/BasicDet/interface/DetUnit.h"
// #include "CommonDet/DetLayout/interface/DetLayer.h"

vector<Trajectory> KFSplittingFitter::fit(const Trajectory& aTraj) const {

  if(aTraj.empty()) return vector<Trajectory>();
  
  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = 
    TrajectoryStateWithArbitraryError()(firstTM.predictedState());
  
  edm::OwnVector<TransientTrackingRecHit> hits = aTraj.recHits();
  edm::OwnVector<TransientTrackingRecHit> result; 
  result.reserve(hits.size());
  for(edm::OwnVector<TransientTrackingRecHit>::iterator ihit = hits.begin(); ihit != hits.end();
      ihit++) {
    if(!(*ihit).isValid()) result.push_back(ihit.get());
    else if((*ihit).transientHits().size() == 1) result.push_back(ihit.get());
    else {
      edm::OwnVector<TransientTrackingRecHit> splitted = RecHitSplitter().split((*ihit).transientHits());
      edm::OwnVector<TransientTrackingRecHit> sorted = 
	RecHitSorter().sortHits(splitted, propagator()->propagationDirection());
      for (edm::OwnVector<TransientTrackingRecHit>::iterator srt = sorted.begin(); srt != sorted.end(); srt++) {
	result.push_back(srt.get());
	//      result.insert(result.end(), sorted.begin(), sorted.end());
      }
    }
  }
  

  return KFTrajectoryFitter::fit(aTraj.seed(), result, firstTsos);
  
}

vector<Trajectory> KFSplittingFitter::fit(const TrajectorySeed& aSeed,
					  const edm::OwnVector<TransientTrackingRecHit>& hits, 
					  const TSOS& firstPredTsos) const {

  edm::OwnVector<TransientTrackingRecHit> result;
  result.reserve(hits.size());
  for(edm::OwnVector<TransientTrackingRecHit>::const_iterator ihit = hits.begin(); ihit != hits.end();
      ihit++) {
    if(!(*ihit).isValid()) result.push_back(ihit->clone());
    else if((*ihit).transientHits().size() == 1) result.push_back(ihit->clone());
    else {      
      edm::OwnVector<TransientTrackingRecHit> splitted = RecHitSplitter().split(ihit->clone()->transientHits());
      edm::OwnVector<TransientTrackingRecHit> sorted = 
	RecHitSorter().sortHits(splitted, propagator()->propagationDirection());
      for (edm::OwnVector<TransientTrackingRecHit>::iterator srt = sorted.begin(); srt != sorted.end(); srt++) {
	result.push_back(srt.get());
	//      result.insert(result.end(), sorted.begin(), sorted.end());
      }
    }
  }
  
  return KFTrajectoryFitter::fit(aSeed, result, firstPredTsos);
  
}

