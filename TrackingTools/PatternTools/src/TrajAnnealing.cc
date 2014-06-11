#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"

TrajAnnealing::TrajAnnealing( Trajectory traj, float ann ){
  traj_ = traj;
  annealing_ = ann;

  std::vector<TrajectoryMeasurement> measurements = traj_.measurements();
  std::vector<TrajectoryMeasurement> vmeas = traj_.measurements();
  std::vector<TrajectoryMeasurement>::reverse_iterator imeas;

  for (imeas = vmeas.rbegin(); imeas != vmeas.rend(); imeas++){
    theHits_.push_back(imeas->recHit());
    if (imeas->recHit()->isValid()){
      SiTrackerMultiRecHit const & mHit = dynamic_cast<SiTrackerMultiRecHit const &>(*imeas->recHit());
      std::vector<const TrackingRecHit*> components = mHit.recHits();
      int iComp = 0;
      for(std::vector<const TrackingRecHit*>::const_iterator iter2 = components.begin(); iter2 != components.end(); iter2++, iComp++){
          theWeights.push_back(mHit.weight(iComp));
      }
    }
  }
}

void TrajAnnealing::Debug() const{

  std::vector<TrajectoryMeasurement> measurements = traj_.measurements();
  std::vector<TrajectoryMeasurement> vmeas = traj_.measurements();
  std::vector<TrajectoryMeasurement>::reverse_iterator imeas;

  //I run inversely on the trajectory obtained and update the state
  for (imeas = vmeas.rbegin(); imeas != vmeas.rend(); imeas++){
    if (imeas->recHit()->isValid()){
      SiTrackerMultiRecHit const & mHit = dynamic_cast<SiTrackerMultiRecHit const &>(*imeas->recHit());
      std::vector<const TrackingRecHit*> components = mHit.recHits();
      int iComp = 0;
      std::cout << "Annealing: " << mHit.getAnnealingFactor();
      for(std::vector<const TrackingRecHit*>::const_iterator iter2 = components.begin(); iter2 != components.end(); iter2++, iComp++){
        std::cout << "\tHit weight in mrh " << mHit.weight(iComp) << "\t";
      }
    }
  }
}

std::pair<float, std::vector<float> > TrajAnnealing::getAnnealingWeight( const TrackingRecHit& aRecHit ) const {

  if (!aRecHit.isValid()) {
    std::vector<float> dumpyVec = {0.0};
//    std::cout << "Invalid RecHit passed " << std::endl;
    return make_pair(0.0,dumpyVec);
  }

  SiTrackerMultiRecHit const & mHit = dynamic_cast<SiTrackerMultiRecHit const &>(aRecHit); 
  return make_pair(mHit.getAnnealingFactor(), mHit.weights());

}

