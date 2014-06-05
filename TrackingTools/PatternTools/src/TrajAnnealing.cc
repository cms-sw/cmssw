#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"

TrajAnnealing::TrajAnnealing( Trajectory traj, float ann ){
  traj_ = traj;
  annealing_ = ann;

  std::vector<TrajectoryMeasurement> measurements = traj_.measurements();
  std::vector<TrajectoryMeasurement> vmeas = traj_.measurements();
  std::vector<TrajectoryMeasurement>::reverse_iterator imeas;

  //I run inversely on the trajectory obtained and update the state
  for (imeas = vmeas.rbegin(); imeas != vmeas.rend(); imeas++){
    theHits_.push_back(imeas->recHit());
  }

  int hitcounter = 1;
  for(TransientTrackingRecHit::RecHitContainer::const_iterator ihit = theHits_.begin();
      ihit != theHits_.end(); ++ihit, ++hitcounter) {

    std::pair<float, std::vector<float> > annweight;
    annweight = getAnnealingWeight(**ihit);
    for(unsigned int i = 0 ; i < annweight.second.size(); i++ ){
      theWeights.push_back( annweight.second.at(i) );
    }
  }
}

void TrajAnnealing::Debug() const{

  std::vector<TrajectoryMeasurement> measurements = traj_.measurements();
  TransientTrackingRecHit::RecHitContainer hits;
  std::vector<TrajectoryMeasurement> vmeas = traj_.measurements();
  std::vector<TrajectoryMeasurement>::reverse_iterator imeas;

  //I run inversely on the trajectory obtained and update the state
  for (imeas = vmeas.rbegin(); imeas != vmeas.rend(); imeas++){
    hits.push_back(imeas->recHit());
  }

  int hitcounter = 1;
  for(TransientTrackingRecHit::RecHitContainer::const_iterator ihit = hits.begin();
      ihit != hits.end(); ++ihit, ++hitcounter) {

    std::pair<float, std::vector<float> > annweight;
    annweight = getAnnealingWeight(**ihit);
    std::cout << "Annealing: " << annweight.first;
    for(unsigned int i = 0 ; i < annweight.second.size(); i++ ){
      std::cout << "\tHit weight in mrh " << annweight.second.at(i) << "\t";
    }
    std::cout << std::endl;
  }

}

std::pair<float, std::vector<float> > TrajAnnealing::getAnnealingWeight( const TrackingRecHit& aRecHit ) const {

  if (!aRecHit.isValid()) {
    std::vector<float> dumpyVec = {0.0};
//    std::cout << "Invalid RecHit passed " << std::endl;
    return make_pair(0.0,dumpyVec);
  }

  TSiTrackerMultiRecHit const & mHit = dynamic_cast<TSiTrackerMultiRecHit const &>(aRecHit);
  return make_pair(mHit.getAnnealingFactor(), mHit.weights());

}

