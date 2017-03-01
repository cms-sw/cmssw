#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"

TrajAnnealing::TrajAnnealing( const Trajectory& traj, float ann ){
  traj_ = traj;
  annealing_ = ann;

  const Trajectory::DataContainer& measurements = traj_.measurements();
  if( measurements.size() > 2 ){
    Trajectory::DataContainer::const_iterator ibegin,iend;
    int increment(0);
    if( traj.direction() == alongMomentum ){
      ibegin = measurements.begin() + 1;
      iend = measurements.end() - 1;
      increment = 1;
    } else {
      ibegin = measurements.end() - 2;
      iend = measurements.begin();
      increment = -1;
    }
    
    for( Trajectory::DataContainer::const_iterator imeas = ibegin; imeas != iend; imeas += increment ){

      theHits_.push_back(imeas->recHit());
      if (imeas->recHit()->isValid()){
        SiTrackerMultiRecHit const & mHit = dynamic_cast<SiTrackerMultiRecHit const &>(*imeas->recHit());
        std::vector<const TrackingRecHit*> components = mHit.recHits();
        int iComp = 0;
        for(std::vector<const TrackingRecHit*>::const_iterator iter = components.begin(); 
  	    iter != components.end(); iter++, iComp++){
            theWeights.push_back(mHit.weight(iComp));
        }
      }
    }

  }

}

std::pair<float, std::vector<float> > TrajAnnealing::getAnnealingWeight( const TrackingRecHit& aRecHit ) const {

  if (!aRecHit.isValid()) {
    std::vector<float> dumpyVec = {0.0};
    return make_pair(0.0,dumpyVec);
  }

  SiTrackerMultiRecHit const & mHit = dynamic_cast<SiTrackerMultiRecHit const &>(aRecHit); 
  return make_pair(mHit.getAnnealingFactor(), mHit.weights());

}

