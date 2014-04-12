#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"


#include<iostream>

#define PSIZE(CNAME) std::cout << #CNAME << ": " << sizeof(CNAME) << std::endl

int main() {
  std::cout << "sizes" << std::endl;
  PSIZE(TrajectoryMeasurement);
  PSIZE(TransientTrackingRecHit);
  PSIZE(Trajectory);
  PSIZE(TempTrajectory);

  return 0;
}
