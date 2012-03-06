#include "TrackingTools/TrackFitters/interface/FlexibleKFFittingSmoother.h"

using namespace std;

FlexibleKFFittingSmoother::~FlexibleKFFittingSmoother() 
{
  delete theStandardFitter;
  delete theLooperFitter;
}

std::vector<Trajectory> 
FlexibleKFFittingSmoother::fit(const Trajectory& t,fitType type) const {
  if(type==standard)
    return theStandardFitter->fit(t);
  else if(type==looper)
    return theLooperFitter->fit(t);
  else
    return std::vector<Trajectory>();
}



std::vector<Trajectory> 
FlexibleKFFittingSmoother::fit(const TrajectorySeed& aSeed,
		       const RecHitContainer& hits, 
		       const TrajectoryStateOnSurface& firstPredTsos,
		       fitType type) const{
  if(type==standard)
    return theStandardFitter->fit(aSeed,hits,firstPredTsos);
  else if(type==looper)
    return theLooperFitter->fit(aSeed,hits,firstPredTsos);
  else
    return std::vector<Trajectory>();
}


std::vector<Trajectory> 
FlexibleKFFittingSmoother::fit(const TrajectorySeed& aSeed,
		       const RecHitContainer& hits,
		       fitType type) const{
  if(type==standard)
    return theStandardFitter->fit(aSeed,hits);
  else if(type==looper)
    return theLooperFitter->fit(aSeed,hits);
  else
    return std::vector<Trajectory>();
}
