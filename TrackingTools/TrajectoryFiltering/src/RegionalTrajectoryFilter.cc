#include "TrackingTools/TrajectoryFiltering/interface/RegionalTrajectoryFilter.h"

RegionalTrajectoryFilter::RegionalTrajectoryFilter (const edm::ParameterSet &  pset, edm::ConsumesCollector& iC) :thePtFilter(0){}
RegionalTrajectoryFilter::RegionalTrajectoryFilter( const TrackingRegion& region): thePtFilter(region.ptMin()) {}
    
bool RegionalTrajectoryFilter::qualityFilter(const TempTrajectory& traj) const { return true;}
bool RegionalTrajectoryFilter::qualityFilter(const Trajectory& traj) const { return true;}

bool RegionalTrajectoryFilter::toBeContinued (TempTrajectory& traj) const {return thePtFilter.toBeContinued(traj); }
bool RegionalTrajectoryFilter::toBeContinued(Trajectory& traj) const { return thePtFilter.toBeContinued(traj); }

std::string RegionalTrajectoryFilter::name () const {return std::string("RegionalTrajectoryFilter");}
