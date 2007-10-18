#include "TrackingTools/TrajectoryFiltering/interface/BaseCkfTrajectoryFilter.h"
BaseCkfTrajectoryFilter::BaseCkfTrajectoryFilter( const edm::ParameterSet & pset)
{
  //define the filters by default in the BaseCkfTrajectoryBuilder
  filters.push_back( new ChargeSignificanceTrajectoryFilter(pset));
  filters.push_back( new MaxLostHitsTrajectoryFilter(pset));
  filters.push_back( new MaxConsecLostHitsTrajectoryFilter(pset));
  filters.push_back( new MinPtTrajectoryFilter(pset));
  filters.push_back( new MaxHitsTrajectoryFilter(pset));
  filters.push_back( new MinHitsTrajectoryFilter(pset));
}

std::string BaseCkfTrajectoryFilter::name() const {return "BaseCkfTrajectoryFilter";}
