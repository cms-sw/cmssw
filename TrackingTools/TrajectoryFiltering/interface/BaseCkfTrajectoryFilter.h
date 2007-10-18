#ifndef BaseCkfTrajectoryFilter_H
#define BaseCkfTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/CompositeTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/ChargeSignificanceTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxConsecLostHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MaxLostHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinHitsTrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/MinPtTrajectoryFilter.h"



class BaseCkfTrajectoryFilter : public CompositeTrajectoryFilter {
public:

  explicit BaseCkfTrajectoryFilter( const edm::ParameterSet & pset);
  virtual std::string name() const;

protected:

};

#endif
