#ifndef TrajectoryToResiduals_h
#define TrajectoryToResiduals_h

#include "DataFormats/TrackReco/interface/TrackResiduals.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFwd.h"
reco::TrackResiduals trajectoryToResiduals(const Trajectory &);

#endif
