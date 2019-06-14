#ifndef TrajectoryToResiduals_h
#define TrajectoryToResiduals_h

#include "DataFormats/TrackReco/interface/TrackResiduals.h"

class Trajectory;
reco::TrackResiduals trajectoryToResiduals(const Trajectory &);

#endif
