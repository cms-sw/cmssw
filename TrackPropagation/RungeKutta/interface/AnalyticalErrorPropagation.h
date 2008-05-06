#ifndef AnalyticalErrorPropagation_H
#define AnalyticalErrorPropagation_H

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

#include <utility>

class Surface;


class AnalyticalErrorPropagation {
public:

    std::pair<TrajectoryStateOnSurface,double>
    operator()( const FreeTrajectoryState& startingState, 
		const Surface& surface, SurfaceSideDefinition::SurfaceSide side,
		const GlobalTrajectoryParameters& destParameters, 
		const double& s) const;
};


#endif
