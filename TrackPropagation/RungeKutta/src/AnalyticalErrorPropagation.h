#ifndef AnalyticalErrorPropagation_H
#define AnalyticalErrorPropagation_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

#include <utility>

class Surface;


class dso_internal AnalyticalErrorPropagation {
public:

    std::pair<TrajectoryStateOnSurface,double>
    operator()( const FreeTrajectoryState& startingState, 
		const Surface& surface, SurfaceSideDefinition::SurfaceSide side,
		const GlobalTrajectoryParameters& destParameters, 
		const double& s) const;
};


#endif
