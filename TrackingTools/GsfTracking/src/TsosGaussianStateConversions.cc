#include "TrackingTools/GsfTracking/interface/TsosGaussianStateConversions.h"

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "boost/shared_ptr.hpp"

using namespace SurfaceSideDefinition;

namespace GaussianStateConversions {

  MultiGaussianState<5> multiGaussianStateFromTSOS (const TrajectoryStateOnSurface tsos)
  {
    if ( !tsos.isValid() )  return MultiGaussianState<5>();

    typedef boost::shared_ptr< SingleGaussianState<5> > SingleStatePtr;
    const std::vector<TrajectoryStateOnSurface>& components = tsos.components();
    MultiGaussianState<5>::SingleStateContainer singleStates;
    singleStates.reserve(components.size());
    for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
	  ic!=components.end(); ic ++ ) {
      if ( ic->isValid() ) {
	SingleStatePtr sgs(new SingleGaussianState<5>(ic->localParameters().vector(),
							      ic->localError().matrix(),
							      ic->weight()));
	singleStates.push_back(sgs);
      }
    }
    return MultiGaussianState<5>(singleStates);
  }

  TrajectoryStateOnSurface tsosFromMultiGaussianState (const MultiGaussianState<5>& multiState,
							  const TrajectoryStateOnSurface refTsos)
  {
    if ( multiState.components().empty() )  return TrajectoryStateOnSurface();
    const Surface& surface = refTsos.surface();
    SurfaceSide side = refTsos.surfaceSide();
    const MagneticField* field = refTsos.magneticField();
    TrajectoryStateOnSurface refTsos1 = refTsos.components().front();
    double pzSign = refTsos1.localParameters().pzSign();
    bool charged = refTsos1.charge()!=0;

    const MultiGaussianState<5>::SingleStateContainer& singleStates = 
      multiState.components();
    std::vector<TrajectoryStateOnSurface> components;
    components.reserve(singleStates.size());
    for ( MultiGaussianState<5>::SingleStateContainer::const_iterator ic=singleStates.begin();
	  ic!=singleStates.end(); ic++ ) {
      components.push_back(TrajectoryStateOnSurface(LocalTrajectoryParameters((**ic).mean(),
									      pzSign,charged),
						    LocalTrajectoryError((**ic).covariance()),
						    surface,field,side,(**ic).weight()));
    }
    return TrajectoryStateOnSurface(new BasicMultiTrajectoryState(components));
  }
}

