#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
// Update of the trajectory state (implemented in base class since general for
//   all classes returning deltaPs and deltaCovs.
//

using namespace SurfaceSideDefinition;

TrajectoryStateOnSurface 
GsfMaterialEffectsUpdator::updateState (const TrajectoryStateOnSurface& TSoS, 
					const PropagationDirection propDir) const {
  //
  // get components of input state and check if material is associated to surface
  //
  const Surface& surface = TSoS.surface();
  if ( !surface.mediumProperties().isValid() )  return TSoS;
  SurfaceSide side = propDir==alongMomentum ? afterSurface : beforeSurface;
  // single input state?
  if (!TSoS.singleState() )
    throw cms::Exception("LogicError") << "GsfMaterialEffectsUpdator::updateState used with MultiTSOS";
  auto weight = TSoS.weight();
  //
  // Get components (will force recalculation, if necessary)
  //
  #if __clang__
  std::vector<Effect> effects(size());
  compute(TSoS,propDir,effects.data());
  #else
  Effect effects[size()];
  compute(TSoS,propDir,effects);
  #endif

  //
  // prepare output vector
  //
  MultiTrajectoryStateAssembler result;
  //
  // loop over components
  //
  LogDebug("GsfMaterialEffectsUpdator") << "found " << size() << " components "
   					     << "  input state has weight " << TSoS.weight();
  for ( auto const & effect : effects ) {
          LogDebug("GsfMaterialEffectsUpdatorDETAIL") << "w, dp, sigp = "
      	 << effect.weight << ", "
      	 << effect.deltaP << ", "
      	 << std::sqrt(effect.deltaCov[materialEffect::elos]);
    //
    // Update momentum. In case of failure: return invalid state.
    // Use deltaP method to ensure update of cache, if necessary!
    //
    LocalTrajectoryParameters lp = TSoS.localParameters();
    if ( !lp.updateP(effect.deltaP) )  
      return TrajectoryStateOnSurface();
    //
    // Update covariance matrix?
    //
    if ( TSoS.hasError() ) {
      AlgebraicSymMatrix55 eloc = TSoS.localError().matrix();
      effect.deltaCov.add(eloc);
      result.addState(TrajectoryStateOnSurface(weight*effect.weight,
                                               lp,
					       LocalTrajectoryError(eloc),
					       surface,
					       &(TSoS.globalParameters().magneticField()),
					       side));
           LogDebug("GsfMaterialEffectsUpdatorDETAIL") 
         	<< "adding state with weight " << weight*effect.weight;
    }
    else {
      result.addState(TrajectoryStateOnSurface(lp,surface,
					       &(TSoS.globalParameters().magneticField()),
					       side));
    }
  }
  LogDebug("GsfMaterialEffectsUpdator") 
       << "  output state has weight " << result.combinedState().weight();
  return result.combinedState();
}

