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
  if ( !surface.mediumProperties() )  return TSoS;
  SurfaceSide side = propDir==alongMomentum ? afterSurface : beforeSurface;
  // single input state?
  if ( TSoS.components().size()>1 )
    throw cms::Exception("LogicError") << "GsfMaterialEffectsUpdator::updateState used with MultiTSOS";
  double weight = TSoS.weight();
  //
  // Get components (will force recalculation, if necessary)
  //
  Effect effects[size()];
  compute(TSoS,propDir,effects);

  //
  // prepare output vector
  //
  MultiTrajectoryStateAssembler result;
  //
  // loop over components
  //
  //   edm::LogDebug("GsfMaterialEffectsUpdator") << "found " << Ws.size() << " components\n"
  // 					     << "  input state has weight " << TSoS.weight();
  for ( auto const & effect : effects ) {
    //      edm::LogDebug("GsfMaterialEffectsUpdator") << "w, dp, sigp = "
    //  	 << Ws[ic] << ", "
    //  	 << dPs[ic] << ", "
    //  	 << sqrt((deltaErrors[ic])[0][0]);
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
      result.addState(TrajectoryStateOnSurface(lp,
					       LocalTrajectoryError(eloc),
					       surface,
					       &(TSoS.globalParameters().magneticField()),
					       side,
					       weight*effect.weight));
      //       edm::LogDebug("GsfMaterialEffectsUpdator") 
      // 	<< "adding state with weight " << weight*Ws[ic];
    }
    else {
      result.addState(TrajectoryStateOnSurface(lp,surface,
					       &(TSoS.globalParameters().magneticField()),
					       side));
    }
  }
  //   edm::LogDebug("GsfMaterialEffectsUpdator") 
  //     << "  output state has weight " << result.combinedState().weight();
  return result.combinedState();
}

