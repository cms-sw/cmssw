#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"

using namespace SurfaceSideDefinition;

//#include "Utilities/UI/interface/SimpleConfigurable.h"
//
// Update of the trajectory state (implemented in base class since general for
//   all classes returning deltaP and deltaCov.
//
TrajectoryStateOnSurface MaterialEffectsUpdator::updateState (const TrajectoryStateOnSurface& TSoS, 
							      const PropagationDirection propDir) const {
  //
  // Check if 
  // - material is associated to surface
  // - propagation direction is not anyDirection
  // - side of surface is not atCenterOfSurface (could be handled with 50% material?)
  //
  const Surface& surface = TSoS.surface();
  if ( !surface.mediumProperties() || propDir==anyDirection || 
       TSoS.surfaceSide()==atCenterOfSurface )  return TSoS;
  //
  // Check, if already on right side of surface
  //
  if ( (propDir==alongMomentum && TSoS.surfaceSide()==afterSurface ) ||
       (propDir==oppositeToMomentum && TSoS.surfaceSide()==beforeSurface ) )  return TSoS;
  //
  // Update momentum. In case of failure: return invalid state
  //
  LocalTrajectoryParameters lp = TSoS.localParameters();
  if ( !lp.updateP(deltaP(TSoS,propDir)) )  
    return TrajectoryStateOnSurface();
  //
  // Update covariance matrix?
  //
  SurfaceSide side = propDir==alongMomentum ? afterSurface : beforeSurface;
  if ( TSoS.hasError() ) {
    AlgebraicSymMatrix55 eloc = TSoS.localError().matrix() + deltaLocalError(TSoS,propDir);
    return TrajectoryStateOnSurface(lp,LocalTrajectoryError(eloc),surface,
				    &(TSoS.globalParameters().magneticField()),side);
  }
  else {
    return TrajectoryStateOnSurface(lp,surface,&(TSoS.globalParameters().magneticField()),side);
  }
}

// static initialization
AlgebraicSymMatrix55  MaterialEffectsUpdator::theNullMatrix;
