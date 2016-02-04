#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"

using namespace SurfaceSideDefinition;

// static initialization
AlgebraicSymMatrix55  MaterialEffectsUpdator::theNullMatrix;



/** Constructor with explicit mass hypothesis
 */
MaterialEffectsUpdator::MaterialEffectsUpdator ( double mass ) :
  theMass(mass),
  theLastOverP(0),
  theLastDxdz(0), 
  theLastRL(0),
  theLastPropDir(anyDirection),
  theDeltaP(0.),
  theDeltaCov() {}

MaterialEffectsUpdator::~MaterialEffectsUpdator () {}

/** Updates TrajectoryStateOnSurface with material effects
 *    (momentum and covariance matrix are potentially affected.
 */
TrajectoryStateOnSurface MaterialEffectsUpdator::updateState (const TrajectoryStateOnSurface& TSoS, 
							      const PropagationDirection propDir) const {
  TrajectoryStateOnSurface shallowCopy = TSoS;
  // A TSOS is a proxy. Its contents will be really copied only if/when the updateStateInPlace attempts to change them
  return updateStateInPlace(shallowCopy, propDir) ? shallowCopy : TrajectoryStateOnSurface();
}


/** Change in |p| from material effects.
 */
double MaterialEffectsUpdator::deltaP (const TrajectoryStateOnSurface& TSoS, const PropagationDirection propDir) const {
  // check for material
  if ( !TSoS.surface().mediumProperties() )  return 0.;
  // check for change (avoid using compute method if possible)
  if ( newArguments(TSoS,propDir) )  compute(TSoS,propDir);
  return theDeltaP;
}


  /** Contribution to covariance matrix (in local co-ordinates) from material effects.
   */
const AlgebraicSymMatrix55 & MaterialEffectsUpdator::deltaLocalError (const TrajectoryStateOnSurface& TSoS, 
								      const PropagationDirection propDir) const {
  // check for material
  if ( !TSoS.surface().mediumProperties() )  return theNullMatrix;
  // check for change (avoid using compute method if possible)
  if ( newArguments(TSoS,propDir) )  compute(TSoS,propDir);
  return theDeltaCov;
}  


//
// Update of the trajectory state (implemented in base class since general for
//   all classes returning deltaP and deltaCov.
//
bool MaterialEffectsUpdator::updateStateInPlace (TrajectoryStateOnSurface& TSoS, 
				                 const PropagationDirection propDir) const {
  //
  // Check if 
  // - material is associated to surface
  // - propagation direction is not anyDirection
  // - side of surface is not atCenterOfSurface (could be handled with 50% material?)
  //
  const Surface& surface = TSoS.surface();
  if ( !surface.mediumProperties() || propDir==anyDirection || 
       TSoS.surfaceSide()==atCenterOfSurface )  return true;
  //
  // Check, if already on right side of surface
  //
  if ( (propDir==alongMomentum && TSoS.surfaceSide()==afterSurface ) ||
       (propDir==oppositeToMomentum && TSoS.surfaceSide()==beforeSurface ) )  return true;
  //
  // Update momentum. In case of failure: return invalid state
  //
  LocalTrajectoryParameters lp = TSoS.localParameters();
  if ( !lp.updateP(deltaP(TSoS,propDir)) )  
    return false;
  //
  // Update covariance matrix?
  //
  SurfaceSide side = propDir==alongMomentum ? afterSurface : beforeSurface;
  if ( TSoS.hasError() ) {
    AlgebraicSymMatrix55 eloc = TSoS.localError().matrix() + deltaLocalError(TSoS,propDir);
    //TSoS = TrajectoryStateOnSurface(lp,LocalTrajectoryError(eloc),surface, &(TSoS.globalParameters().magneticField()),side);
    TSoS.update(lp,LocalTrajectoryError(eloc),surface,
                &(TSoS.globalParameters().magneticField()),side);
  }
  else {
    TSoS.update(lp,surface,&(TSoS.globalParameters().magneticField()),side);
    //TSoS = TrajectoryStateOnSurface(lp,surface,&(TSoS.globalParameters().magneticField()),side);
  }
  return true;
}

bool MaterialEffectsUpdator::newArguments (const TrajectoryStateOnSurface & TSoS, PropagationDirection  propDir) const {
  // check that track as same momentum and direction, surface has same radLen
  // it optimize also against multiple evaluations on different "surfaces" 
  // belonging to contigous detectors with same radLem 
  bool ok = 
    theLastOverP != TSoS.localParameters().qbp() ||
    theLastDxdz != TSoS.localParameters().dxdz() || 
    theLastRL    != TSoS.surface().mediumProperties()->radLen() ||
    theLastPropDir != propDir;
  if (ok) {
    theLastOverP = TSoS.localParameters().qbp();
    theLastDxdz  = TSoS.localParameters().dxdz(); 
    theLastRL  = TSoS.surface().mediumProperties()->radLen();
    theLastPropDir = propDir;
  }
  return ok;
}
  
