#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>

PropagatorWithMaterial::PropagatorWithMaterial (PropagationDirection dir,
						const float mass, 
						const MagneticField * mf,
						const float maxDPhi) :
  Propagator(dir),
  theGeometricalPropagator(new AnalyticalPropagator(mf,dir,maxDPhi)),
  theMEUpdator(new CombinedMaterialEffectsUpdator(mass)),
  theMaterialLocation(atDestination), field(mf) {}

PropagatorWithMaterial::PropagatorWithMaterial (const Propagator& aPropagator,
						const MaterialEffectsUpdator& aMEUpdator,
						const MagneticField * mf) :
  Propagator(aPropagator.propagationDirection()),
  theGeometricalPropagator(aPropagator.clone()),
  theMEUpdator(aMEUpdator.clone()),
  theMaterialLocation(atDestination), field(mf) {}

pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const FreeTrajectoryState& fts, 
					   const Plane& plane) const {
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(fts,plane);
  if ( !(newTsosWP.first).isValid() || materialAtSource() )  return newTsosWP;
  TrajectoryStateOnSurface updatedTSoS = 
    theMEUpdator->updateState(newTsosWP.first,
			      PropagationDirectionFromPath()(newTsosWP.second,
							     propagationDirection()));
  return TsosWP(updatedTSoS,newTsosWP.second);
}

pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const FreeTrajectoryState& fts, 
					   const Cylinder& cylinder) const {
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(fts,cylinder);
  if ( !(newTsosWP.first).isValid() || materialAtSource() )  return newTsosWP;
  TrajectoryStateOnSurface updatedTSoS = 
    theMEUpdator->updateState(newTsosWP.first,
			      PropagationDirectionFromPath()(newTsosWP.second,
							     propagationDirection()));
  return TsosWP(updatedTSoS,newTsosWP.second);
}


pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const TrajectoryStateOnSurface& tsos, 
					   const Plane& plane) const {
  //
  // add material at starting surface, if requested
  //
  TrajectoryStateOnSurface stateAtSource;
  if ( materialAtSource() )
    stateAtSource = theMEUpdator->updateState(tsos,propagationDirection());
  else
    stateAtSource = tsos;
  if ( !stateAtSource.isValid() )  return TsosWP(stateAtSource,0.);
  //
  // geometrical propagation
  //
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(stateAtSource,plane);
  if ( !(newTsosWP.first).isValid() || materialAtSource() )  return newTsosWP;
  //
  // add material at destination surface, if requested
  //
  TrajectoryStateOnSurface updatedTSoS = 
    theMEUpdator->updateState(newTsosWP.first,
			      PropagationDirectionFromPath()(newTsosWP.second,
							     propagationDirection()));
  return TsosWP(updatedTSoS,newTsosWP.second);
}

pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const TrajectoryStateOnSurface& tsos,
					   const Cylinder& cylinder) const {
  //
  // add material at starting surface, if requested
  //
  TrajectoryStateOnSurface stateAtSource;
  if ( materialAtSource() )
    stateAtSource = theMEUpdator->updateState(tsos,propagationDirection());
  else
    stateAtSource = tsos;
  if ( !stateAtSource.isValid() )  return TsosWP(stateAtSource,0.);
  //
  // geometrical propagation
  //
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(stateAtSource,cylinder);
  if ( !(newTsosWP.first).isValid() || materialAtSource() )  return newTsosWP;
  //
  // add material at destination surface, if requested
  //
  TrajectoryStateOnSurface updatedTSoS = 
    theMEUpdator->updateState(newTsosWP.first,
			      PropagationDirectionFromPath()(newTsosWP.second,
							     propagationDirection()));
  return TsosWP(updatedTSoS,newTsosWP.second);
}

void PropagatorWithMaterial::setPropagationDirection (PropagationDirection dir) {
  theGeometricalPropagator->setPropagationDirection(dir);
  Propagator::setPropagationDirection(dir);
}

bool
PropagatorWithMaterial::materialAtSource() const {
  if ( propagationDirection()==anyDirection ) {
    if ( theMaterialLocation!=atDestination ) { 
      string message("PropagatorWithMaterial: propagation direction = anyDirection is ");
      message += "incompatible with adding of material at source";
      throw cms::Exception("TrackingTools/MaterialEffects",message);
    }
  }
  return theMaterialLocation==atSource || (theMaterialLocation==fromDirection&&
					   propagationDirection()==alongMomentum);
}
