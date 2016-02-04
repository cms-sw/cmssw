#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TrackPropagation/RungeKutta/interface/RKTestPropagator.h"
#include <string>

using namespace std;
PropagatorWithMaterial::PropagatorWithMaterial (PropagationDirection dir,
						const float mass, 
						const MagneticField * mf,
						const float maxDPhi,
						bool useRungeKutta,
                                                float ptMin) :
  Propagator(dir),
  theGeometricalPropagator(),
  theMEUpdator(new CombinedMaterialEffectsUpdator(mass, ptMin)),
  theMaterialLocation(atDestination), field(mf),useRungeKutta_(useRungeKutta) {
  
  if(useRungeKutta_)    
    theGeometricalPropagator = DeepCopyPointerByClone<Propagator>(new RKTestPropagator(mf,dir));
  else theGeometricalPropagator = DeepCopyPointerByClone<Propagator>(new AnalyticalPropagator(mf,dir,maxDPhi));
   
}

PropagatorWithMaterial::PropagatorWithMaterial (const Propagator& aPropagator,
						const MaterialEffectsUpdator& aMEUpdator,
						const MagneticField * mf,
						bool useRungeKutta) :
  Propagator(aPropagator.propagationDirection()),
  theGeometricalPropagator(aPropagator.clone()),
  theMEUpdator(aMEUpdator.clone()),
  theMaterialLocation(atDestination), field(mf),useRungeKutta_(useRungeKutta) {}

pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const FreeTrajectoryState& fts, 
					   const Plane& plane) const {
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(fts,plane);
  if ( (newTsosWP.first).isValid() && !materialAtSource() ) { 
      bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first,
                                                       PropagationDirectionFromPath()(newTsosWP.second,
                                                                                      propagationDirection()));
      if (!updateOk) newTsosWP.first = TrajectoryStateOnSurface();
  }
  return newTsosWP;
}

pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const FreeTrajectoryState& fts, 
					   const Cylinder& cylinder) const {
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(fts,cylinder);
  if ( (newTsosWP.first).isValid() && !materialAtSource() ) { 
      bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first,
                                                       PropagationDirectionFromPath()(newTsosWP.second,
                                                                                      propagationDirection()));
      if (!updateOk) newTsosWP.first = TrajectoryStateOnSurface();
  }
  return newTsosWP;
}


pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const TrajectoryStateOnSurface& tsos, 
					   const Plane& plane) const {
  //
  // add material at starting surface, if requested
  //
  TsosWP newTsosWP(tsos,0.);
  if ( materialAtSource() ) {
    bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first,propagationDirection());
    if (!updateOk) newTsosWP.first = TrajectoryStateOnSurface();
  }
  if ( !newTsosWP.first.isValid() )  return newTsosWP;
  //
  // geometrical propagation
  //
  newTsosWP = theGeometricalPropagator->propagateWithPath(newTsosWP.first,plane);
  if ( !newTsosWP.first.isValid() || materialAtSource() )  return newTsosWP;
  //
  // add material at destination surface, if requested
  //
  bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first,
                                                   PropagationDirectionFromPath()(newTsosWP.second,
                                                                                  propagationDirection()));
  if (!updateOk) newTsosWP.first = TrajectoryStateOnSurface();
  return newTsosWP;
}

pair<TrajectoryStateOnSurface,double> 
PropagatorWithMaterial::propagateWithPath (const TrajectoryStateOnSurface& tsos,
					   const Cylinder& cylinder) const {
  //
  // add material at starting surface, if requested
  //
  TsosWP newTsosWP(tsos,0.);
  if ( materialAtSource() ) {
    bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first,propagationDirection());
    if (!updateOk) newTsosWP.first = TrajectoryStateOnSurface();
  }
  if ( !newTsosWP.first.isValid() )  return newTsosWP;
  //
  // geometrical propagation
  //
  newTsosWP = theGeometricalPropagator->propagateWithPath(newTsosWP.first,cylinder);
  if ( !(newTsosWP.first).isValid() || materialAtSource() )  return newTsosWP;
  //
  // add material at destination surface, if requested
  //
  bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first,
                                                   PropagationDirectionFromPath()(newTsosWP.second,
                                                                                  propagationDirection()));
  if (!updateOk) newTsosWP.first = TrajectoryStateOnSurface();
  return newTsosWP;
}

void PropagatorWithMaterial::setPropagationDirection (PropagationDirection dir) const {
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
