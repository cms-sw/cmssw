#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <string>

using namespace std;

PropagatorWithMaterial::~PropagatorWithMaterial() {}

PropagatorWithMaterial::PropagatorWithMaterial(PropagationDirection dir,
                                               const float mass,
                                               const MagneticField* mf,
                                               const float maxDPhi,
                                               bool useRungeKutta,
                                               float ptMin,
                                               bool useOldAnalPropLogic)
    : Propagator(dir),
      rkProduct(mf, dir),
      theGeometricalPropagator(useRungeKutta ? rkProduct.propagator.clone()
                                             : new AnalyticalPropagator(mf, dir, maxDPhi, useOldAnalPropLogic)),
      theMEUpdator(new CombinedMaterialEffectsUpdator(mass, ptMin)),
      theMaterialLocation(atDestination),
      field(mf),
      useRungeKutta_(useRungeKutta) {}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterial::propagateWithPath(const FreeTrajectoryState& fts,
                                                                                 const Plane& plane) const {
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(fts, plane);
  if ((newTsosWP.first).isValid() && !materialAtSource()) {
    bool updateOk = theMEUpdator->updateStateInPlace(
        newTsosWP.first, PropagationDirectionFromPath()(newTsosWP.second, propagationDirection()));
    if UNLIKELY (!updateOk)
      newTsosWP.first = TrajectoryStateOnSurface();
  }
  return newTsosWP;
}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterial::propagateWithPath(const FreeTrajectoryState& fts,
                                                                                 const Cylinder& cylinder) const {
  TsosWP newTsosWP = theGeometricalPropagator->propagateWithPath(fts, cylinder);
  if ((newTsosWP.first).isValid() && !materialAtSource()) {
    bool updateOk = theMEUpdator->updateStateInPlace(
        newTsosWP.first, PropagationDirectionFromPath()(newTsosWP.second, propagationDirection()));
    if UNLIKELY (!updateOk)
      newTsosWP.first = TrajectoryStateOnSurface();
  }
  return newTsosWP;
}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterial::propagateWithPath(const TrajectoryStateOnSurface& tsos,
                                                                                 const Plane& plane) const {
  //
  // add material at starting surface, if requested
  //
  TsosWP newTsosWP(tsos, 0.);
  if (materialAtSource()) {
    bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first, propagationDirection());
    if UNLIKELY (!updateOk)
      newTsosWP.first = TrajectoryStateOnSurface();
  }
  if UNLIKELY (!newTsosWP.first.isValid())
    return newTsosWP;
  //
  // geometrical propagation
  //
  newTsosWP = theGeometricalPropagator->propagateWithPath(newTsosWP.first, plane);
  if UNLIKELY (!newTsosWP.first.isValid() || materialAtSource())
    return newTsosWP;
  //
  // add material at destination surface, if requested
  //
  bool updateOk = theMEUpdator->updateStateInPlace(
      newTsosWP.first, PropagationDirectionFromPath()(newTsosWP.second, propagationDirection()));
  if UNLIKELY (!updateOk)
    newTsosWP.first = TrajectoryStateOnSurface();
  return newTsosWP;
}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterial::propagateWithPath(const TrajectoryStateOnSurface& tsos,
                                                                                 const Cylinder& cylinder) const {
  //
  // add material at starting surface, if requested
  //
  TsosWP newTsosWP(tsos, 0.);
  if (materialAtSource()) {
    bool updateOk = theMEUpdator->updateStateInPlace(newTsosWP.first, propagationDirection());
    if UNLIKELY (!updateOk)
      newTsosWP.first = TrajectoryStateOnSurface();
  }
  if UNLIKELY (!newTsosWP.first.isValid())
    return newTsosWP;
  //
  // geometrical propagation
  //
  newTsosWP = theGeometricalPropagator->propagateWithPath(newTsosWP.first, cylinder);
  if UNLIKELY (!(newTsosWP.first).isValid() || materialAtSource())
    return newTsosWP;
  //
  // add material at destination surface, if requested
  //
  bool updateOk = theMEUpdator->updateStateInPlace(
      newTsosWP.first, PropagationDirectionFromPath()(newTsosWP.second, propagationDirection()));
  if UNLIKELY (!updateOk)
    newTsosWP.first = TrajectoryStateOnSurface();
  return newTsosWP;
}

void PropagatorWithMaterial::setPropagationDirection(PropagationDirection dir) {
  theGeometricalPropagator->setPropagationDirection(dir);
  Propagator::setPropagationDirection(dir);
}

bool PropagatorWithMaterial::materialAtSource() const {
  if UNLIKELY ((propagationDirection() == anyDirection) && (theMaterialLocation != atDestination))
    throw cms::Exception("TrackingTools/MaterialEffects",
                         "PropagatorWithMaterial: propagation direction = anyDirection is incompatible with adding of "
                         "material at source");

  return theMaterialLocation == atSource ||
         (theMaterialLocation == fromDirection && propagationDirection() == alongMomentum);
}
