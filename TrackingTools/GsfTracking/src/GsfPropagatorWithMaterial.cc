#include "TrackingTools/GsfTracking/interface/GsfPropagatorWithMaterial.h"

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <atomic>

GsfPropagatorWithMaterial::GsfPropagatorWithMaterial(const Propagator& aPropagator,
                                                     const GsfMaterialEffectsUpdator& aMEUpdator)
    : Propagator(aPropagator.propagationDirection()),
      theGeometricalPropagator(new GsfPropagatorAdapter(aPropagator)),
      theConvolutor(new FullConvolutionWithMaterial(aMEUpdator)),
      theMaterialLocation(atDestination) {}

GsfPropagatorWithMaterial::GsfPropagatorWithMaterial(const GsfPropagatorAdapter& aGsfPropagator,
                                                     const FullConvolutionWithMaterial& aConvolutor)
    : Propagator(aGsfPropagator.propagationDirection()),
      theGeometricalPropagator(aGsfPropagator.clone()),
      theConvolutor(aConvolutor.clone()),
      theMaterialLocation(atDestination) {}

std::pair<TrajectoryStateOnSurface, double> GsfPropagatorWithMaterial::propagateWithPath(
    const TrajectoryStateOnSurface& tsos, const Plane& plane) const {
  // add material before propagation?
  //
  TrajectoryStateOnSurface stateAtSource;
  if (materialAtSource())
    stateAtSource = convoluteStateWithMaterial(tsos, propagationDirection());
  else
    stateAtSource = tsos;
  if (!stateAtSource.isValid())
    return TsosWP(stateAtSource, 0.);
  //
  // geometrical propagation
  //
  TsosWP propStateWP(theGeometricalPropagator->propagateWithPath(stateAtSource, plane));
  if (!(propStateWP.first).isValid())
    return propStateWP;
  //
  // return convoluted state
  //
  return convoluteWithMaterial(propStateWP);
}

std::pair<TrajectoryStateOnSurface, double> GsfPropagatorWithMaterial::propagateWithPath(
    const TrajectoryStateOnSurface& tsos, const Cylinder& cylinder) const {
  // add material before propagation?
  //
  TrajectoryStateOnSurface stateAtSource;
  if (materialAtSource())
    stateAtSource = convoluteStateWithMaterial(tsos, propagationDirection());
  else
    stateAtSource = tsos;
  if (!stateAtSource.isValid())
    return TsosWP(stateAtSource, 0.);
  //
  // geometrical propagation
  //
  TsosWP propStateWP(theGeometricalPropagator->propagateWithPath(stateAtSource, cylinder));
  if (!(propStateWP.first).isValid())
    return propStateWP;
  //
  // return convoluted state
  //
  return convoluteWithMaterial(propStateWP);
}

std::pair<TrajectoryStateOnSurface, double> GsfPropagatorWithMaterial::propagateWithPath(const FreeTrajectoryState& fts,
                                                                                         const Plane& plane) const {
  static std::atomic<int> nWarn(0);
  if (nWarn++ < 5)
    edm::LogInfo("GsfPropagatorWithMaterial")
        << "GsfPropagatorWithMaterial used from FTS: input state might have been collapsed!";
  TsosWP propStateWP = theGeometricalPropagator->propagateWithPath(fts, plane);
  if (!(propStateWP.first).isValid() || materialAtSource())
    return propStateWP;
  //
  // return convoluted state
  //
  return convoluteWithMaterial(propStateWP);
}

std::pair<TrajectoryStateOnSurface, double> GsfPropagatorWithMaterial::propagateWithPath(
    const FreeTrajectoryState& fts, const Cylinder& cylinder) const {
  static std::atomic<int> nWarn(0);
  if (nWarn++ < 5)
    edm::LogInfo("GsfPropagatorWithMaterial")
        << "GsfPropagatorWithMaterial used from FTS: input state might have been collapsed!";
  TsosWP propStateWP = theGeometricalPropagator->propagateWithPath(fts, cylinder);
  if (!(propStateWP.first).isValid() || materialAtSource())
    return propStateWP;
  //
  // return convoluted state
  //
  return convoluteWithMaterial(propStateWP);
}

void GsfPropagatorWithMaterial::setPropagationDirection(PropagationDirection dir) {
  theGeometricalPropagator->setPropagationDirection(dir);
  Propagator::setPropagationDirection(dir);
}

std::pair<TrajectoryStateOnSurface, double> GsfPropagatorWithMaterial::convoluteWithMaterial(
    const TsosWP& aStateWithPath) const {
  //
  // convolute with material
  //
  PropagationDirection propDir = PropagationDirectionFromPath()(aStateWithPath.second, propagationDirection());
  return TsosWP((*theConvolutor)(aStateWithPath.first, propDir), aStateWithPath.second);
}

TrajectoryStateOnSurface GsfPropagatorWithMaterial::convoluteStateWithMaterial(
    const TrajectoryStateOnSurface tsos, const PropagationDirection propDir) const {
  //
  // convolute with material
  //
  return (*theConvolutor)(tsos, propDir);
}

bool GsfPropagatorWithMaterial::materialAtSource() const {
  if (propagationDirection() == anyDirection) {
    if (theMaterialLocation != atDestination) {
      throw cms::Exception("LogicError") << "PropagatorWithMaterial: propagation direction = anyDirection is "
                                         << "incompatible with adding of material at source";
    }
  }
  return theMaterialLocation == atSource ||
         (theMaterialLocation == fromDirection && propagationDirection() == alongMomentum);
}
