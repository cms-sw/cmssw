#include "TBLayer.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"

TBLayer::~TBLayer() {
  for (auto i : theComps)
    delete i;
}

void TBLayer::groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                                     const Propagator& prop,
                                     const MeasurementEstimator& est,
                                     std::vector<DetGroup>& result) const {
  SubLayerCrossings crossings;
  crossings = computeCrossings(tsos, prop.propagationDirection());
  if (!crossings.isValid())
    return;

  std::vector<DetGroup> closestResult;
  addClosest(tsos, prop, est, crossings.closest(), closestResult);
  // for TIB this differs from compatibleDets logic, which checks next in such cases!!!
  if (closestResult.empty()) {
    if (!isTIB())
      addClosest(tsos, prop, est, crossings.other(), result);
    return;
  }

  DetGroupElement closestGel(closestResult.front().front());
  float window = computeWindowSize(closestGel.det(), closestGel.trajectoryState(), est);

  searchNeighbors(tsos, prop, est, crossings.closest(), window, closestResult, false);

  std::vector<DetGroup> nextResult;
  searchNeighbors(tsos, prop, est, crossings.other(), window, nextResult, true);

  int crossingSide = LayerCrossingSide().barrelSide(closestGel.trajectoryState(), prop);
  DetGroupMerger::orderAndMergeTwoLevels(
      std::move(closestResult), std::move(nextResult), result, crossings.closestIndex(), crossingSide);
}

SubLayerCrossings TBLayer::computeCrossings(const TrajectoryStateOnSurface& startingState,
                                            PropagationDirection propDir) const {
  GlobalPoint startPos(startingState.globalPosition());
  GlobalVector startDir(startingState.globalMomentum());
  double rho(startingState.transverseCurvature());

  bool inBetween = ((theOuterCylinder->position() - startPos).perp() < theOuterCylinder->radius()) &&
                   ((theInnerCylinder->position() - startPos).perp() > theInnerCylinder->radius());

  HelixBarrelCylinderCrossing innerCrossing(
      startPos, startDir, rho, propDir, *theInnerCylinder, HelixBarrelCylinderCrossing::onlyPos);
  if (!inBetween && !innerCrossing.hasSolution())
    return SubLayerCrossings();

  HelixBarrelCylinderCrossing outerCrossing(
      startPos, startDir, rho, propDir, *theOuterCylinder, HelixBarrelCylinderCrossing::onlyPos);

  if (!innerCrossing.hasSolution() && outerCrossing.hasSolution()) {
    innerCrossing = outerCrossing;
  } else if (!outerCrossing.hasSolution() && innerCrossing.hasSolution()) {
    outerCrossing = innerCrossing;
  }

  GlobalPoint gInnerPoint(innerCrossing.position());
  GlobalPoint gOuterPoint(outerCrossing.position());

  int innerIndex, outerIndex;
  bool inLess;
  std::tie(inLess, innerIndex, outerIndex) = computeIndexes(gInnerPoint, gOuterPoint);

  SubLayerCrossing innerSLC(0, innerIndex, gInnerPoint);
  SubLayerCrossing outerSLC(1, outerIndex, gOuterPoint);

  if (inLess) {
    return SubLayerCrossings(innerSLC, outerSLC, 0);
  } else {
    return SubLayerCrossings(outerSLC, innerSLC, 1);
  }
}

bool TBLayer::addClosest(const TrajectoryStateOnSurface& tsos,
                         const Propagator& prop,
                         const MeasurementEstimator& est,
                         const SubLayerCrossing& crossing,
                         std::vector<DetGroup>& result) const {
  auto const& sub = subLayer(crossing.subLayerIndex());
  auto det = sub[crossing.closestDetIndex()];
  return CompatibleDetToGroupAdder().add(*det, tsos, prop, est, result);
}
