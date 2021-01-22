#include "Phase2OTBarrelRod.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

namespace {
  class DetGroupElementPerpLess {
  public:
    bool operator()(DetGroup a, DetGroup b) const {
      return (a.front().det()->position().perp() < b.front().det()->position().perp());
    }
  };
}  // namespace

Phase2OTBarrelRod::Phase2OTBarrelRod(vector<const GeomDet*>& innerDets,
                                     vector<const GeomDet*>& outerDets,
                                     const vector<const GeomDet*>& innerDetBrothers,
                                     const vector<const GeomDet*>& outerDetBrothers)
    : DetRod(true),
      theInnerDets(innerDets),
      theOuterDets(outerDets),
      theInnerDetBrothers(innerDetBrothers),
      theOuterDetBrothers(outerDetBrothers) {
  theDets.assign(theInnerDets.begin(), theInnerDets.end());
  theDets.insert(theDets.end(), theOuterDets.begin(), theOuterDets.end());
  theDets.insert(theDets.end(), theInnerDetBrothers.begin(), theInnerDetBrothers.end());
  theDets.insert(theDets.end(), theOuterDetBrothers.begin(), theOuterDetBrothers.end());

  RodPlaneBuilderFromDet planeBuilder;
  setPlane(planeBuilder(theDets));
  theInnerPlane = planeBuilder(theInnerDets);
  theOuterPlane = planeBuilder(theOuterDets);

  // modules be already sorted in z

  theInnerBinFinder = BinFinderType(theInnerDets.begin(), theInnerDets.end());
  theOuterBinFinder = BinFinderType(theOuterDets.begin(), theOuterDets.end());

#ifdef EDM_ML_DEBUG
  LogDebug("TkDetLayers") << "==== DEBUG Phase2OTBarrelRod =====";
  for (vector<const GeomDet*>::const_iterator i = theInnerDets.begin(); i != theInnerDets.end(); i++) {
    LogDebug("TkDetLayers") << "inner Phase2OTBarrelRod's Det pos z,perp,eta,phi: " << (**i).position().z() << " , "
                            << (**i).position().perp() << " , " << (**i).position().eta() << " , "
                            << (**i).position().phi();
  }

  for (vector<const GeomDet*>::const_iterator i = theInnerDetBrothers.begin(); i != theInnerDetBrothers.end(); i++) {
    LogDebug("TkDetLayers") << "inner Phase2OTBarrelRod's Det Brother pos z,perp,eta,phi: " << (**i).position().z()
                            << " , " << (**i).position().perp() << " , " << (**i).position().eta() << " , "
                            << (**i).position().phi();
  }
  if (theInnerDetBrothers.empty() && theOuterDetBrothers.empty())
    LogDebug("TkDetLayers") << "====       with stacks       =====";
  if (!theInnerDetBrothers.empty() && !theOuterDetBrothers.empty())
    LogDebug("TkDetLayers") << "====     without stacks      =====";

  for (vector<const GeomDet*>::const_iterator i = theOuterDets.begin(); i != theOuterDets.end(); i++) {
    LogDebug("TkDetLayers") << "outer Phase2OTBarrelRod's Det pos z,perp,eta,phi: " << (**i).position().z() << " , "
                            << (**i).position().perp() << " , " << (**i).position().eta() << " , "
                            << (**i).position().phi();
  }

  for (vector<const GeomDet*>::const_iterator i = theOuterDetBrothers.begin(); i != theOuterDetBrothers.end(); i++) {
    LogDebug("TkDetLayers") << "outer Phase2OTBarrelRod's Det Brother pos z,perp,eta,phi: " << (**i).position().z()
                            << " , " << (**i).position().perp() << " , " << (**i).position().eta() << " , "
                            << (**i).position().phi();
  }
  LogDebug("TkDetLayers") << "==== end DEBUG Phase2OTBarrelRod =====";
#endif
}

Phase2OTBarrelRod::~Phase2OTBarrelRod() {}

const vector<const GeometricSearchDet*>& Phase2OTBarrelRod::components() const {
  throw DetLayerException("Phase2OTBarrelRod doesn't have GeometricSearchDet components");
}

pair<bool, TrajectoryStateOnSurface> Phase2OTBarrelRod::compatible(const TrajectoryStateOnSurface& ts,
                                                                   const Propagator&,
                                                                   const MeasurementEstimator&) const {
  edm::LogError("TkDetLayers") << "temporary dummy implementation of Phase2OTBarrelRod::compatible()!!";
  return pair<bool, TrajectoryStateOnSurface>();
}

void Phase2OTBarrelRod::groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                                               const Propagator& prop,
                                               const MeasurementEstimator& est,
                                               std::vector<DetGroup>& result) const {
  SubLayerCrossings crossings;
  crossings = computeCrossings(tsos, prop.propagationDirection());
  if (!crossings.isValid())
    return;

  std::vector<DetGroup> closestResult;
  std::vector<DetGroup> closestBrotherResult;
  addClosest(tsos, prop, est, crossings.closest(), closestResult, closestBrotherResult);
  if (closestResult.empty()) {
    std::vector<DetGroup> nextResult;
    std::vector<DetGroup> nextBrotherResult;
    addClosest(tsos, prop, est, crossings.other(), nextResult, nextBrotherResult);
    if (nextResult.empty())
      return;

    DetGroupElement nextGel(nextResult.front().front());
    int crossingSide = LayerCrossingSide().barrelSide(nextGel.trajectoryState(), prop);
    std::vector<DetGroup> closestCompleteResult;
    DetGroupMerger::orderAndMergeTwoLevels(
        std::move(closestResult), std::move(closestBrotherResult), closestCompleteResult, 0, crossingSide);
    std::vector<DetGroup> nextCompleteResult;
    DetGroupMerger::orderAndMergeTwoLevels(
        std::move(nextResult), std::move(nextBrotherResult), nextCompleteResult, 0, crossingSide);

    DetGroupMerger::orderAndMergeTwoLevels(
        std::move(closestCompleteResult), std::move(nextCompleteResult), result, crossings.closestIndex(), crossingSide);
  } else {
    DetGroupElement closestGel(closestResult.front().front());
    int crossingSide = LayerCrossingSide().barrelSide(closestGel.trajectoryState(), prop);
    float window = computeWindowSize(closestGel.det(), closestGel.trajectoryState(), est);

    searchNeighbors(tsos, prop, est, crossings.closest(), window, closestResult, closestBrotherResult, false);

    std::vector<DetGroup> closestCompleteResult;
    DetGroupMerger::orderAndMergeTwoLevels(
        std::move(closestResult), std::move(closestBrotherResult), closestCompleteResult, 0, crossingSide);

    std::vector<DetGroup> nextResult;
    std::vector<DetGroup> nextBrotherResult;
    searchNeighbors(tsos, prop, est, crossings.other(), window, nextResult, nextBrotherResult, true);

    std::vector<DetGroup> nextCompleteResult;
    DetGroupMerger::orderAndMergeTwoLevels(
        std::move(nextResult), std::move(nextBrotherResult), nextCompleteResult, 0, crossingSide);

    DetGroupMerger::orderAndMergeTwoLevels(
        std::move(closestCompleteResult), std::move(nextCompleteResult), result, crossings.closestIndex(), crossingSide);
  }

  //due to propagator problems, when we add single pt sub modules, we should order them in r (barrel)
  sort(result.begin(), result.end(), DetGroupElementPerpLess());
  for (auto& grp : result) {
    if (grp.empty())
      continue;
    LogTrace("TkDetLayers") << "New group in Phase2OTBarrelRod made by : ";
    for (auto const& det : grp) {
      LogTrace("TkDetLayers") << " geom det at r: " << det.det()->position().perp()
                              << " id:" << det.det()->geographicalId().rawId()
                              << " tsos at:" << det.trajectoryState().globalPosition();
    }
  }
}

SubLayerCrossings Phase2OTBarrelRod::computeCrossings(const TrajectoryStateOnSurface& startingState,
                                                      PropagationDirection propDir) const {
  GlobalPoint startPos(startingState.globalPosition());
  GlobalVector startDir(startingState.globalMomentum());
  double rho(startingState.transverseCurvature());

  HelixBarrelPlaneCrossingByCircle crossing(startPos, startDir, rho, propDir);

  std::pair<bool, double> outerPath = crossing.pathLength(*theOuterPlane);
  if (!outerPath.first)
    return SubLayerCrossings();
  GlobalPoint gOuterPoint(crossing.position(outerPath.second));

  std::pair<bool, double> innerPath = crossing.pathLength(*theInnerPlane);
  if (!innerPath.first)
    return SubLayerCrossings();
  GlobalPoint gInnerPoint(crossing.position(innerPath.second));

  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.z());
  float innerDist = std::abs(theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.z());
  SubLayerCrossing innerSLC(0, innerIndex, gInnerPoint);

  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.z());
  float outerDist = std::abs(theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.z());
  SubLayerCrossing outerSLC(1, outerIndex, gOuterPoint);

  if (innerDist < outerDist) {
    return SubLayerCrossings(innerSLC, outerSLC, 0);
  } else {
    return SubLayerCrossings(outerSLC, innerSLC, 1);
  }
}

bool Phase2OTBarrelRod::addClosest(const TrajectoryStateOnSurface& tsos,
                                   const Propagator& prop,
                                   const MeasurementEstimator& est,
                                   const SubLayerCrossing& crossing,
                                   vector<DetGroup>& result,
                                   vector<DetGroup>& brotherresult) const {
  const vector<const GeomDet*>& sRod(subRod(crossing.subLayerIndex()));
  bool firstgroup = CompatibleDetToGroupAdder::add(*sRod[crossing.closestDetIndex()], tsos, prop, est, result);
  if (theInnerDetBrothers.empty() && theOuterDetBrothers.empty())
    return firstgroup;

  // it assumes that the closestDetIndex is ok also for the brother detectors: the crossing is NOT recomputed
  const vector<const GeomDet*>& sRodBrothers(subRodBrothers(crossing.subLayerIndex()));
  bool brothergroup =
      CompatibleDetToGroupAdder::add(*sRodBrothers[crossing.closestDetIndex()], tsos, prop, est, brotherresult);

  return firstgroup || brothergroup;
}

float Phase2OTBarrelRod::computeWindowSize(const GeomDet* det,
                                           const TrajectoryStateOnSurface& tsos,
                                           const MeasurementEstimator& est) const {
  return est.maximalLocalDisplacement(tsos, det->surface()).y();
}

namespace {

  inline bool overlap(const GlobalPoint& crossPoint, const GeomDet& det, float window) {
    // check if the z window around TSOS overlaps with the detector theDet (with a 1% margin added)

    //   const float tolerance = 0.1;
    constexpr float relativeMargin = 1.01;

    LocalPoint localCrossPoint(det.surface().toLocal(crossPoint));
    //   if (std::abs(localCrossPoint.z()) > tolerance) {
    //     edm::LogInfo(TkDetLayers) << "TOBRod::overlap calculation assumes point on surface, but it is off by "
    // 	 << localCrossPoint.z() ;
    //   }

    float localY = localCrossPoint.y();
    float detHalfLength = 0.5f * det.surface().bounds().length();

    //   edm::LogInfo(TkDetLayers) << "TOBRod::overlap: Det at " << det.position() << " hit at " << localY
    //        << " Window " << window << " halflength "  << detHalfLength ;

    return (std::abs(localY) - window) < relativeMargin * detHalfLength;
  }

}  // namespace

void Phase2OTBarrelRod::searchNeighbors(const TrajectoryStateOnSurface& tsos,
                                        const Propagator& prop,
                                        const MeasurementEstimator& est,
                                        const SubLayerCrossing& crossing,
                                        float window,
                                        vector<DetGroup>& result,
                                        vector<DetGroup>& brotherresult,
                                        bool checkClosest) const {
  const GlobalPoint& gCrossingPos = crossing.position();

  const vector<const GeomDet*>& sRod(subRod(crossing.subLayerIndex()));
  const vector<const GeomDet*>& sBrotherRod(subRodBrothers(crossing.subLayerIndex()));

  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex - 1;
  int posStartIndex = closestIndex + 1;

  if (checkClosest) {  // must decide if the closest is on the neg or pos side
    if (gCrossingPos.z() < sRod[closestIndex]->surface().position().z()) {
      posStartIndex = closestIndex;
    } else {
      negStartIndex = closestIndex;
    }
  }

  typedef CompatibleDetToGroupAdder Adder;
  for (int idet = negStartIndex; idet >= 0; idet--) {
    if (!overlap(gCrossingPos, *sRod[idet], window))
      break;
    if (!Adder::add(*sRod[idet], tsos, prop, est, result))
      break;
    if (theInnerDetBrothers.empty() && theOuterDetBrothers.empty())
      break;
    // If the two above checks are passed also the brother module will be added with no further checks
    Adder::add(*sBrotherRod[idet], tsos, prop, est, brotherresult);
  }
  for (int idet = posStartIndex; idet < static_cast<int>(sRod.size()); idet++) {
    if (!overlap(gCrossingPos, *sRod[idet], window))
      break;
    if (!Adder::add(*sRod[idet], tsos, prop, est, result))
      break;
    if (theInnerDetBrothers.empty() && theOuterDetBrothers.empty())
      break;
    // If the two above checks are passed also the brother module will be added with no further checks
    Adder::add(*sBrotherRod[idet], tsos, prop, est, brotherresult);
  }
}
