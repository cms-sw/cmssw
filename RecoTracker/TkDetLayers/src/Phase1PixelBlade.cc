#include "Phase1PixelBlade.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "BladeShapeBuilderFromDet.h"
#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

Phase1PixelBlade::~Phase1PixelBlade() {}

Phase1PixelBlade::Phase1PixelBlade(vector<const GeomDet*>& frontDets, vector<const GeomDet*>& backDets)
    : GeometricSearchDet(true),
      theFrontDets(frontDets),
      theBackDets(backDets),
      front_radius_range_(std::make_pair(0, 0)),
      back_radius_range_(std::make_pair(0, 0)) {
  theDets.assign(theFrontDets.begin(), theFrontDets.end());
  theDets.insert(theDets.end(), theBackDets.begin(), theBackDets.end());

  theDiskSector = BladeShapeBuilderFromDet::build(theDets);
  theFrontDiskSector = BladeShapeBuilderFromDet::build(theFrontDets);
  theBackDiskSector = BladeShapeBuilderFromDet::build(theBackDets);

  //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "DEBUG INFO for Phase1PixelBlade";
  LogDebug("TkDetLayers") << "Blade z, perp, innerRadius, outerR[disk, front, back]: " << this->position().z() << " , "
                          << this->position().perp() << " , (" << theDiskSector->innerRadius() << " , "
                          << theDiskSector->outerRadius() << "), (" << theFrontDiskSector->innerRadius() << " , "
                          << theFrontDiskSector->outerRadius() << "), (" << theBackDiskSector->innerRadius() << " , "
                          << theBackDiskSector->outerRadius() << ")" << std::endl;

  for (vector<const GeomDet*>::const_iterator it = theFrontDets.begin(); it != theFrontDets.end(); it++) {
    LogDebug("TkDetLayers") << "frontDet phi,z,r: " << (*it)->position().phi() << " , " << (*it)->position().z()
                            << " , " << (*it)->position().perp() << " , "
                            << " rmin: " << (*it)->surface().rSpan().first
                            << " rmax: " << (*it)->surface().rSpan().second << std::endl;
  }

  for (vector<const GeomDet*>::const_iterator it = theBackDets.begin(); it != theBackDets.end(); it++) {
    LogDebug("TkDetLayers") << "backDet phi,z,r: " << (*it)->position().phi() << " , " << (*it)->position().z() << " , "
                            << (*it)->position().perp() << " , "
                            << " rmin: " << (*it)->surface().rSpan().first
                            << " rmax: " << (*it)->surface().rSpan().second << std::endl;
  }
}

const vector<const GeometricSearchDet*>& Phase1PixelBlade::components() const {
  throw DetLayerException("TOBRod doesn't have GeometricSearchDet components");
}

pair<bool, TrajectoryStateOnSurface> Phase1PixelBlade::compatible(const TrajectoryStateOnSurface& ts,
                                                                  const Propagator&,
                                                                  const MeasurementEstimator&) const {
  edm::LogError("TkDetLayers") << "temporary dummy implementation of Phase1PixelBlade::compatible()!!";
  return pair<bool, TrajectoryStateOnSurface>();
}

void Phase1PixelBlade::groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                                              const Propagator& prop,
                                              const MeasurementEstimator& est,
                                              std::vector<DetGroup>& result) const {
  SubLayerCrossings crossings;

  crossings = computeCrossings(tsos, prop.propagationDirection());
  if (!crossings.isValid())
    return;

  vector<DetGroup> closestResult;
  addClosest(tsos, prop, est, crossings.closest(), closestResult);

  if (closestResult.empty()) {
    vector<DetGroup> nextResult;
    addClosest(tsos, prop, est, crossings.other(), nextResult);
    if (nextResult.empty())
      return;

    //DetGroupElement nextGel( nextResult.front().front());
    //int crossingSide = LayerCrossingSide().endcapSide( nextGel.trajectoryState(), prop);

    DetGroupMerger::orderAndMergeTwoLevels(std::move(closestResult),
                                           std::move(nextResult),
                                           result,
                                           0,
                                           0);  //fixme gc patched for SLHC - already correctly sorted for SLHC
    //crossings.closestIndex(), crossingSide);
  } else {
    DetGroupElement closestGel(closestResult.front().front());
    float window = computeWindowSize(closestGel.det(), closestGel.trajectoryState(), est);

    searchNeighbors(tsos, prop, est, crossings.closest(), window, closestResult, false);

    vector<DetGroup> nextResult;
    searchNeighbors(tsos, prop, est, crossings.other(), window, nextResult, true);

    //int crossingSide = LayerCrossingSide().endcapSide( closestGel.trajectoryState(), prop);
    DetGroupMerger::orderAndMergeTwoLevels(std::move(closestResult),
                                           std::move(nextResult),
                                           result,
                                           0,
                                           0);  //fixme gc patched for SLHC - already correctly sorted for SLHC
    //crossings.closestIndex(), crossingSide);
  }
}

SubLayerCrossings Phase1PixelBlade::computeCrossings(const TrajectoryStateOnSurface& startingState,
                                                     PropagationDirection propDir) const {
  HelixPlaneCrossing::PositionType startPos(startingState.globalPosition());
  HelixPlaneCrossing::DirectionType startDir(startingState.globalMomentum());
  double rho(startingState.transverseCurvature());
  HelixArbitraryPlaneCrossing crossing(startPos, startDir, rho, propDir);

  pair<bool, double> innerPath = crossing.pathLength(*theFrontDiskSector);
  if (!innerPath.first)
    return SubLayerCrossings();

  GlobalPoint gInnerPoint(crossing.position(innerPath.second));
  //Code for use of binfinder
  //int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.perp());
  //float innerDist = std::abs( theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.z());

  //  int innerIndex = findBin(gInnerPoint.perp(),0);
  int innerIndex = findBin2(gInnerPoint, 0);

  //fixme gc patched for SLHC - force order here to be in z
  //float innerDist = std::abs( findPosition(innerIndex,0).perp() - gInnerPoint.perp());
  float innerDist = (startingState.globalPosition() - gInnerPoint).mag();
  SubLayerCrossing innerSLC(0, innerIndex, gInnerPoint);

  pair<bool, double> outerPath = crossing.pathLength(*theBackDiskSector);
  if (!outerPath.first)
    return SubLayerCrossings();

  GlobalPoint gOuterPoint(crossing.position(outerPath.second));
  //Code for use of binfinder
  //int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.perp());
  //float outerDist = std::abs( theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.perp());
  //  int outerIndex  = findBin(gOuterPoint.perp(),1);
  int outerIndex = findBin2(gOuterPoint, 1);

  //fixme gc patched for SLHC - force order here to be in z
  //float outerDist = std::abs( findPosition(outerIndex,1).perp() - gOuterPoint.perp());
  float outerDist = (startingState.globalPosition() - gOuterPoint).mag();
  SubLayerCrossing outerSLC(1, outerIndex, gOuterPoint);

  if (innerDist < outerDist) {
    return SubLayerCrossings(innerSLC, outerSLC, 0);
  } else {
    return SubLayerCrossings(outerSLC, innerSLC, 1);
  }
}

bool Phase1PixelBlade::addClosest(const TrajectoryStateOnSurface& tsos,
                                  const Propagator& prop,
                                  const MeasurementEstimator& est,
                                  const SubLayerCrossing& crossing,
                                  vector<DetGroup>& result) const {
  const vector<const GeomDet*>& sBlade(subBlade(crossing.subLayerIndex()));

  return CompatibleDetToGroupAdder().add(*sBlade[crossing.closestDetIndex()], tsos, prop, est, result);
}

float Phase1PixelBlade::computeWindowSize(const GeomDet* det,
                                          const TrajectoryStateOnSurface& tsos,
                                          const MeasurementEstimator& est) const {
  return est.maximalLocalDisplacement(tsos, det->surface()).x();
}

void Phase1PixelBlade::searchNeighbors(const TrajectoryStateOnSurface& tsos,
                                       const Propagator& prop,
                                       const MeasurementEstimator& est,
                                       const SubLayerCrossing& crossing,
                                       float window,
                                       vector<DetGroup>& result,
                                       bool checkClosest) const {
  const GlobalPoint& gCrossingPos = crossing.position();
  const vector<const GeomDet*>& sBlade(subBlade(crossing.subLayerIndex()));

  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex - 1;
  int posStartIndex = closestIndex + 1;

  if (checkClosest) {  // must decide if the closest is on the neg or pos side
    if (gCrossingPos.perp() < sBlade[closestIndex]->surface().position().perp()) {
      posStartIndex = closestIndex;
    } else {
      negStartIndex = closestIndex;
    }
  }

  typedef CompatibleDetToGroupAdder Adder;
  for (int idet = negStartIndex; idet >= 0; idet--) {
    if (!overlap(gCrossingPos, *sBlade[idet], window))
      break;
    if (!Adder::add(*sBlade[idet], tsos, prop, est, result))
      break;
  }
  for (int idet = posStartIndex; idet < static_cast<int>(sBlade.size()); idet++) {
    if (!overlap(gCrossingPos, *sBlade[idet], window))
      break;
    if (!Adder::add(*sBlade[idet], tsos, prop, est, result))
      break;
  }
}

bool Phase1PixelBlade::overlap(const GlobalPoint& crossPoint, const GeomDet& det, float window) const {
  // check if the z window around TSOS overlaps with the detector theDet (with a 1% margin added)

  //   const float tolerance = 0.1;
  const float relativeMargin = 1.01;

  LocalPoint localCrossPoint(det.surface().toLocal(crossPoint));
  //   if (std::abs(localCrossPoint.z()) > tolerance) {
  //     edm::LogInfo(TkDetLayers) << "Phase1PixelBlade::overlap calculation assumes point on surface, but it is off by "
  // 	 << localCrossPoint.z() ;
  //   }

  float localX = localCrossPoint.x();
  float detHalfLength = det.surface().bounds().length() / 2.;

  //   edm::LogInfo(TkDetLayers) << "Phase1PixelBlade::overlap: Det at " << det.position() << " hit at " << localY
  //        << " Window " << window << " halflength "  << detHalfLength ;

  if ((std::abs(localX) - window) < relativeMargin * detHalfLength) {  // FIXME: margin hard-wired!
    return true;
  } else {
    return false;
  }
}

int Phase1PixelBlade::findBin(float R, int diskSectorIndex) const {
  vector<const GeomDet*> localDets = diskSectorIndex == 0 ? theFrontDets : theBackDets;

  int theBin = 0;
  float rDiff = std::abs(R - localDets.front()->surface().position().perp());
  for (vector<const GeomDet*>::const_iterator i = localDets.begin(); i != localDets.end(); i++) {
    float testDiff = std::abs(R - (**i).surface().position().perp());
    if (testDiff < rDiff) {
      rDiff = testDiff;
      theBin = i - localDets.begin();
    }
  }
  return theBin;
}

int Phase1PixelBlade::findBin2(GlobalPoint thispoint, int diskSectorIndex) const {
  const vector<const GeomDet*>& localDets = diskSectorIndex == 0 ? theFrontDets : theBackDets;

  int theBin = 0;
  float sDiff = (thispoint - localDets.front()->surface().position()).mag();

  for (vector<const GeomDet*>::const_iterator i = localDets.begin(); i != localDets.end(); i++) {
    float testDiff = (thispoint - (**i).surface().position()).mag();
    if (testDiff < sDiff) {
      sDiff = testDiff;
      theBin = i - localDets.begin();
    }
  }
  return theBin;
}

GlobalPoint Phase1PixelBlade::findPosition(int index, int diskSectorType) const {
  vector<const GeomDet*> diskSector = diskSectorType == 0 ? theFrontDets : theBackDets;
  return (diskSector[index])->surface().position();
}

std::pair<float, float> Phase1PixelBlade::computeRadiusRanges(const std::vector<const GeomDet*>& current_dets) {
  typedef Surface::PositionType::BasicVectorType Vector;
  Vector posSum(0, 0, 0);
  for (auto i : current_dets)
    posSum += (*i).surface().position().basicVector();

  Surface::PositionType meanPos(0., 0., posSum.z() / float(current_dets.size()));

  // temporary plane - for the computation of bounds
  const Plane& tmpplane = current_dets.front()->surface();

  GlobalVector xAxis;
  GlobalVector yAxis;
  GlobalVector zAxis;

  GlobalVector planeXAxis = tmpplane.toGlobal(LocalVector(1, 0, 0));
  const GlobalPoint& planePosition = tmpplane.position();

  if (planePosition.x() * planeXAxis.x() + planePosition.y() * planeXAxis.y() > 0.) {
    yAxis = planeXAxis;
  } else {
    yAxis = -planeXAxis;
  }

  GlobalVector planeZAxis = tmpplane.toGlobal(LocalVector(0, 0, 1));
  if (planeZAxis.z() * planePosition.z() > 0.) {
    zAxis = planeZAxis;
  } else {
    zAxis = -planeZAxis;
  }

  xAxis = yAxis.cross(zAxis);

  Surface::RotationType rotation = Surface::RotationType(xAxis, yAxis);
  Plane plane(meanPos, rotation);

  Surface::PositionType tmpPos = current_dets.front()->surface().position();

  float rmin(plane.toLocal(tmpPos).perp());
  float rmax(plane.toLocal(tmpPos).perp());

  for (auto it : current_dets) {
    vector<GlobalPoint> corners = BoundingBox().corners(it->specificSurface());
    for (const auto& i : corners) {
      float r = plane.toLocal(i).perp();
      rmin = min(rmin, r);
      rmax = max(rmax, r);
    }
  }

  return std::make_pair(rmin, rmax);
}
