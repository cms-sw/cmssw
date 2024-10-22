#ifndef TkDetLayers_TkDetUtil_h
#define TkDetLayers_TkDetUtil_h

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"

#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "DetGroupMerger.h"

class GeomDet;
class Plane;
class TrajectoryStateOnSurface;

#pragma GCC visibility push(hidden)

namespace tkDetUtil {

  struct RingPar {
    float theRingR, thetaRingMin, thetaRingMax;
  };

  inline bool overlapInPhi(float phi, const GeomDet& det, float phiWindow) {
    std::pair<float, float> phiRange(phi - phiWindow, phi + phiWindow);
    return rangesIntersect(phiRange, det.surface().phiSpan(), [](auto x, auto y) { return Geom::phiLess(x, y); });
  }

  inline bool overlapInPhi(GlobalPoint crossPoint, const GeomDet& det, float phiWindow) {
    return overlapInPhi(crossPoint.barePhi(), det, phiWindow);
  }

  float computeWindowSize(const GeomDet* det, const TrajectoryStateOnSurface& tsos, const MeasurementEstimator& est);

  float calculatePhiWindow(const MeasurementEstimator::Local2DVector& maxDistance,
                           const TrajectoryStateOnSurface& ts,
                           const Plane& plane);

  float computeYdirWindowSize(const GeomDet* det,
                              const TrajectoryStateOnSurface& tsos,
                              const MeasurementEstimator& est);

  std::array<int, 3> findThreeClosest(const std::vector<RingPar>& ringParams,
                                      const std::vector<GlobalPoint>& ringCrossing,
                                      const int ringSize);

  bool overlapInR(const TrajectoryStateOnSurface& tsos, int index, double ymax, const std::vector<RingPar>& ringParams);

  RingPar fillRingParametersFromDisk(const BoundDisk& ringDisk);

  template <class T>
  std::array<int, 3> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
                                                    const Propagator& prop,
                                                    const int ringSize,
                                                    const T& diskComponents,
                                                    const std::vector<RingPar>& ringParams) {
    typedef HelixForwardPlaneCrossing Crossing;
    typedef MeasurementEstimator::Local2DVector Local2DVector;

    HelixPlaneCrossing::PositionType startPos(startingState.globalPosition());
    HelixPlaneCrossing::DirectionType startDir(startingState.globalMomentum());
    PropagationDirection propDir(prop.propagationDirection());
    float rho(startingState.transverseCurvature());

    // calculate the crossings with the ring surfaces
    // rings are assumed to be sorted in R !

    Crossing myXing(startPos, startDir, rho, propDir);

    std::vector<GlobalPoint> ringCrossings;
    ringCrossings.reserve(ringSize);

    for (int i = 0; i < ringSize; i++) {
      const BoundDisk& theRing = static_cast<const BoundDisk&>(diskComponents[i]->surface());
      std::pair<bool, double> pathlen = myXing.pathLength(theRing);
      if (pathlen.first) {
        ringCrossings.push_back(GlobalPoint(myXing.position(pathlen.second)));
      } else {
        // TO FIX.... perhaps there is something smarter to do
        ringCrossings.push_back(GlobalPoint(0., 0., 0.));
      }
    }

    //find three closest rings to the crossing

    std::array<int, 3> closests = findThreeClosest(ringParams, ringCrossings, ringSize);

    return closests;
  }

  template <class T>
  void groupedCompatibleDetsV(const TrajectoryStateOnSurface& startingState,
                              const Propagator& prop,
                              const MeasurementEstimator& est,
                              std::vector<DetGroup>& result,
                              const int ringSize,
                              const std::vector<const T*>& diskComponents,
                              const std::vector<RingPar>& ringParams) {
    std::array<int, 3> const& ringIndices =
        ringIndicesByCrossingProximity(startingState, prop, ringSize, diskComponents, ringParams);
    if (ringIndices[0] == -1 || ringIndices[1] == -1 || ringIndices[2] == -1) {
      edm::LogError("TkDetLayers") << "TkRingedForwardLayer::groupedCompatibleDets : error in CrossingProximity";
      return;
    }

    //order is: rings in front = 0; rings in back = 1
    //rings should be already ordered in r
    //if the layer has 1 ring, it does not matter
    //FIXME: to be optimized once the geometry is stable
    std::vector<int> ringOrder(ringSize);
    std::fill(ringOrder.begin(), ringOrder.end(), 1);
    if (ringSize > 1) {
      if (std::abs(diskComponents[0]->position().z()) < std::abs(diskComponents[1]->position().z())) {
        for (int i = 0; i < ringSize; i++) {
          if (i % 2 == 0)
            ringOrder[i] = 0;
        }
      } else if (std::abs(diskComponents[0]->position().z()) > std::abs(diskComponents[1]->position().z())) {
        std::fill(ringOrder.begin(), ringOrder.end(), 0);
        for (int i = 0; i < ringSize; i++) {
          if (i % 2 == 0)
            ringOrder[i] = 1;
        }
      } else {
        throw DetLayerException("Rings in Endcap Layer have same z position, no idea how to order them!");
      }
    }

    auto index = [&ringIndices, &ringOrder](int i) { return ringOrder[ringIndices[i]]; };

    std::vector<DetGroup> closestResult;
    diskComponents[ringIndices[0]]->groupedCompatibleDetsV(startingState, prop, est, closestResult);
    // if the closest is empty, use the next one and exit: inherited from TID !
    if (closestResult.empty()) {
      diskComponents[ringIndices[1]]->groupedCompatibleDetsV(startingState, prop, est, result);
      return;
    }

    DetGroupElement closestGel(closestResult.front().front());
    float rWindow = computeYdirWindowSize(closestGel.det(), closestGel.trajectoryState(), est);

    // check if next ring and next next ring are found and if there is overlap

    bool ring1ok =
        ringIndices[1] != -1 && overlapInR(closestGel.trajectoryState(), ringIndices[1], rWindow, ringParams);
    bool ring2ok =
        ringIndices[2] != -1 && overlapInR(closestGel.trajectoryState(), ringIndices[2], rWindow, ringParams);

    // look for the two rings in the same plane (are they only two?)

    // determine if we are propagating from in to out (0) or from out to in (1)

    int direction = 0;
    if (startingState.globalPosition().z() * startingState.globalMomentum().z() > 0) {
      if (prop.propagationDirection() == alongMomentum)
        direction = 0;
      else
        direction = 1;
    } else {
      if (prop.propagationDirection() == alongMomentum)
        direction = 1;
      else
        direction = 0;
    }

    if ((index(0) == index(1)) && (index(0) == index(2))) {
      edm::LogInfo("AllRingsInOnePlane") << " All rings: " << ringIndices[0] << " " << ringIndices[1] << " "
                                         << ringIndices[2] << " in one plane. Only the first two will be considered";
      ring2ok = false;
    }

    if (index(0) == index(1)) {
      if (ring1ok) {
        std::vector<DetGroup> ring1res;
        diskComponents[ringIndices[1]]->groupedCompatibleDetsV(startingState, prop, est, ring1res);
        DetGroupMerger::addSameLevel(std::move(ring1res), closestResult);
      }
      if (ring2ok) {
        std::vector<DetGroup> ring2res;
        diskComponents[ringIndices[2]]->groupedCompatibleDetsV(startingState, prop, est, ring2res);
        DetGroupMerger::orderAndMergeTwoLevels(
            std::move(closestResult), std::move(ring2res), result, index(0), direction);
        return;
      } else {
        result.swap(closestResult);
        return;
      }
    } else if (index(0) == index(2)) {
      if (ring2ok) {
        std::vector<DetGroup> ring2res;
        diskComponents[ringIndices[2]]->groupedCompatibleDetsV(startingState, prop, est, ring2res);
        DetGroupMerger::addSameLevel(std::move(ring2res), closestResult);
      }
      if (ring1ok) {
        std::vector<DetGroup> ring1res;
        diskComponents[ringIndices[1]]->groupedCompatibleDetsV(startingState, prop, est, ring1res);
        DetGroupMerger::orderAndMergeTwoLevels(
            std::move(closestResult), std::move(ring1res), result, index(0), direction);
        return;
      } else {
        result.swap(closestResult);
        return;
      }
    } else {
      std::vector<DetGroup> ring12res;
      if (ring1ok) {
        std::vector<DetGroup> ring1res;
        diskComponents[ringIndices[1]]->groupedCompatibleDetsV(startingState, prop, est, ring1res);
        ring12res.swap(ring1res);
      }
      if (ring2ok) {
        std::vector<DetGroup> ring2res;
        diskComponents[ringIndices[2]]->groupedCompatibleDetsV(startingState, prop, est, ring2res);
        DetGroupMerger::addSameLevel(std::move(ring2res), ring12res);
      }
      if (!ring12res.empty()) {
        DetGroupMerger::orderAndMergeTwoLevels(
            std::move(closestResult), std::move(ring12res), result, index(0), direction);
        return;
      } else {
        result.swap(closestResult);
        return;
      }
    }
  }

  template <class T>
  BoundDisk* computeDisk(const std::vector<const T*>& structures) {
    float theRmin = structures.front()->specificSurface().innerRadius();
    float theRmax = structures.front()->specificSurface().outerRadius();
    float theZmin = structures.front()->position().z() - structures.front()->surface().bounds().thickness() / 2;
    float theZmax = structures.front()->position().z() + structures.front()->surface().bounds().thickness() / 2;

    for (typename std::vector<const T*>::const_iterator i = structures.begin(); i != structures.end(); i++) {
      float rmin = (**i).specificSurface().innerRadius();
      float rmax = (**i).specificSurface().outerRadius();
      float zmin = (**i).position().z() - (**i).surface().bounds().thickness() / 2.;
      float zmax = (**i).position().z() + (**i).surface().bounds().thickness() / 2.;
      theRmin = std::min(theRmin, rmin);
      theRmax = std::max(theRmax, rmax);
      theZmin = std::min(theZmin, zmin);
      theZmax = std::max(theZmax, zmax);
    }

    float zPos = (theZmax + theZmin) / 2.;
    Plane::PositionType pos(0., 0., zPos);
    Plane::RotationType rot;

    return new BoundDisk(pos, rot, new SimpleDiskBounds(theRmin, theRmax, theZmin - zPos, theZmax - zPos));
  }

}  // namespace tkDetUtil

#pragma GCC visibility pop
#endif  // TkDetLayers_TkDetUtil_h
