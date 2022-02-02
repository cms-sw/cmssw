#ifndef TkDetLayers_TkDetUtil_h
#define TkDetLayers_TkDetUtil_h

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

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

  std::array<int, 3> findThreeClosest(std::vector<RingPar> ringParams,
                                      std::vector<GlobalPoint> ringCrossing,
                                      const int ringSize);

  bool overlapInR(const TrajectoryStateOnSurface& tsos, int index, double ymax, std::vector<RingPar> ringParams);

  RingPar fillRingParametersFromDisk(const BoundDisk& ringDisk);

}  // namespace tkDetUtil

#pragma GCC visibility pop
#endif  // TkDetLayers_TkDetUtil_h
