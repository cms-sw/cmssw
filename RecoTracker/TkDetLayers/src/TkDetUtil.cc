#include "TkDetUtil.h"

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

namespace tkDetUtil {

  float computeWindowSize(const GeomDet* det, const TrajectoryStateOnSurface& tsos, const MeasurementEstimator& est) {
    const Plane& startPlane = det->surface();
    auto maxDistance = est.maximalLocalDisplacement(tsos, startPlane);
    return std::copysign(calculatePhiWindow(maxDistance, tsos, startPlane), maxDistance.x());
  }

  float calculatePhiWindow(const MeasurementEstimator::Local2DVector& imaxDistance,
                           const TrajectoryStateOnSurface& ts,
                           const Plane& plane) {
    MeasurementEstimator::Local2DVector maxDistance(std::abs(imaxDistance.x()), std::abs(imaxDistance.y()));

    constexpr float tolerance = 1.e-6;
    LocalPoint start = ts.localPosition();
    //     std::cout << "plane z " << plane.normalVector() << std::endl;
    float dphi = 0;
    if LIKELY (std::abs(1.f - std::abs(plane.normalVector().z())) < tolerance) {
      auto ori = plane.toLocal(GlobalPoint(0., 0., 0.));
      auto xc = std::abs(start.x() - ori.x());
      auto yc = std::abs(start.y() - ori.y());

      if (yc < maxDistance.y() && xc < maxDistance.x())
        return M_PI;

      auto hori = yc > maxDistance.y();  // quadrant 1 (&2), otherwiase quadrant 1&4
      auto y0 = hori ? yc + std::copysign(maxDistance.y(), xc - maxDistance.x()) : xc - maxDistance.x();
      auto x0 = hori ? xc - maxDistance.x() : -yc - maxDistance.y();
      auto y1 = hori ? yc - maxDistance.y() : xc - maxDistance.x();
      auto x1 = hori ? xc + maxDistance.x() : -yc + maxDistance.y();

      auto sp = (x0 * x1 + y0 * y1) / std::sqrt((x0 * x0 + y0 * y0) * (x1 * x1 + y1 * y1));
      sp = std::min(std::max(sp, -1.f), 1.f);
      dphi = std::acos(sp);

      return dphi;
    }

    // generic algo
    float corners[] = {plane.toGlobal(LocalPoint(start.x() + maxDistance.x(), start.y() + maxDistance.y())).barePhi(),
                       plane.toGlobal(LocalPoint(start.x() - maxDistance.x(), start.y() + maxDistance.y())).barePhi(),
                       plane.toGlobal(LocalPoint(start.x() - maxDistance.x(), start.y() - maxDistance.y())).barePhi(),
                       plane.toGlobal(LocalPoint(start.x() + maxDistance.x(), start.y() - maxDistance.y())).barePhi()};

    float phimin = corners[0];
    float phimax = phimin;
    for (int i = 1; i < 4; i++) {
      float cPhi = corners[i];
      if (Geom::phiLess(cPhi, phimin)) {
        phimin = cPhi;
      }
      if (Geom::phiLess(phimax, cPhi)) {
        phimax = cPhi;
      }
    }
    float phiWindow = phimax - phimin;
    if (phiWindow < 0.) {
      phiWindow += 2. * Geom::pi();
    }
    // std::cout << "phiWindow " << phiWindow << ' ' << dphi << ' ' << dphi-phiWindow  << std::endl;
    return phiWindow;
  }

  float computeYdirWindowSize(const GeomDet* det,
                              const TrajectoryStateOnSurface& tsos,
                              const MeasurementEstimator& est) {
    const Plane& startPlane = det->surface();
    MeasurementEstimator::Local2DVector maxDistance = est.maximalLocalDisplacement(tsos, startPlane);
    return maxDistance.y();
  }

  std::array<int, 3> findThreeClosest(const std::vector<RingPar>& ringParams,
                                      const std::vector<GlobalPoint>& ringCrossing,
                                      const int ringSize) {
    std::array<int, 3> theBins = {{-1, -1, -1}};
    theBins[0] = 0;
    float initialR = ringParams[0].theRingR;
    float rDiff0 = std::abs(ringCrossing[0].perp() - initialR);
    float rDiff1 = -1.;
    float rDiff2 = -1.;
    for (int i = 1; i < ringSize; i++) {
      float ringR = ringParams[i].theRingR;
      float testDiff = std::abs(ringCrossing[i].perp() - ringR);
      if (testDiff < rDiff0) {
        rDiff2 = rDiff1;
        rDiff1 = rDiff0;
        rDiff0 = testDiff;
        theBins[2] = theBins[1];
        theBins[1] = theBins[0];
        theBins[0] = i;
      } else if (rDiff1 < 0 || testDiff < rDiff1) {
        rDiff2 = rDiff1;
        rDiff1 = testDiff;
        theBins[2] = theBins[1];
        theBins[1] = i;
      } else if (rDiff2 < 0 || testDiff < rDiff2) {
        rDiff2 = testDiff;
        theBins[2] = i;
      }
    }

    return theBins;
  }

  bool overlapInR(const TrajectoryStateOnSurface& tsos, int index, double ymax, const std::vector<RingPar>& ringParams) {
    // assume "fixed theta window", i.e. margin in local y = r is changing linearly with z
    float tsRadius = tsos.globalPosition().perp();
    float thetamin =
        (std::max(0., tsRadius - ymax)) / (std::abs(tsos.globalPosition().z()) + 10.f);  // add 10 cm contingency
    float thetamax = (tsRadius + ymax) / (std::abs(tsos.globalPosition().z()) - 10.f);

    // do the theta regions overlap ?

    return !(thetamin > ringParams[index].thetaRingMax || ringParams[index].thetaRingMin > thetamax);
  }

  RingPar fillRingParametersFromDisk(const BoundDisk& ringDisk) {
    float ringMinZ = std::abs(ringDisk.position().z()) - ringDisk.bounds().thickness() / 2.;
    float ringMaxZ = std::abs(ringDisk.position().z()) + ringDisk.bounds().thickness() / 2.;
    RingPar tempPar;
    tempPar.thetaRingMin = ringDisk.innerRadius() / ringMaxZ;
    tempPar.thetaRingMax = ringDisk.outerRadius() / ringMinZ;
    tempPar.theRingR = (ringDisk.innerRadius() + ringDisk.outerRadius()) / 2.;
    return tempPar;
  }

}  // namespace tkDetUtil
