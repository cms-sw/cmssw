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

}  // namespace tkDetUtil
