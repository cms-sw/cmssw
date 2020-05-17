#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"

using namespace std;

// Warning, remember to assign this pointer to a ReferenceCountingPointer!
BoundDisk* ForwardRingDiskBuilderFromDet::operator()(const vector<const GeomDet*>& dets) const {
  auto bo = computeBounds(dets);

  //   LogDebug("DetLayers") << "Creating disk at Z: " << bo.second << "\n"
  //        << "Bounds are (rmin/rmax/thick) " << bo.first.innerRadius()
  //        << " / " <<  bo.first.outerRadius()
  //        << " / " <<  bo.first.thickness()  ;

  //   typedef Det::PositionType::BasicVectorType Vector;
  //   Vector posSum(0,0,0);
  //   for (vector<Det*>::const_iterator i=dets.begin(); i!=dets.end(); i++) {
  //     Vector pp = (**i).position().basicVector();
  //     //    LogDebug("DetLayers") << "  "<< (int) ( i-dets.begin()) << " at " << pp ;
  //     posSum += pp;
  //   }
  //   Det::PositionType meanPos( posSum/float(dets.size()));
  //   LogDebug("DetLayers") << "  meanPos "<< meanPos ;

  Surface::PositionType pos(0., 0., bo.second);
  Surface::RotationType rot;
  return new BoundDisk(pos, rot, bo.first);
}

pair<SimpleDiskBounds*, float> ForwardRingDiskBuilderFromDet::computeBounds(const vector<const GeomDet*>& dets) const {
  // go over all corners and compute maximum deviations from mean pos.
  float rmin((**(dets.begin())).surface().position().perp());
  float rmax(rmin);
  float zmin((**(dets.begin())).surface().position().z());
  float zmax(zmin);
  for (auto det : dets) {
    /* ---- original implementation. Is it obsolete?
    vector<DetUnit*> detUnits = (**idet).detUnits();
    for (vector<DetUnit*>::const_iterator detu=detUnits.begin();
    detu!=detUnits.end(); detu++) {
    vector<GlobalPoint> corners = BoundingBox().corners(
    dynamic_cast<const Plane&>((**detu).surface()));
    }
    ----- */
    vector<GlobalPoint> corners = BoundingBox().corners((*det).specificSurface());
    for (const auto& corner : corners) {
      float r = corner.perp();
      float z = corner.z();
      rmin = min(rmin, r);
      rmax = max(rmax, r);
      zmin = min(zmin, z);
      zmax = max(zmax, z);
    }
    // in addition to the corners we have to check the middle of the
    // det +/- length/2, since the min (max) radius for typical fw
    // dets is reached there

    float rdet = (*det).position().perp();
    float len = (*det).surface().bounds().length();
    float width = (*det).surface().bounds().width();

    GlobalVector xAxis = (*det).toGlobal(LocalVector(1, 0, 0));
    GlobalVector yAxis = (*det).toGlobal(LocalVector(0, 1, 0));
    GlobalVector perpDir = GlobalVector((*det).position() - GlobalPoint(0, 0, (*det).position().z()));

    double xAxisCos = xAxis.unit().dot(perpDir.unit());
    double yAxisCos = yAxis.unit().dot(perpDir.unit());

    if (fabs(xAxisCos) > fabs(yAxisCos)) {
      rmin = min(rmin, rdet - width / 2.F);
      rmax = max(rmax, rdet + width / 2.F);
    } else {
      rmin = min(rmin, rdet - len / 2.F);
      rmax = max(rmax, rdet + len / 2.F);
    }
  }

  float zPos = (zmax + zmin) / 2.;
  return make_pair(new SimpleDiskBounds(rmin, rmax, zmin - zPos, zmax - zPos), zPos);
}
