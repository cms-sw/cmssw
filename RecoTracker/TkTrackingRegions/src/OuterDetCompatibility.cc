#include "OuterDetCompatibility.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

using namespace std;

bool OuterDetCompatibility::operator() (const BoundPlane& plane) const
{
  if (barrel) {
    if (!checkPhi(plane.phiSpan())) return false;
    if (!checkZ(plane.zSpan())) return false;
  } else {
    if (!checkPhi(plane.phiSpan())) return false;
    if (!checkR(plane.rSpan())) return false;
  }
  return true;
}


bool OuterDetCompatibility::checkPhi(
    const OuterHitPhiPrediction::Range & detPhiRange) const
{ return rangesIntersect(detPhiRange, hitDetPhiRange,
        [](auto x, auto y){ return Geom::phiLess(x,y); }); }

bool OuterDetCompatibility::checkR(
    const Range & detRRange) const
{ return rangesIntersect(detRRange, hitDetRRange); }

bool OuterDetCompatibility::checkZ(
    const Range & detZRange) const
{ return rangesIntersect(detZRange, hitDetZRange); }


GlobalPoint OuterDetCompatibility::center() const
{
  float phi = hitDetPhiRange.mean();
  float r   = hitDetRRange.mean();
  return GlobalPoint( r*cos(phi), r*sin(phi), hitDetZRange.mean() );
}

MeasurementEstimator::Local2DVector
    OuterDetCompatibility::maximalLocalDisplacement(
       const GlobalPoint & ts, const BoundPlane& plane) const
{
  float x_loc = 0.;
  float y_loc = 0.;
  if(barrel) {
    double  radius = ts.perp();
    GlobalVector planeNorm = plane.normalVector();
    GlobalVector tsDir = GlobalVector( ts.x(), ts.y(),0. ).unit();
    double ts_phi = tsDir.phi();
    if (! hitDetPhiRange.inside(ts_phi) ) {
     while (ts_phi >= hitDetPhiRange.max() ) ts_phi -= 2*M_PI;
     while (ts_phi < hitDetPhiRange.min() ) ts_phi += 2*M_PI;
     if (!hitDetPhiRange.inside(ts_phi)) return MeasurementEstimator::Local2DVector(0.,0.);
    }
    double cosGamma = tsDir.dot(planeNorm);

    double dx1 = loc_dist( radius, ts_phi, hitDetPhiRange.min(), cosGamma);
    double dx2 = loc_dist( radius, ts_phi, hitDetPhiRange.max(), cosGamma);

    double ts_z = ts.z();
    double dy1 = ts_z - hitDetZRange.min();
    double dy2 = hitDetZRange.max() - ts_z;

    x_loc = max(dx1,dx2);
    y_loc = max(dy1,dy2);

    // debug only
/*
    double r1 = dx1 * fabs(cosGamma) / sin(ts_phi-hitDetPhiRange.min());
    double r2 = dx2 * fabs(cosGamma) / sin(hitDetPhiRange.max()-ts_phi);
    GlobalPoint p1( r1* cos(hitDetPhiRange.min()), r1 * sin(hitDetPhiRange.min()), hitDetZRange.min());
    GlobalPoint p2( r2* cos(hitDetPhiRange.max()), r2 * sin(hitDetPhiRange.max()), hitDetZRange.min());
    GlobalPoint p3( r1* cos(hitDetPhiRange.min()), r1 * sin(hitDetPhiRange.min()), hitDetZRange.max());
    GlobalPoint p4( r2* cos(hitDetPhiRange.max()), r2 * sin(hitDetPhiRange.max()), hitDetZRange.max());
    cout << " Local1: " << plane.toLocal(ts-p1) << endl;
    cout << " Local2: " << plane.toLocal(ts-p2) << endl;
    cout << " Local3: " << plane.toLocal(ts-p3) << endl;
    cout << " Local4: " << plane.toLocal(ts-p4) << endl;
*/
  }
  else {
    LocalPoint ts_loc = plane.toLocal(ts);
    GlobalVector planeNorm = plane.normalVector();

    double x_glob[4], y_glob[4], z_glob[4];
    x_glob[0] = hitDetRRange.min()*cos(hitDetPhiRange.min());
    y_glob[0] = hitDetRRange.min()*sin(hitDetPhiRange.min());
    x_glob[1] = hitDetRRange.max()*cos(hitDetPhiRange.min());
    y_glob[1] = hitDetRRange.max()*sin(hitDetPhiRange.min());
    x_glob[2] = hitDetRRange.min()*cos(hitDetPhiRange.max());
    y_glob[2] = hitDetRRange.min()*sin(hitDetPhiRange.max());
    x_glob[3] = hitDetRRange.max()*cos(hitDetPhiRange.max());
    y_glob[3] = hitDetRRange.max()*sin(hitDetPhiRange.max());

    for (int idx = 0; idx < 4; idx++) {
      double dx_glob = x_glob[idx] - ts.x();
      double dy_glob = y_glob[idx] - ts.y();
      double dz_glob = -(dx_glob * planeNorm.x() + dy_glob*planeNorm.y()) / planeNorm.z();
      z_glob[idx] = dz_glob + ts.z();
    }

    for (int idx=0; idx <4; idx++) {
      LocalPoint lp = plane.toLocal( GlobalPoint( x_glob[idx], y_glob[idx], z_glob[idx]));
      x_loc = max(x_loc, fabs(lp.x()-ts_loc.x()));
      y_loc = max(y_loc, fabs(lp.y()-ts_loc.y())); 
    }
  }
  MeasurementEstimator::Local2DVector distance(x_loc,y_loc);
  return distance;
}

double OuterDetCompatibility::loc_dist(
      double radius, double ts_phi, double range_phi, double cosGamma) const
{
    double sinDphi = sin(ts_phi - range_phi);
    double cosDphi = sqrt(1-sinDphi*sinDphi);
    double sinGamma = sqrt(1-cosGamma*cosGamma);
    double sinBeta = fabs(cosDphi*cosGamma -  sinDphi* sinGamma); 
    return radius * fabs(sinDphi) / sinBeta; 
}
