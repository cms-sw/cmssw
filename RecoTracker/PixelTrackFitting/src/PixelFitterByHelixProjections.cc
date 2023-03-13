#include "RecoTracker/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CommonTools/Statistics/interface/LinearFit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/PixelTrackFitting/interface/RZLine.h"
#include "RecoTracker/PixelTrackFitting/interface/CircleFromThreePoints.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackErrorParam.h"
#include "DataFormats/GeometryVector/interface/Pi.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "CommonTools/Utils/interface/DynArray.h"

using namespace std;

namespace {

  int charge(DynArray<GlobalPoint> const& points) {
    // the cross product will tell me...
    float dir = (points[1].x() - points[0].x()) * (points[2].y() - points[1].y()) -
                (points[1].y() - points[0].y()) * (points[2].x() - points[1].x());

    /*
      GlobalVector v21 = points[1]-points[0];
      GlobalVector v32 = points[2]-points[1];
      float dphi = v32.phi() - v21.phi();
      while (dphi >  Geom::fpi()) dphi -=  Geom::ftwoPi();
      while (dphi < -Geom::fpi()) dphi +=  Geom::ftwoPi();
      return (dphi > 0) ? -1 : 1;
    */
    return (dir > 0) ? -1 : 1;
  }

  float cotTheta(const GlobalPoint& inner, const GlobalPoint& outer) {
    float dr = outer.perp() - inner.perp();
    float dz = outer.z() - inner.z();
    return (std::abs(dr) > 1.e-3f) ? dz / dr : 0;
  }

  inline float phi(float xC, float yC, int charge) { return (charge > 0) ? std::atan2(xC, -yC) : std::atan2(-xC, yC); }

  float zip(float d0, float phi_p, float curv, const GlobalPoint& pinner, const GlobalPoint& pouter) {
    //
    //phi = asin(r*rho/2) with asin(x) ~= x+x**3/(2*3) = x(1+x*x/6);
    //

    float phi0 = phi_p - Geom::fhalfPi();
    GlobalPoint pca(d0 * std::cos(phi0), d0 * std::sin(phi0), 0.);

    constexpr float o24 = 1.f / 24.f;
    float rho2 = curv * curv;
    float r1s = (pinner - pca).perp2();
    double phi1 = std::sqrt(r1s) * (curv * 0.5f) * (1.f + r1s * (rho2 * o24));
    float r2s = (pouter - pca).perp2();
    double phi2 = std::sqrt(r2s) * (curv * 0.5f) * (1.f + r2s * (rho2 * o24));
    double z1 = pinner.z();
    double z2 = pouter.z();

    if (fabs(curv) > 1.e-5)
      return z1 - phi1 / (phi1 - phi2) * (z1 - z2);
    else {
      double dr = std::max(std::sqrt(r2s) - std::sqrt(r1s), 1.e-5f);
      return z1 - std::sqrt(r1s) * (z2 - z1) / dr;
    }
  }
}  // namespace

PixelFitterByHelixProjections::PixelFitterByHelixProjections(const TrackerTopology* ttopo,
                                                             const MagneticField* field,
                                                             bool scaleErrorsForBPix1,
                                                             float scaleFactor)
    : theTopo(ttopo), theField(field), thescaleErrorsForBPix1(scaleErrorsForBPix1), thescaleFactor(scaleFactor) {}

std::unique_ptr<reco::Track> PixelFitterByHelixProjections::run(const std::vector<const TrackingRecHit*>& hits,
                                                                const TrackingRegion& region) const {
  std::unique_ptr<reco::Track> ret;

  int nhits = hits.size();
  if (nhits < 2)
    return ret;

  declareDynArray(GlobalPoint, nhits, points);
  declareDynArray(GlobalError, nhits, errors);
  declareDynArray(bool, nhits, isBarrel);

  for (int i = 0; i != nhits; ++i) {
    auto const& recHit = hits[i];
    points[i] = GlobalPoint(recHit->globalPosition().basicVector() - region.origin().basicVector());
    errors[i] = recHit->globalPositionError();
    isBarrel[i] = recHit->detUnit()->type().isBarrel();
  }

  CircleFromThreePoints circle = (nhits == 2) ? CircleFromThreePoints(GlobalPoint(0., 0., 0.), points[0], points[1])
                                              : CircleFromThreePoints(points[0], points[1], points[2]);

  float valPhi, valTip, valPt;

  int iCharge = charge(points);
  float curvature = circle.curvature();

  if ((curvature > 1.e-4) && (LIKELY(theField->inverseBzAtOriginInGeV()) > 0.01)) {
    float invPt = PixelRecoUtilities::inversePt(circle.curvature(), *theField);
    valPt = (invPt > 1.e-4f) ? 1.f / invPt : 1.e4f;
    CircleFromThreePoints::Vector2D center = circle.center();
    valTip = iCharge * (center.mag() - 1.f / curvature);
    valPhi = phi(center.x(), center.y(), iCharge);
  } else {
    valPt = 1.e4f;
    GlobalVector direction(points[1] - points[0]);
    valPhi = direction.barePhi();
    valTip = -points[0].x() * sin(valPhi) + points[0].y() * cos(valPhi);
  }

  float valCotTheta = cotTheta(points[0], points[1]);
  float valEta = std::asinh(valCotTheta);
  float valZip = zip(valTip, valPhi, curvature, points[0], points[1]);

  // Rescale down the error to take into accont the fact that the
  // inner pixel barrel layer for PhaseI is closer to the interaction
  // point. The effective scale factor has been derived by checking
  // that the pulls of the pixelVertices derived from the pixelTracks
  // have the correct mean and sigma.
  float errFactor = 1.;
  if (thescaleErrorsForBPix1 && (hits[0]->geographicalId().subdetId() == PixelSubdetector::PixelBarrel) &&
      (theTopo->pxbLayer(hits[0]->geographicalId()) == 1))
    errFactor = thescaleFactor;

  PixelTrackErrorParam param(valEta, valPt);
  float errValPt = errFactor * param.errPt();
  float errValCot = errFactor * param.errCot();
  float errValTip = errFactor * param.errTip();
  float errValPhi = errFactor * param.errPhi();
  float errValZip = errFactor * param.errZip();

  float chi2 = 0;
  if (nhits > 2) {
    RZLine rzLine(points, errors, isBarrel);
    chi2 = rzLine.chi2();
  }

  PixelTrackBuilder builder;
  Measurement1D pt(valPt, errValPt);
  Measurement1D phi(valPhi, errValPhi);
  Measurement1D cotTheta(valCotTheta, errValCot);
  Measurement1D tip(valTip, errValTip);
  Measurement1D zip(valZip, errValZip);

  ret.reset(builder.build(pt, phi, cotTheta, tip, zip, chi2, iCharge, hits, theField, region.origin()));
  return ret;
}
