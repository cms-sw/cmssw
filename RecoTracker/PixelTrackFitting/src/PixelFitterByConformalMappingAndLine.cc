#include "RecoTracker/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "CommonTools/Statistics/interface/LinearFit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "ConformalMappingFit.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoTracker/PixelTrackFitting/interface/RZLine.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"

using namespace std;

template <class T>
T sqr(T t) {
  return t * t;
}

PixelFitterByConformalMappingAndLine::PixelFitterByConformalMappingAndLine(
    const TransientTrackingRecHitBuilder *ttrhBuilder,
    const TrackerGeometry *tracker,
    const MagneticField *field,
    double fixImpactParameter,
    bool useFixImpactParameter)
    : theTTRHBuilder(ttrhBuilder),
      theTracker(tracker),
      theField(field),
      theFixImpactParameter(fixImpactParameter),
      theUseFixImpactParameter(useFixImpactParameter) {}

std::unique_ptr<reco::Track> PixelFitterByConformalMappingAndLine::run(const std::vector<const TrackingRecHit *> &hits,
                                                                       const TrackingRegion &region) const {
  int nhits = hits.size();

  vector<GlobalPoint> points;
  vector<GlobalError> errors;
  vector<bool> isBarrel;

  for (vector<const TrackingRecHit *>::const_iterator ih = hits.begin(); ih != hits.end(); ih++) {
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(*ih);
    points.push_back(recHit->globalPosition());
    errors.push_back(recHit->globalPositionError());
    isBarrel.push_back(recHit->detUnit()->type().isBarrel());
  }

  //    if (useMultScatt) {
  //      MultipleScatteringParametrisation ms(hits[i].layer());
  //      float cotTheta = (p.z()-zVtx)/p.perp();
  //      err += sqr( ms( pt, cotTheta, PixelRecoPointRZ(0.,zVtx) ) );
  //    }

  //
  // simple fit to get pt, phi0 used for precise calcul.
  //
  typedef ConformalMappingFit::PointXY PointXY;
  vector<PointXY> xy;
  vector<float> errRPhi2;
  for (int i = 0; i < nhits; ++i) {
    const GlobalPoint &point = points[i];
    xy.push_back(PointXY(point.x() - region.origin().x(), point.y() - region.origin().y()));
    float phiErr2 = errors[i].phierr(point);
    errRPhi2.push_back(point.perp2() * phiErr2);
  }
  ConformalMappingFit parabola(xy, errRPhi2);
  if (theUseFixImpactParameter)
    parabola.fixImpactParmaeter(theFixImpactParameter);
  else if (nhits < 3)
    parabola.fixImpactParmaeter(0.);

  Measurement1D curv = parabola.curvature();
  float invPt = PixelRecoUtilities::inversePt(curv.value(), *theField);
  float valPt = (invPt > 1.e-4) ? 1. / invPt : 1.e4;
  float errPt = PixelRecoUtilities::inversePt(curv.error(), *theField) * sqr(valPt);
  Measurement1D pt(valPt, errPt);
  Measurement1D phi = parabola.directionPhi();
  Measurement1D tip = parabola.impactParameter();

  //
  // precalculate theta to correct errors:
  //
  vector<float> r(nhits), z(nhits), errZ(nhits);
  float simpleCot = (points.back().z() - points.front().z()) / (points.back().perp() - points.front().perp());
  for (int i = 0; i < nhits; ++i) {
    const GlobalPoint &point = points[i];
    const GlobalError &error = errors[i];
    r[i] = sqrt(sqr(point.x() - region.origin().x()) + sqr(point.y() - region.origin().y()));
    r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt.value(), *theField)(r[i]);
    z[i] = point.z() - region.origin().z();
    errZ[i] = (isBarrel[i]) ? sqrt(error.czz()) : sqrt(error.rerr(point)) * simpleCot;
  }

  //
  // line fit (R-Z plane)
  //
  RZLine rzLine(r, z, errZ);

  //
  // parameters for track builder
  //
  Measurement1D zip(rzLine.intercept(), sqrt(rzLine.covii()));
  Measurement1D cotTheta(rzLine.cotTheta(), sqrt(rzLine.covss()));
  float chi2 = parabola.chi2() + rzLine.chi2();
  int charge = parabola.charge();

  PixelTrackBuilder builder;
  return std::unique_ptr<reco::Track>(
      builder.build(pt, phi, cotTheta, tip, zip, chi2, charge, hits, theField, region.origin()));
}
