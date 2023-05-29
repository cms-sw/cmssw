#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelNtupletsFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackErrorParam.h"
#include "RecoTracker/PixelTrackFitting/interface/RiemannFit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

using namespace std;

PixelNtupletsFitter::PixelNtupletsFitter(float nominalB, const MagneticField* field, bool useRiemannFit)
    : nominalB_(nominalB), field_(field), useRiemannFit_(useRiemannFit) {}

std::unique_ptr<reco::Track> PixelNtupletsFitter::run(const std::vector<const TrackingRecHit*>& hits,
                                                      const TrackingRegion& region) const {
  using namespace riemannFit;

  std::unique_ptr<reco::Track> ret;

  unsigned int nhits = hits.size();

  if (nhits < 2)
    return ret;

  declareDynArray(GlobalPoint, nhits, points);
  declareDynArray(GlobalError, nhits, errors);
  declareDynArray(bool, nhits, isBarrel);

  for (unsigned int i = 0; i != nhits; ++i) {
    auto const& recHit = hits[i];
    points[i] = GlobalPoint(recHit->globalPosition().basicVector() - region.origin().basicVector());
    errors[i] = recHit->globalPositionError();
    isBarrel[i] = recHit->detUnit()->type().isBarrel();
  }

  assert(nhits == 4);
  riemannFit::Matrix3xNd<4> hits_gp;

  Eigen::Matrix<float, 6, 4> hits_ge = Eigen::Matrix<float, 6, 4>::Zero();

  for (unsigned int i = 0; i < nhits; ++i) {
    hits_gp.col(i) << points[i].x(), points[i].y(), points[i].z();

    hits_ge.col(i) << errors[i].cxx(), errors[i].cyx(), errors[i].cyy(), errors[i].czx(), errors[i].czy(),
        errors[i].czz();
  }

  HelixFit fittedTrack = useRiemannFit_ ? riemannFit::helixFit(hits_gp, hits_ge, nominalB_, true)
                                        : brokenline::helixFit(hits_gp, hits_ge, nominalB_);

  int iCharge = fittedTrack.qCharge;

  // parameters are:
  // 0: phi
  // 1: tip
  // 2: curvature
  // 3: cottheta
  // 4: zip
  float valPhi = fittedTrack.par(0);

  float valTip = fittedTrack.par(1);

  float valCotTheta = fittedTrack.par(3);

  float valZip = fittedTrack.par(4);
  float valPt = fittedTrack.par(2);
  //
  //  PixelTrackErrorParam param(valEta, valPt);
  float errValPhi = std::sqrt(fittedTrack.cov(0, 0));
  float errValTip = std::sqrt(fittedTrack.cov(1, 1));

  float errValPt = std::sqrt(fittedTrack.cov(2, 2));

  float errValCotTheta = std::sqrt(fittedTrack.cov(3, 3));
  float errValZip = std::sqrt(fittedTrack.cov(4, 4));

  float chi2 = fittedTrack.chi2_line + fittedTrack.chi2_circle;

  PixelTrackBuilder builder;
  Measurement1D phi(valPhi, errValPhi);
  Measurement1D tip(valTip, errValTip);

  Measurement1D pt(valPt, errValPt);
  Measurement1D cotTheta(valCotTheta, errValCotTheta);
  Measurement1D zip(valZip, errValZip);

  ret.reset(builder.build(pt, phi, cotTheta, tip, zip, chi2, iCharge, hits, field_, region.origin()));
  return ret;
}
