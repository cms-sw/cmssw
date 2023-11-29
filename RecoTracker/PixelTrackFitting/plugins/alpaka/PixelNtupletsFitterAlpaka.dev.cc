#include <alpaka/alpaka.hpp>
#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/PixelNtupletsFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackErrorParam.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/RiemannFit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

using namespace std;
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  PixelNtupletsFitter::PixelNtupletsFitter(Queue& queue, float nominalB, const MagneticField* field, bool useRiemannFit)
      : queue_(queue), nominalB_(nominalB), field_(field), useRiemannFit_(useRiemannFit) {}

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

    ALPAKA_ASSERT_OFFLOAD(nhits == 4);
    auto hits_gp_h = cms::alpakatools::make_host_buffer<riemannFit::Matrix3xNd<4>>(queue_);

    Eigen::Matrix<float, 6, 4> hits_ge_h = Eigen::Matrix<float, 6, 4>::Zero();

    for (unsigned int i = 0; i < nhits; ++i) {
      hits_gp_h.data()->col(i) << points[i].x(), points[i].y(), points[i].z();

      hits_ge_h.col(i) << errors[i].cxx(), errors[i].cyx(), errors[i].cyy(), errors[i].czx(), errors[i].czy(),
          errors[i].czz();
    }
    auto fittedTrack_d = cms::alpakatools::make_device_buffer<HelixFit>(queue_);
    auto workdiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    auto hits_gp_d = cms::alpakatools::make_device_buffer<riemannFit::Matrix3xNd<4>>(queue_);
    auto hits_ge_d = cms::alpakatools::make_device_buffer<Eigen::Matrix<float, 6, 4>>(queue_);

    alpaka::memcpy(queue_, hits_gp_d, hits_gp_h);
    auto hits_ge_h_view = cms::alpakatools::make_host_view(hits_ge_h);
    alpaka::memcpy(queue_, hits_ge_d, hits_ge_h_view);

    useRiemannFit_ ? alpaka::exec<Acc1D>(queue_,
                                         workdiv,
                                         riemannFit::helixFit<4>{},
                                         hits_gp_d.data(),
                                         hits_ge_d.data(),
                                         nominalB_,
                                         true,
                                         fittedTrack_d.data())
                   : alpaka::exec<Acc1D>(queue_,
                                         workdiv,
                                         brokenline::helixFit<4>{},
                                         hits_gp_d.data(),
                                         hits_ge_d.data(),
                                         nominalB_,
                                         fittedTrack_d.data());

    auto fittedTrack_h = cms::alpakatools::make_host_buffer<HelixFit>(queue_);
    alpaka::memcpy(queue_, fittedTrack_h, fittedTrack_d);
    int iCharge = fittedTrack_h.data()->qCharge;

    // parameters are:
    // 0: phi
    // 1: tip
    // 2: curvature
    // 3: cottheta
    // 4: zip
    float valPhi = fittedTrack_h.data()->par(0);

    float valTip = fittedTrack_h.data()->par(1);

    float valCotTheta = fittedTrack_h.data()->par(3);

    float valZip = fittedTrack_h.data()->par(4);
    float valPt = fittedTrack_h.data()->par(2);
    //
    //  PixelTrackErrorParam param(valEta, valPt);
    float errValPhi = std::sqrt(fittedTrack_h.data()->cov(0, 0));
    float errValTip = std::sqrt(fittedTrack_h.data()->cov(1, 1));

    float errValPt = std::sqrt(fittedTrack_h.data()->cov(2, 2));

    float errValCotTheta = std::sqrt(fittedTrack_h.data()->cov(3, 3));
    float errValZip = std::sqrt(fittedTrack_h.data()->cov(4, 4));

    float chi2 = fittedTrack_h.data()->chi2_line + fittedTrack_h.data()->chi2_circle;

    PixelTrackBuilder builder;
    Measurement1D phi(valPhi, errValPhi);
    Measurement1D tip(valTip, errValTip);

    Measurement1D pt(valPt, errValPt);
    Measurement1D cotTheta(valCotTheta, errValCotTheta);
    Measurement1D zip(valZip, errValZip);

    ret.reset(builder.build(pt, phi, cotTheta, tip, zip, chi2, iCharge, hits, field_, region.origin()));
    return ret;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
