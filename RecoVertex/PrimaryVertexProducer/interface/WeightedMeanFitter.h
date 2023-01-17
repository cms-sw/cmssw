#ifndef RecoVertex_PrimaryVertexProducer_WeightedMeanFitter_h
#define RecoVertex_PrimaryVertexProducer_WeightedMeanFitter_h

#include <vector>
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

namespace WeightedMeanFitter {

  constexpr float startError = 20.0;
  constexpr float precision = 1e-24;
  constexpr float corr_x = 1.2;
  constexpr float corr_x_bs = 1.0;  // corr_x for beam spot
  constexpr float corr_z = 1.4;
  constexpr int maxIterations = 50;
  constexpr float muSquare = 9.;

  inline std::pair<GlobalPoint, double> nearestPoint(const GlobalPoint& vertex, reco::Track iclus) {
    double ox = iclus.vx();
    double oy = iclus.vy();
    double oz = iclus.vz();

    double vx = iclus.px();
    double vy = iclus.py();
    double vz = iclus.pz();

    double opx = vertex.x() - ox;
    double opy = vertex.y() - oy;
    double opz = vertex.z() - oz;

    double vnorm2 = (vx * vx + vy * vy + vz * vz);
    double t = (vx * opx + vy * opy + vz * opz) / (vnorm2);

    GlobalPoint p(ox + t * vx, oy + t * vy, oz + t * vz);
    return std::pair<GlobalPoint, double>(
        p,
        std::sqrt(std::pow(p.x() - vertex.x(), 2) + std::pow(p.y() - vertex.y(), 2) + std::pow(p.z() - vertex.z(), 2)));
  }

  inline TransientVertex weightedMeanOutlierRejection(const std::vector<std::pair<GlobalPoint, GlobalPoint>>& points,
                                                      std::vector<reco::TransientTrack> iclus) {
    float x = 0., y = 0., z = 0.;
    float s_wx = 0., s_wz = 0.;
    float s2_wx = 0., s2_wz = 0.;
    float wx = 0., wz = 0., chi2 = 0.;
    float ndof_x = 0.;

    AlgebraicSymMatrix33 err;
    err(0, 0) = startError / 10 * startError / 10;
    err(1, 1) = startError / 10 * startError / 10;
    err(2, 2) = startError * startError;  // error is 20 cm, so cov -> is 20 ^ 2
    for (const auto& p : points) {
      wx = p.second.x() <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(p.second.x(), 2);

      wz = p.second.z() <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(p.second.z(), 2);

      x += p.first.x() * wx;
      y += p.first.y() * wx;
      z += p.first.z() * wz;

      s_wx += wx;
      s_wz += wz;
    }

    if (s_wx == 0. || s_wz == 0.) {
      edm::LogWarning("WeightedMeanFitter") << "Vertex fitting failed at beginning\n";
      return TransientVertex(GlobalPoint(0, 0, 0), err, iclus, 0, 0);
    }

    x /= s_wx;
    y /= s_wx;
    z /= s_wz;

    float old_x, old_y, old_z;

    float xpull;
    int niter = 0;

    float err_x, err_z;

    err_x = 1. / s_wx;
    err_z = 1. / s_wz;

    while ((niter++) < 2) {
      old_x = x;
      old_y = y;
      old_z = z;
      s_wx = 0;
      s_wz = 0;
      s2_wx = 0;
      s2_wz = 0;
      x = 0;
      y = 0;
      z = 0;
      ndof_x = 0;

      for (unsigned int i = 0; i < (unsigned int)points.size(); i++) {
        std::pair<GlobalPoint, double> p = nearestPoint(GlobalPoint(old_x, old_y, old_z), (iclus)[i].track());

        wx = points[i].second.x() <= precision ? std::pow(precision, 2) : std::pow(points[i].second.x(), 2);
        wz = points[i].second.z() <= precision ? std::pow(precision, 2) : std::pow(points[i].second.z(), 2);

        xpull = 0.;

        if (std::pow(p.first.x() - old_x, 2) / (wx + err_x) < muSquare &&
            std::pow(p.first.y() - old_y, 2) / (wx + err_x) < muSquare &&
            std::pow(p.first.z() - old_z, 2) / (wz + err_z) < muSquare)
          xpull = 1.;

        ndof_x += xpull;

        wx = xpull / wx;
        wz = xpull / wz;

        x += wx * p.first.x();
        y += wx * p.first.y();
        z += wz * p.first.z();

        s_wx += wx;
        s_wz += wz;

        s2_wx += wx * xpull;
        s2_wz += wz * xpull;
      }

      if (s_wx == 0. || s_wz == 0.) {
        edm::LogWarning("WeightedMeanFitter")
            << "Vertex fitting failed, either all tracks are outliers or they have a very large error\n";
        return TransientVertex(GlobalPoint(0, 0, 0), err, iclus, 0, 0);
      }
      x /= s_wx;
      y /= s_wx;
      z /= s_wz;

      err_x = (s2_wx / std::pow(s_wx, 2));
      err_z = (s2_wz / std::pow(s_wz, 2));

      if (std::abs(x - old_x) < (precision / 1.) && std::abs(y - old_y) < (precision / 1.) &&
          std::abs(z - old_z) < (precision / 1.)) {
        break;
      }
    }

    err(0, 0) = err_x * corr_x * corr_x;
    err(1, 1) = err_x * corr_x * corr_x;
    err(2, 2) = err_z * corr_z * corr_z;

    float dist = 0;
    for (const auto& p : points) {
      wx = p.second.x();
      wx = wx <= precision ? precision : wx;

      wz = p.second.z();
      wz = wz <= precision ? precision : wz;

      dist = std::pow(p.first.x() - x, 2) / (std::pow(wx, 2) + err(0, 0));
      dist += std::pow(p.first.y() - y, 2) / (std::pow(wx, 2) + err(1, 1));
      dist += std::pow(p.first.z() - z, 2) / (std::pow(wz, 2) + err(2, 2));
      chi2 += dist;
    }
    TransientVertex v(GlobalPoint(x, y, z), err, iclus, chi2, (int)ndof_x);
    return v;
  }

  inline TransientVertex weightedMeanOutlierRejectionBeamSpot(
      const std::vector<std::pair<GlobalPoint, GlobalPoint>>& points,
      std::vector<reco::TransientTrack> iclus,
      const reco::BeamSpot& beamSpot) {
    float x = 0., y = 0., z = 0.;
    float s_wx = 0., s_wz = 0.;
    float s2_wx = 0., s2_wz = 0.;
    float wx = 0., wz = 0., chi2 = 0.;
    float wy = 0., s_wy = 0., s2_wy = 0.;
    float ndof_x = 0.;

    AlgebraicSymMatrix33 err;
    err(0, 0) = startError / 10 * startError / 10;
    err(1, 1) = startError / 10 * startError / 10;
    err(2, 2) = startError * startError;  // error is 20 cm, so cov -> is 20 ^ 2

    GlobalError bse(beamSpot.rotatedCovariance3D());
    GlobalPoint bsp(Basic3DVector<float>(beamSpot.position()));

    for (const auto& p : points) {
      wx = p.second.x() <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(p.second.x(), 2);
      wy = p.second.y() <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(p.second.y(), 2);

      wz = p.second.z() <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(p.second.z(), 2);

      x += p.first.x() * wx;
      y += p.first.y() * wy;
      z += p.first.z() * wz;

      s_wx += wx;
      s_wy += wy;
      s_wz += wz;
    }

    if (s_wx == 0. || s_wy == 0. || s_wz == 0.) {
      edm::LogWarning("WeightedMeanFitter") << "Vertex fitting failed at beginning\n";
      return TransientVertex(GlobalPoint(0, 0, 0), err, iclus, 0, 0);
    }
    // use the square of covariance element to increase it's weight: it will be the most important
    wx = bse.cxx() <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(bse.cxx(), 2);
    wy = bse.cyy() <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(bse.cyy(), 2);

    x += bsp.x() * wx;
    y += bsp.y() * wy;

    x /= (s_wx + wx);
    y /= (s_wy + wy);
    z /= s_wz;

    float old_x, old_y, old_z;

    float xpull;
    int niter = 0;

    float err_x, err_y, err_z;

    err_x = 1. / s_wx;
    err_y = 1. / s_wy;
    err_z = 1. / s_wz;

    while ((niter++) < 2) {
      old_x = x;
      old_y = y;
      old_z = z;
      s_wx = 0;
      s_wz = 0;
      s2_wx = 0;
      s2_wz = 0;

      s_wy = 0;
      s2_wy = 0;

      x = 0;
      y = 0;
      z = 0;
      ndof_x = 0;

      for (unsigned int i = 0; i < (unsigned int)points.size(); i++) {
        std::pair<GlobalPoint, double> p = nearestPoint(GlobalPoint(old_x, old_y, old_z), (iclus)[i].track());

        wx = points[i].second.x() <= precision ? std::pow(precision, 2) : std::pow(points[i].second.x(), 2);
        wy = points[i].second.y() <= precision ? std::pow(precision, 2) : std::pow(points[i].second.y(), 2);

        wz = points[i].second.z() <= precision ? std::pow(precision, 2) : std::pow(points[i].second.z(), 2);

        xpull = 0.;
        if (std::pow(p.first.x() - old_x, 2) / (wx + err_x) < muSquare &&
            std::pow(p.first.y() - old_y, 2) / (wy + err_y) < muSquare &&
            std::pow(p.first.z() - old_z, 2) / (wz + err_z) < muSquare)
          xpull = 1.;

        ndof_x += xpull;

        wx = xpull / wx;
        wy = xpull / wy;
        wz = xpull / wz;

        x += wx * p.first.x();
        y += wy * p.first.y();
        z += wz * p.first.z();

        s_wx += wx;
        s_wy += wy;
        s_wz += wz;

        s2_wx += wx * xpull;
        s2_wy += wy * xpull;
        s2_wz += wz * xpull;
      }

      if (s_wx == 0. || s_wy == 0. || s_wz == 0.) {
        edm::LogWarning("WeightedMeanFitter")
            << "Vertex fitting failed, either all tracks are outliers or they have a very large error\n";
        return TransientVertex(GlobalPoint(0, 0, 0), err, iclus, 0, 0);
      }
      wx = bse.cxx() <= std::pow(precision, 2) ? 1. / std::pow(precision, 2) : 1. / bse.cxx();
      wy = bse.cyy() <= std::pow(precision, 2) ? 1. / std::pow(precision, 2) : 1. / bse.cyy();

      x += bsp.x() * wx;
      y += bsp.y() * wy;
      s_wx += wx;
      s2_wx += wx;
      s_wy += wy;
      s2_wy += wy;

      x /= s_wx;
      y /= s_wy;
      z /= s_wz;

      err_x = (s2_wx / std::pow(s_wx, 2));
      err_y = (s2_wy / std::pow(s_wy, 2));
      err_z = (s2_wz / std::pow(s_wz, 2));

      if (std::abs(x - old_x) < (precision) && std::abs(y - old_y) < (precision) && std::abs(z - old_z) < (precision)) {
        break;
      }
    }
    err(0, 0) = err_x * corr_x_bs * corr_x_bs;
    err(1, 1) = err_y * corr_x_bs * corr_x_bs;
    err(2, 2) = err_z * corr_z * corr_z;

    float dist = 0;
    for (const auto& p : points) {
      wx = p.second.x();
      wx = wx <= precision ? precision : wx;

      wz = p.second.z();
      wz = wz <= precision ? precision : wz;

      dist = std::pow(p.first.x() - x, 2) / (std::pow(wx, 2) + err(0, 0));
      dist += std::pow(p.first.y() - y, 2) / (std::pow(wx, 2) + err(1, 1));
      dist += std::pow(p.first.z() - z, 2) / (std::pow(wz, 2) + err(2, 2));
      chi2 += dist;
    }
    TransientVertex v(GlobalPoint(x, y, z), err, iclus, chi2, (int)ndof_x);
    return v;
  }

  inline TransientVertex weightedMeanOutlierRejectionVarianceAsError(
      const std::vector<std::pair<GlobalPoint, GlobalPoint>>& points,
      std::vector<std::vector<reco::TransientTrack>>::const_iterator iclus) {
    float x = 0, y = 0, z = 0, s_wx = 0, s_wy = 0, s_wz = 0, wx = 0, wy = 0, wz = 0, chi2 = 0;
    float ndof_x = 0;
    AlgebraicSymMatrix33 err;
    err(2, 2) = startError * startError;  // error is 20 cm, so cov -> is 20 ^ 2
    err(0, 0) = err(1, 1) = err(2, 2) / 100.;

    for (const auto& p : points) {
      wx = p.second.x();
      wx = wx <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(wx, 2);

      wz = p.second.z();
      wz = wz <= precision ? 1. / std::pow(precision, 2) : 1. / std::pow(wz, 2);

      x += p.first.x() * wx;
      y += p.first.y() * wx;
      z += p.first.z() * wz;

      s_wx += wx;
      s_wz += wz;
    }

    if (s_wx == 0. || s_wz == 0.) {
      edm::LogWarning("WeightedMeanFitter") << "Vertex fitting failed at beginning\n";
      return TransientVertex(GlobalPoint(0, 0, 0), err, *iclus, 0, 0);
    }

    x /= s_wx;
    y /= s_wx;
    z /= s_wz;

    float old_x, old_y, old_z;
    float xpull;
    int niter = 0;
    float err_x, err_y, err_z;
    err_x = 1. / s_wx;
    err_y = 1. / s_wx;
    err_z = 1. / s_wz;
    float s_err_x = 0., s_err_y = 0., s_err_z = 0.;
    while ((niter++) < maxIterations) {
      old_x = x;
      old_y = y;
      old_z = z;
      s_wx = 0;
      s_wy = 0;
      s_wz = 0;
      x = 0;
      y = 0;
      z = 0;
      s_err_x = 0.;
      s_err_y = 0.;
      s_err_z = 0.;

      for (const auto& p : points) {
        wx = p.second.x();
        wx = wx <= precision ? precision : wx;

        wy = wx * wx + err_y;
        wx = wx * wx + err_x;

        wz = p.second.z();
        wz = wz <= precision ? precision : wz;
        wz = wz * wz + err_z;

        xpull = std::pow((p.first.x() - old_x), 2) / wx;
        xpull += std::pow((p.first.y() - old_y), 2) / wy;
        xpull += std::pow((p.first.z() - old_z), 2) / wz;
        xpull = 1. / (1. + std::exp(-0.5 * ((muSquare)-xpull)));
        ndof_x += xpull;

        wx = 1. / wx;
        wy = 1. / wy;
        wz = 1. / wz;

        wx *= xpull;
        wy *= xpull;
        wz *= xpull;

        x += wx * p.first.x();
        y += wy * p.first.y();
        z += wz * p.first.z();

        s_wx += wx;
        s_wy += wy;
        s_wz += wz;

        s_err_x += wx * pow(p.first.x() - old_x, 2);
        s_err_y += wy * pow(p.first.y() - old_y, 2);
        s_err_z += wz * pow(p.first.z() - old_z, 2);
      }
      if (s_wx == 0. || s_wy == 0. || s_wz == 0.) {
        edm::LogWarning("WeightedMeanFitter")
            << "Vertex fitting failed, either all tracks are outliers or they have a very large error\n";
        return TransientVertex(GlobalPoint(0, 0, 0), err, *iclus, 0, 0);
      }
      x /= s_wx;
      y /= s_wy;
      z /= s_wz;

      err_x = s_err_x / s_wx;
      err_y = s_err_y / s_wy;
      err_z = s_err_z / s_wz;

      if (std::abs(x - old_x) < (precision / 1.) && std::abs(y - old_y) < (precision / 1.) &&
          std::abs(z - old_z) < (precision / 1.))
        break;
    }

    err(0, 0) = err_x;
    err(1, 1) = err_y;
    err(2, 2) = err_z;

    float dist = 0.f;
    for (const auto& p : points) {
      wx = p.second.x();
      wx = wx <= precision ? precision : wx;

      wz = p.second.z();
      wz = wz <= precision ? precision : wz;

      dist = std::pow(p.first.x() - x, 2) / (std::pow(wx, 2) + std::pow(err(0, 0), 2));
      dist += std::pow(p.first.y() - y, 2) / (std::pow(wx, 2) + std::pow(err(1, 1), 2));
      dist += std::pow(p.first.z() - z, 2) / (std::pow(wz, 2) + std::pow(err(2, 2), 2));
      chi2 += dist;
    }
    TransientVertex v(GlobalPoint(x, y, z), err, *iclus, chi2, (int)ndof_x);
    return v;
  }

};  // namespace WeightedMeanFitter

#endif
