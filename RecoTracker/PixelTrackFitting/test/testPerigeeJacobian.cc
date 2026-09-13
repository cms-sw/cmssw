// Finite-difference check of riemannFit::transformToPerigeePlane (FitUtils.h).
//
// The map takes the SoA perigee state (phi, tip, q/pT, cot theta, zip), all of it referred to the
// beam spot, to the local parameters (q/p, dx/dz, dy/dz, x, y) of the plane the converters build at
// the beam spot (PixelTrackProducerFromSoAAlpaka.cc, L2TauTagNNProducerAlpaka.cc):
//
//   Surface::RotationType rot(sp, -cp, 0,  0, 0, -1,  cp, sp, 0);   Plane plane(beamSpot, rot);
//
// i.e. local x = (sin phi0, -cos phi0, 0), local y = -z_global, local z = (cos phi0, sin phi0, 0),
// with phi0 the nominal azimuth of the track being converted.  The plane is FIXED while the state
// varies, so the Jacobian is the derivative of the parameters of the trajectory's crossing of that
// plane with respect to the SoA state.  Since the crossing moves by O(dphi), the curvature enters
// the crossing at O(dphi^2) and a straight line through the PCA gives the exact first derivative.
//
// The frame here is read off the Plane object built exactly as in the converters; the arithmetic is
// redone in double precision because Surface is float and the check is at the 1e-6 level.

#include "RecoTracker/PixelTrackFitting/interface/FitUtils.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using riemannFit::Matrix5d;
using riemannFit::Vector5d;

namespace {

  using Vec3 = std::array<double, 3>;

  double dot(Vec3 const& a, Vec3 const& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  struct Frame {
    Vec3 org, xh, yh, zh;
  };

  // The plane of the converters, in double precision, cross-checked against the Plane they build.
  Frame makeFrame(double phi0, GlobalPoint const& beamSpot, double& worstFrame) {
    float sp = std::sin(phi0);
    float cp = std::cos(phi0);
    Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);
    Plane plane(beamSpot, rot);
    const Frame f{{beamSpot.x(), beamSpot.y(), beamSpot.z()},
                  {std::sin(phi0), -std::cos(phi0), 0.},
                  {0., 0., -1.},
                  {std::cos(phi0), std::sin(phi0), 0.}};
    auto const& r = plane.rotation();
    auto const& p = plane.position();
    const double d[12] = {p.x() - f.org[0],
                          p.y() - f.org[1],
                          p.z() - f.org[2],
                          r.xx() - f.xh[0],
                          r.xy() - f.xh[1],
                          r.xz() - f.xh[2],
                          r.yx() - f.yh[0],
                          r.yy() - f.yh[1],
                          r.yz() - f.yh[2],
                          r.zx() - f.zh[0],
                          r.zy() - f.zh[1],
                          r.zz() - f.zh[2]};
    for (double v : d)
      worstFrame = std::max(worstFrame, std::abs(v));
    return f;
  }

  // Local parameters of the trajectory of the SoA state `ip` on the fixed plane `f`.
  Vector5d localOnPlane(Vector5d const& ip, Frame const& f) {
    const double phi = ip(0), tip = ip(1), qOverPt = ip(2), cot = ip(3), zip = ip(4);
    const double sinTheta = 1. / std::sqrt(1. + cot * cot);
    const double pt = 1. / std::abs(qOverPt);
    // The state as the converter hands it to the geometry: PCA and momentum, beam-spot relative.
    const Vec3 pos{f.org[0] + tip * std::sin(phi), f.org[1] - tip * std::cos(phi), f.org[2] + zip};
    const Vec3 mom{pt * std::cos(phi), pt * std::sin(phi), pt * cot};
    Vec3 d{pos[0] - f.org[0], pos[1] - f.org[1], pos[2] - f.org[2]};
    const double lz = dot(d, f.zh), pz = dot(mom, f.zh);
    const double s = -lz / pz;  // straight line to the plane; exact to first order (see above)
    Vec3 x{d[0] + s * mom[0], d[1] + s * mom[1], d[2] + s * mom[2]};
    Vector5d op;
    op << qOverPt * sinTheta, dot(mom, f.xh) / pz, dot(mom, f.yh) / pz, dot(x, f.xh), dot(x, f.yh);
    return op;
  }

  // The Jacobian the code under test applies, written out here to be checked entry by entry.
  Matrix5d codeJacobian(Vector5d const& ip) {
    const double sinTheta2 = 1. / (1. + ip(3) * ip(3));
    const double sinTheta = std::sqrt(sinTheta2);
    const double cosTheta = ip(3) * sinTheta;
    Matrix5d jMat = Matrix5d::Zero();
    jMat(0, 2) = sinTheta;
    jMat(0, 3) = -sinTheta2 * cosTheta * ip(2);
    jMat(1, 0) = -1.;
    jMat(2, 3) = -1.;
    jMat(3, 1) = 1.;
    jMat(4, 0) = ip(1) * ip(3);
    jMat(4, 4) = -1.;
    return jMat;
  }

  Matrix5d loadCov(Vector5d const& e) {
    Matrix5d cov = Matrix5d::Zero();
    for (int i = 0; i < 5; ++i)
      cov(i, i) = e(i) * e(i);
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < i; ++j) {
        const double v = 0.3 * std::sqrt(cov(i, i) * cov(j, j));
        cov(i, j) = cov(j, i) = (i + j) % 2 ? -0.4 * v : 0.1 * v;
      }
    return cov;
  }

}  // namespace

int main() {
  const GlobalPoint beamSpot(0.08f, -0.03f, 1.7f);
  const double tol = 1e-6;

  double worstJac = 0., worstVal = 0., worstCov = 0., worstFrame = 0.;
  int worstRow = -1, worstCol = -1;
  long nStates = 0;

  for (double phi : {-2.9, -1.2, 0., 0.7, 1.9, 3.0})
    for (double tip : {-5., -1.3, -0.05, 0., 0.05, 1.3, 5.})
      for (double pt : {0.5, 1., 5., 20.})
        for (int charge : {-1, 1})
          for (double cot : {-3., -1.4, -0.3, 0., 0.3, 1.4, 3.})
            for (double zip : {-12., 0., 12.}) {
              Vector5d ip;
              ip << phi, tip, charge / pt, cot, zip;
              ++nStates;

              const Frame f = makeFrame(phi, beamSpot, worstFrame);

              // 1. the values: what the code publishes must be the parameters on that plane.
              Vector5d op;
              Matrix5d ocov;
              const Matrix5d icov = loadCov((Vector5d() << 2e-3, 2e-2, 0.02 * std::abs(ip(2)), 5e-3, 5e-2).finished());
              riemannFit::transformToPerigeePlane(ip, icov, op, ocov);
              const Vector5d truth = localOnPlane(ip, f);
              for (int i = 0; i < 5; ++i)
                worstVal = std::max(worstVal, std::abs(op(i) - truth(i)));

              // 2. every entry of the Jacobian, by central finite differences of the same map.
              const Matrix5d jCode = codeJacobian(ip);
              Matrix5d jNum;
              for (int j = 0; j < 5; ++j) {
                const double h = 1e-6 * std::max(1., std::abs(ip(j)));
                Vector5d pp = ip, pm = ip;
                pp(j) += h;
                pm(j) -= h;
                jNum.col(j) = (localOnPlane(pp, f) - localOnPlane(pm, f)) / (2. * h);
              }
              for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j) {
                  const double d = std::abs(jNum(i, j) - jCode(i, j));
                  if (d > worstJac) {
                    worstJac = d;
                    worstRow = i;
                    worstCol = j;
                  }
                }

              // 3. the covariance the code returns must be that Jacobian's similarity transform.
              const Matrix5d expected = jCode * icov * jCode.transpose();
              for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                  worstCov = std::max(
                      worstCov,
                      std::abs(ocov(i, j) - expected(i, j)) / std::sqrt(expected(i, i) * expected(j, j) + 1e-300));
            }

  std::printf("testPerigeeJacobian: %ld states\n", nStates);
  std::printf("  max |frame - Plane built by the converters|= %.3e\n", worstFrame);
  std::printf("  max |op - parameters on the plane|        = %.3e\n", worstVal);
  std::printf("  max |jNum - jCode|                        = %.3e  (entry %d,%d)\n", worstJac, worstRow, worstCol);
  std::printf("  max |ocov - jCode icov jCode^T| (relative)= %.3e\n", worstCov);

  if (worstVal > tol || worstJac > tol || worstCov > 1e-12 || worstFrame > 1e-6) {
    std::printf("FAILED\n");
    return EXIT_FAILURE;
  }
  std::printf("OK\n");
  return EXIT_SUCCESS;
}
