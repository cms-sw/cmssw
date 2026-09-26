// The fast BrokenLine fit's magnetic-field treatment (fitCorrections on), against tracks propagated by RK4 in
// the real CMS solenoid lattice (test/blBFieldMapFixture.h). B_r turns the dip angle along the path and Bz(r,z)
// falls by ~7.5 % towards the endcaps; the fit removes both as deterministic offsets. A track is propagated in
// the lattice, sampled on the layer radii and disc planes, and fitted. Asserted:
//   1. noiseless, no scattering: with the field rows on, z0 and cot(theta) reproduce the generated values far
//      better than the resolution; off, the fit carries the z-odd, charge-even bias of A/R1;
//   2. the fitted pt is the same with the origin field and with the hit-averaged effective field;
//   3. with hit noise and multiple scattering the z0 and cot(theta) pulls stay near one with no z-odd mean.
// The same assertions run on every backend; the noiseless values must agree between backends to the printed
// precision.

#include <algorithm>
#include <cmath>
#include <memory>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/test/blBFieldMapFixture.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr int kN = 10;    // hits of a pixel-seeded track (one fit template instance)
  constexpr int kNOt = 6;   // hits of the outer-tracker-only layout
  constexpr int kOut = 8;   // outputs per (track, variant)
  constexpr int kNVar = 5;  // 0 = field rows off, 1 = rows on, 2 = rows on with the origin scalar,
                            // 3 = rows on + the state moved to the fitted circle's perigee,
                            // 4 = rows on + the full inward transport through the map (production)
  constexpr double kBzOrigin = 0.29979246 * blBFieldMapFixture::kBzOriginTesla / 100.;  // GeV/cm

  // ---------------------------------------------------------------- the field and the material, on the host
  //!< B in GeV/cm at a point, from the lattice: solenoid symmetry, B = (Br*rhat, Bz). With
  //!< tanLambda*cosAlpha = 0 the bending value bBendAndBrAt returns is Bz itself.
  std::array<double, 3> bFieldAt(const float* map, const std::array<double, 3>& p) {
    const double r = std::hypot(p[0], p[1]);
    double brNorm = 0.;
    const double bz = blBFieldMap::bBendAndBrAt(map, r, p[2], 0., brNorm) * kBzOrigin;
    const double br = brNorm * kBzOrigin;
    if (r > 0.)
      return {br * p[0] / r, br * p[1] / r, bz};
    return {0., 0., bz};
  }

  // ------------------------------------------------------------------------------- the sampled surfaces
  // Phase-2 tracker layer radii and disc planes (nominal), with the acceptance that decides which of them a
  // given trajectory crosses.
  struct Barrel {
    double r, halfZ;
  };
  struct Disc {
    double z, rMin, rMax;
  };
  const std::vector<Barrel> kBarrels = {{3.0, 25.},
                                        {6.8, 25.},
                                        {10.2, 25.},
                                        {16.0, 25.},
                                        {23.0, 120.},
                                        {36.0, 120.},
                                        {51.0, 120.},
                                        {68.0, 120.},
                                        {86.0, 120.},
                                        {108.0, 120.}};
  const std::vector<Disc> kDiscs = {{33., 4.5, 25.},
                                    {40., 4.5, 25.},
                                    {48., 4.5, 25.},
                                    {58., 4.5, 25.},
                                    {70., 4.5, 25.},
                                    {85., 4.5, 25.},
                                    {103., 4.5, 25.},
                                    {126., 4.5, 25.},
                                    {131., 22., 112.},
                                    {155., 22., 112.},
                                    {178., 22., 112.},
                                    {201., 22., 112.},
                                    {225., 22., 112.},
                                    {250., 22., 112.}};

  //!< one generated track: the hits it leaves on the surfaces above, in path order.
  struct Track {
    double hits[3][kN];
    bool ok = false;
  };

  //!< Highland planar angle for a thickness x = X/X0 at momentum p (pion), as the fit models it.
  double theta0Of(double x, double p) {
    if (!(x > 0.))
      return 0.;
    constexpr double kMassPion = 0.13957;
    const double beta = p / std::sqrt(p * p + kMassPion * kMassPion);
    return 13.6e-3 / (beta * p) * std::sqrt(x) * (1. + 0.038 * std::log(x));
  }

  //!< RK4 through the lattice from the origin, optionally scattering in the material map; records the
  //!< crossings of the surfaces above until kN of them are collected.
  Track makeTrack(const float* map,
                  const blMaterialMap::Map* rho,
                  double pT,
                  double eta,
                  int q,
                  std::mt19937_64* rng,
                  int nWanted,
                  bool otOnly) {
    Track tk;
    const double tanl = std::sinh(eta);
    const double p = pT * std::cosh(eta);
    const double invn = 1. / std::sqrt(1. + tanl * tanl);
    std::array<double, 3> pos = {0., 0., 0.};
    std::array<double, 3> dir = {invn, 0., tanl * invn};  // phi0 = 0
    const double ds = 0.2;
    int nHit = 0;
    double xCluster = 0.;  // running X/X0 of the material cluster being crossed
    std::normal_distribution<double> gauss(0., 1.);
    for (int step = 0; step < 4000 && nHit < nWanted; ++step) {
      const auto prev = pos;
      auto deriv = [&](const std::array<double, 3>& x, const std::array<double, 3>& t) {
        const auto b = bFieldAt(map, x);
        return std::array<double, 3>{(q / p) * (t[1] * b[2] - t[2] * b[1]),
                                     (q / p) * (t[2] * b[0] - t[0] * b[2]),
                                     (q / p) * (t[0] * b[1] - t[1] * b[0])};
      };
      std::array<double, 3> k1p = dir, k1t = deriv(pos, dir), p2{}, t2{}, p3{}, t3{}, p4{}, t4{};
      for (int i = 0; i < 3; ++i) {
        p2[i] = pos[i] + 0.5 * ds * k1p[i];
        t2[i] = dir[i] + 0.5 * ds * k1t[i];
      }
      auto k2p = t2, k2t = deriv(p2, t2);
      for (int i = 0; i < 3; ++i) {
        p3[i] = pos[i] + 0.5 * ds * k2p[i];
        t3[i] = dir[i] + 0.5 * ds * k2t[i];
      }
      auto k3p = t3, k3t = deriv(p3, t3);
      for (int i = 0; i < 3; ++i) {
        p4[i] = pos[i] + ds * k3p[i];
        t4[i] = dir[i] + ds * k3t[i];
      }
      auto k4p = t4, k4t = deriv(p4, t4);
      for (int i = 0; i < 3; ++i) {
        pos[i] += ds / 6. * (k1p[i] + 2. * k2p[i] + 2. * k3p[i] + k4p[i]);
        dir[i] += ds / 6. * (k1t[i] + 2. * k2t[i] + 2. * k3t[i] + k4t[i]);
      }
      const double dn = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
      for (int i = 0; i < 3; ++i)
        dir[i] /= dn;

      // multiple scattering: one kink per contiguous material cluster, Highland at the cluster's thickness
      if (rng != nullptr) {
        const double rr = std::hypot(pos[0], pos[1]);
        const double density = blMaterialMap::rhoAt(*rho, float(rr), float(pos[2]));
        xCluster += density * ds;
        // one Highland kink per crossed layer (the map's air, 3.3e-5 per cm, is charged in lumps of the
        // same size so that its total is not lost): thickness collected, then spent when the dense
        // material ends or the lump is big enough.
        constexpr double kDense = 1.e-3, kLump = 2.e-3;
        if (xCluster > 0. && (density < kDense || xCluster > kLump)) {
          const double th0 = theta0Of(xCluster, p);
          xCluster = 0.;
          // two independent planar kinks in the plane transverse to the direction
          std::array<double, 3> u = {-dir[1], dir[0], 0.};
          const double un = std::hypot(u[0], u[1]);
          for (int i = 0; i < 3; ++i)
            u[i] /= un;
          const std::array<double, 3> v = {-dir[2] * u[1], dir[2] * u[0], dir[0] * u[1] - dir[1] * u[0]};
          const double a = th0 * gauss(*rng), b = th0 * gauss(*rng);
          double nn = 0.;
          for (int i = 0; i < 3; ++i) {
            dir[i] += a * u[i] + b * v[i];
            nn += dir[i] * dir[i];
          }
          nn = std::sqrt(nn);
          for (int i = 0; i < 3; ++i)
            dir[i] /= nn;
        }
      }

      // surface crossings of this step (linear interpolation inside the step)
      const double r0 = std::hypot(prev[0], prev[1]), r1 = std::hypot(pos[0], pos[1]);
      for (auto const& bl : kBarrels) {
        if (nHit >= nWanted)
          break;
        if (otOnly && bl.r < 20.)
          continue;
        if ((r0 - bl.r) * (r1 - bl.r) < 0.) {
          const double f = (bl.r - r0) / (r1 - r0);
          const double z = prev[2] + f * (pos[2] - prev[2]);
          if (std::abs(z) < bl.halfZ) {
            tk.hits[0][nHit] = prev[0] + f * (pos[0] - prev[0]);
            tk.hits[1][nHit] = prev[1] + f * (pos[1] - prev[1]);
            tk.hits[2][nHit] = z;
            ++nHit;
          }
        }
      }
      for (auto const& dc : kDiscs) {
        if (nHit >= nWanted)
          break;
        if (otOnly && dc.rMax < 30.)
          continue;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          const double zt = sgn * dc.z;
          if ((prev[2] - zt) * (pos[2] - zt) < 0.) {
            const double f = (zt - prev[2]) / (pos[2] - prev[2]);
            const double x = prev[0] + f * (pos[0] - prev[0]), y = prev[1] + f * (pos[1] - prev[1]);
            const double r = std::hypot(x, y);
            if (r > dc.rMin && r < dc.rMax) {
              tk.hits[0][nHit] = x;
              tk.hits[1][nHit] = y;
              tk.hits[2][nHit] = zt;
              ++nHit;
            }
          }
        }
      }
      if (r1 > 112. || std::abs(pos[2]) > 265.)
        break;
    }
    tk.ok = (nHit == nWanted);
    return tk;
  }

  // --------------------------------------------------------------------------------- the fit, on the device
  //!< per-track effective bending field, the same hit average blEffectiveBField (BrokenLineFitKernels.h)
  //!< forms for the production fit.
  template <int NH>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double effectiveBField(Acc1D const& acc,
                                                        Eigen::Matrix<double, 3, NH> const& hits,
                                                        Eigen::Vector4d const& ff,
                                                        double b,
                                                        const float* map) {
    const double absR = alpaka::math::abs(acc, ff(2));
    double sum = 0.;
    for (int i = 0; i < NH; ++i) {
      const double r = alpaka::math::sqrt(acc, hits(0, i) * hits(0, i) + hits(1, i) * hits(1, i));
      const double den = ff(3) * absR * r;
      const double tlca = (den != 0.) ? -(ff(0) * hits(1, i) - ff(1) * hits(0, i)) / den : 0.;
      sum += blBFieldMap::bBendAt(map, r, hits(2, i), tlca);
    }
    return b * sum / double(NH);
  }

  template <int NH>
  struct FitKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  const blMaterialMap::Map* rho,
                                  const float* map,
                                  int nTracks,
                                  double* scratch,
                                  double* out) const {
      for (auto j : cms::alpakatools::uniform_elements(acc, nTracks * kNVar)) {
        const int t = int(j) / kNVar, v = int(j) % kNVar;
        Eigen::Matrix<double, 3, NH> hits;
        Eigen::Matrix<float, 6, NH> hits_ge;
        for (int c = 0; c < NH; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * NH * t + 3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * NH * t + 6 * c + r];
        }
        Eigen::Vector4d ff;
        bld::fastFit(acc, hits, ff);
        const double bEff = (v == 2) ? kBzOrigin : effectiveBField<NH>(acc, hits, ff, kBzOrigin, map);
        const float* fitMap = (v == 0) ? nullptr : map;
        double* mine = scratch + std::size_t(j) * std::size_t(bld::kLegacyFitScratchDoubles<NH>);
        bld::PreparedBrokenLineDataMap<NH, 1> data(mine);
        bld::LegacyFitWorkspaceMap<NH, 1> fitWs(mine + bld::kPreparedDataDoubles<NH>);
        bld::karimaki_circle_fit circle;
        ::riemannFit::LineFit line;
        bld::prepareBrokenLineData(
            acc, hits, ff, bEff, rho, data, fitWs, /*fitCorrections=*/true, /*elossGaps=*/true, fitMap, kBzOrigin);
        bld::lineFit(acc, hits_ge, ff, bEff, data, line, fitWs, /*fitCorrections=*/true);
        bld::circleFit(acc,
                       hits,
                       hits_ge,
                       ff,
                       bEff,
                       data,
                       circle,
                       fitWs,
                       /*fitCorrections=*/true,
                       /*elossGaps=*/false,
                       fitMap,
                       kBzOrigin);
        if (v == 3)  // the single-pass reference's own bias: the state belongs at the FITTED circle's perigee
          bld::transportToFittedPca(acc, hits, data, bEff, circle, line);
        else if (v == 4)  // and the trajectory is not that circle either, over the lever inside hit 0
          bld::transportToFittedPca(acc, hits, data, bEff, circle, line, fitMap, kBzOrigin);
        double* o = out + std::size_t(j) * kOut;
        o[0] = line.par(0);  // cot(theta)
        o[1] = line.par(1);  // z0 (zip)
        o[2] = line.cov(0, 0);
        o[3] = line.cov(1, 1);
        o[4] = bEff / alpaka::math::abs(acc, circle.par(2));  // pt
        o[5] = circle.chi2 + line.chi2;
        o[6] = circle.par(1);  // d0
        o[7] = bEff;
      }
    }
  };

  struct Config {
    double pT, eta;
    int q;
  };

  //!< hit resolutions of the generated points [cm]
  constexpr double kSigXY = 20.e-4, kSigZ = 40.e-4;
  constexpr int kRep = 200;  // noisy replicas per pull configuration

  //!< Generates, fits and checks one hit layout: `otOnly` false = a pixel-seeded track sampled on every
  //!< layer it crosses, true = the outer-tracker-only layout (first hit at
  //!< r = 23 cm, the state extrapolated back over the whole pixel volume).
  template <int NH>
  void runLayout(Queue& queue, const float* map, const blMaterialMap::Map* rho, bool otOnly, const char* label) {
    std::vector<Config> cfgs;
    for (double pT : {1., 3., 10.})
      for (double aeta : {1.0, 1.5, 1.7, 2.3})
        for (int zs : {+1, -1})
          for (int q : {+1, -1})
            cfgs.push_back({pT, zs * aeta, q});
    std::vector<Track> tracks;
    std::vector<Config> kept;
    for (auto const& c : cfgs) {
      auto tk = makeTrack(map, rho, c.pT, c.eta, c.q, nullptr, NH, otOnly);
      REQUIRE(tk.ok);
      tracks.push_back(tk);
      kept.push_back(c);
    }
    const int nNoiseless = int(tracks.size());

    // noisy replicas (hit noise + simulated multiple scattering) for the pull check
    std::mt19937_64 rng(20260913);
    std::normal_distribution<double> gauss(0., 1.);
    std::vector<Config> pullCfg;
    for (double pT : {1., 3.})
      for (double aeta : {1.0, 1.5, 2.3})
        for (int zs : {+1, -1})
          pullCfg.push_back({pT, zs * aeta, +1});
    std::vector<int> pullIndex;
    for (auto const& c : pullCfg) {
      pullIndex.push_back(int(tracks.size()));
      for (int r = 0; r < kRep; ++r) {
        Track tk;
        for (int try_ = 0; try_ < 5 && !tk.ok; ++try_)  // scattering can cost a crossing
          tk = makeTrack(map, rho, c.pT, c.eta, c.q, &rng, NH, otOnly);
        for (int i = 0; i < NH; ++i) {
          tk.hits[0][i] += kSigXY * gauss(rng);
          tk.hits[1][i] += kSigXY * gauss(rng);
          tk.hits[2][i] += kSigZ * gauss(rng);
        }
        tracks.push_back(tk);
        kept.push_back(c);
      }
    }
    const int nTracks = int(tracks.size());

    std::vector<double> hitsHost(std::size_t(nTracks) * 3 * NH);
    std::vector<float> geHost(std::size_t(nTracks) * 6 * NH, 0.f);
    for (int t = 0; t < nTracks; ++t)
      for (int i = 0; i < NH; ++i) {
        for (int r = 0; r < 3; ++r)
          hitsHost[std::size_t(t) * 3 * NH + 3 * i + r] = tracks[t].hits[r][i];
        float* g = geHost.data() + std::size_t(t) * 6 * NH + 6 * i;
        g[0] = float(kSigXY * kSigXY);  // xx
        g[2] = float(kSigXY * kSigXY);  // yy
        g[5] = float(kSigZ * kSigZ);    // zz
      }

    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(hitsHost.size());
    std::copy(hitsHost.begin(), hitsHost.end(), hits_h.data());
    auto ge_h = cms::alpakatools::make_host_buffer<float[], Platform>(geHost.size());
    std::copy(geHost.begin(), geHost.end(), ge_h.data());
    auto rho_h = cms::alpakatools::make_host_buffer<blMaterialMap::Map, Platform>();
    *rho_h.data() = *rho;
    auto map_h = cms::alpakatools::make_host_buffer<float[], Platform>(blBFieldMap::kNValues);
    std::copy_n(map, blBFieldMap::kNValues, map_h.data());
    auto out_h = cms::alpakatools::make_host_buffer<double[], Platform>(std::size_t(nTracks) * kNVar * kOut);
    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, hitsHost.size());
    auto ge_d = cms::alpakatools::make_device_buffer<float[]>(queue, geHost.size());
    auto rho_d = cms::alpakatools::make_device_buffer<blMaterialMap::Map>(queue);
    auto map_d = cms::alpakatools::make_device_buffer<float[]>(queue, blBFieldMap::kNValues);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, std::size_t(nTracks) * kNVar * kOut);
    auto scr_d = cms::alpakatools::make_device_buffer<double[]>(
        queue, std::size_t(nTracks) * kNVar * std::size_t(bld::kLegacyFitScratchDoubles<NH>));
    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ge_d, ge_h);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::memcpy(queue, map_d, map_h);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>((nTracks * kNVar + 63) / 64, 64),
                        FitKernel<NH>{},
                        hits_d.data(),
                        ge_d.data(),
                        rho_d.data(),
                        map_d.data(),
                        nTracks,
                        scr_d.data(),
                        out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::wait(queue);
    const double* out = out_h.data();
    auto at = [&](int t, int v) { return out + (std::size_t(t) * kNVar + v) * kOut; };

    printf("\n== fast-BL field rows, %s, %d hits ==\n", label, NH);
    printf(
        "   pt  eta  q |  z0 off [um]  z0 rows [um]  z0 +pca [um]  z0 +transport [um] | d0 +pca [um]"
        "  d0 +transport [um] | cot off [1e-4] cot on [1e-4] | pt/true off     on  on(B0) | chi2 off    on\n");
    double maxZOn = 0., maxCotOn = 0., maxZOff = 0., maxCotOff = 0., maxPtOff = 0., maxPtOn = 0., maxChi2On = 0.;
    double maxZPca = 0., maxZFull = 0., maxD0Pca = 0., maxD0Full = 0.;
    for (int t = 0; t < nNoiseless; ++t) {
      const double cotTrue = std::sinh(kept[t].eta);
      const double z0Off = at(t, 0)[1] * 1e4, z0On = at(t, 1)[1] * 1e4;
      const double z0Pca = at(t, 3)[1] * 1e4, z0Full = at(t, 4)[1] * 1e4;
      const double d0Pca = at(t, 3)[6] * 1e4, d0Full = at(t, 4)[6] * 1e4;
      const double cotOff = (at(t, 0)[0] - cotTrue) * 1e4, cotOn = (at(t, 1)[0] - cotTrue) * 1e4;
      printf(
          "%5.1f %+4.1f %+d | %11.1f %13.1f %13.1f %19.1f | %12.1f %19.1f | %14.2f %13.2f | %8.4f %6.4f %7.4f |"
          " %8.2f %5.2f\n",
          kept[t].pT,
          kept[t].eta,
          kept[t].q,
          z0Off,
          z0On,
          z0Pca,
          z0Full,
          d0Pca,
          d0Full,
          cotOff,
          cotOn,
          at(t, 0)[4] / kept[t].pT,
          at(t, 1)[4] / kept[t].pT,
          at(t, 2)[4] / kept[t].pT,
          at(t, 0)[5],
          at(t, 1)[5]);
      maxZOn = std::max(maxZOn, std::abs(z0On));
      maxZPca = std::max(maxZPca, std::abs(z0Pca));
      maxZFull = std::max(maxZFull, std::abs(z0Full));
      maxD0Pca = std::max(maxD0Pca, std::abs(d0Pca));
      maxD0Full = std::max(maxD0Full, std::abs(d0Full));
      maxCotOn = std::max(maxCotOn, std::abs(cotOn) * 1e-4);
      maxZOff = std::max(maxZOff, std::abs(z0Off));
      maxCotOff = std::max(maxCotOff, std::abs(cotOff) * 1e-4);
      maxPtOff = std::max(maxPtOff, std::abs(at(t, 0)[4] / kept[t].pT - 1.));
      maxPtOn = std::max(maxPtOn, std::abs(at(t, 1)[4] / kept[t].pT - 1.));
      maxChi2On = std::max(maxChi2On, at(t, 1)[5]);
    }
    printf(
        "   worst of the %d: |z0| %.1f -> %.1f -> %.1f -> %.1f um (off -> rows -> +pca -> +transport), "
        "|d0| %.1f -> %.1f um, |dcot| %.2e -> %.2e, |dpt|/pt %.4f -> %.4f, chi2 on <= %.3f\n",
        nNoiseless,
        maxZOff,
        maxZOn,
        maxZPca,
        maxZFull,
        maxD0Pca,
        maxD0Full,
        maxCotOff,
        maxCotOn,
        maxPtOff,
        maxPtOn,
        maxChi2On);

    // 1. the field rows leave a noiseless track with no residual: the model is now the trajectory
    CHECK(maxChi2On < 0.02);
    CHECK(maxZOn <= maxZOff);
    CHECK(maxCotOn <= maxCotOff);
    // the single-pass reference's own bias, which the two field rows cannot reach: the state belongs at the
    // perigee of the trajectory, not at the pre-fit circle's. Reporting it at the FITTED circle's perigee
    // takes a third to a half of what the rows leave in z0, and continuing the transport inwards through the
    // map -- the arc the trajectory does NOT spend on that circle over the empty lever inside hit 0 -- takes
    // the rest of it, and the impact parameter with it. Neither piece touches the dip angle.
    CHECK(maxZPca < 0.75 * maxZOn);
    CHECK(maxZFull <= maxZPca);
    CHECK(maxZFull < std::max(2.0, 0.2 * maxZOn));
    CHECK(maxD0Full < std::max(2.0, 0.1 * maxD0Pca));
    for (int t = 0; t < nNoiseless; ++t) {
      CHECK(at(t, 3)[0] == at(t, 1)[0]);
      CHECK(at(t, 4)[0] == at(t, 1)[0]);
    }
    // 2. the momentum: the profile row, not the choice of the single scalar, carries the field's shape
    CHECK(maxPtOn < 0.0015);
    CHECK(maxPtOn < 0.5 * maxPtOff);
    for (int t = 0; t < nNoiseless; ++t)
      CHECK(std::abs(at(t, 1)[4] - at(t, 2)[4]) < 0.01 * kept[t].pT);
    // 3. what is left uncorrected is z-odd and charge-even (A/R1's signature)
    for (int t = 0; t < nNoiseless; ++t) {
      if (kept[t].pT != 1. || kept[t].eta < 0. || kept[t].q < 0)
        continue;
      int tMirror = -1, tCharge = -1;
      for (int u = 0; u < nNoiseless; ++u) {
        if (kept[u].pT == 1. && std::abs(kept[u].eta + kept[t].eta) < 1e-9 && kept[u].q == kept[t].q)
          tMirror = u;
        if (kept[u].pT == 1. && std::abs(kept[u].eta - kept[t].eta) < 1e-9 && kept[u].q == -kept[t].q)
          tCharge = u;
      }
      REQUIRE(tMirror >= 0);
      REQUIRE(tCharge >= 0);
      const double a = at(t, 0)[1], m = at(tMirror, 0)[1], c = at(tCharge, 0)[1];
      CHECK(a * m < 0.);                           // odd in z
      CHECK(std::abs(a - c) < 0.3 * std::abs(a));  // even in charge
    }

    // 4. pulls with hit noise and simulated multiple scattering
    printf("   pulls (%d replicas):  pt  eta |  z0 mean  z0 sigma | cot mean cot sigma\n", kRep);
    double oddZ = 0., oddC = 0.;
    for (std::size_t c = 0; c < pullCfg.size(); ++c) {
      double sz = 0., szz = 0., sc = 0., scc = 0.;
      int nn = 0;
      for (int r = 0; r < kRep; ++r) {
        const double* o = at(pullIndex[c] + r, 4);
        if (!(o[3] > 0.) || !(o[2] > 0.))
          continue;
        const double pz = o[1] / std::sqrt(o[3]);
        const double pc = (o[0] - std::sinh(pullCfg[c].eta)) / std::sqrt(o[2]);
        sz += pz;
        szz += pz * pz;
        sc += pc;
        scc += pc * pc;
        ++nn;
      }
      REQUIRE(nn > kRep / 2);
      const double mz = sz / nn, sgz = std::sqrt(szz / nn - mz * mz);
      const double mc = sc / nn, sgc = std::sqrt(scc / nn - mc * mc);
      printf("                    %5.1f %+4.1f | %8.3f %9.3f | %8.3f %9.3f\n",
             pullCfg[c].pT,
             pullCfg[c].eta,
             mz,
             sgz,
             mc,
             sgc);
      oddZ = std::max(oddZ, std::abs(mz));
      oddC = std::max(oddC, std::abs(mc));
      CHECK(std::abs(mz) < 0.3);
      CHECK(std::abs(mc) < 0.3);
      CHECK(sgz > 0.3);
      CHECK(sgz < 1.5);
      CHECK(sgc > 0.3);
      CHECK(sgc < 1.5);
    }
    printf("   worst pull mean: z0 %.3f, cot %.3f\n", oddZ, oddC);
  }

}  // namespace

TEST_CASE("fast BrokenLine field rows on the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  const float* map = blBFieldMapFixture::kCmsLattice;
  auto rhoTable = std::make_unique<blMaterialMap::Map>();
  // the fixture's physics is the D121 (T35) map, loaded from the shipped binary file
  blMaterialMap::readFile(
      edm::FileInPath("RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap_T35_BP2030v3_v1.bin").fullPath(),
      *rhoTable);
  const blMaterialMap::Map* rho = rhoTable.get();

  // diagnostic lattice: the same Bz profile with B_r removed, used to generate AND to fit, so the residual
  // biases it leaves are the ones the two rows cannot reach (they belong to the single-pass fit's own
  // reference: the arc abscissa of the line fit comes from the fast-fit circle, which a non-circular
  // trajectory displaces).
  std::vector<float> noBr(map, map + blBFieldMap::kNValues);
  std::fill(noBr.begin() + blBFieldMap::kNNodes, noBr.end(), 0.f);

  for (auto const& device : devices) {
    auto queue = Queue(device);
    printf("\n%s\n", alpaka::getName(device).c_str());
    runLayout<kN>(queue, map, rho, /*otOnly=*/false, "pixel-seeded");
    runLayout<kNOt>(queue, map, rho, /*otOnly=*/true, "outer-tracker only (the displaced layout)");
    runLayout<kNOt>(queue, noBr.data(), rho, /*otOnly=*/true, "outer-tracker only, B_r removed (diagnostic)");
  }
}
