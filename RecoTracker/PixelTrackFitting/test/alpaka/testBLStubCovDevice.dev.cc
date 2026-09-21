// The covariance the fast BrokenLine fit declares for an outer-tracker-only track (first hit at r ~ 23 cm, the
// state extrapolated back over an empty inner lever), against tracks generated in a constant field, scattered
// in the material map and read out with the outer tracker's own sensors (macro-pixel and strip, the along-strip
// coordinate uniform over the strip). Three variants are fitted from the same events:
//   0  Highland's logarithm at each node's own thickness, stub errors as published
//   1  the logarithm at the track's total thickness, stub errors as published
//   2  the same, with the empirical stub pull-width factors removed from the local-x variance
// The kinks are generated with Highland's logarithm at the whole thickness crossed, so a chain of thin
// scatterers adds up to one thick one. Reported per (layout, pt, eta): the spread of each fitted parameter
// over the replicas divided by its declared error (the pull sigma).

#include <algorithm>
#include <array>
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

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMapFile.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr int kNOt = 6;                              // hits of the outer-tracker-only layout
  constexpr int kNPix = 8;                             // hits of the pixel-seeded control layout
  constexpr int kOut = 10;                             // outputs per (track, variant)
  constexpr int kNVar = 3;                             // see the file head
  constexpr double kBz = 3.8112;                       // T
  constexpr double kBField = 0.29979246 * kBz / 100.;  // GeV/cm
  constexpr int kRep = 400;                            // replicas per configuration

  // ------------------------------------------------------------------------------- sensors and surfaces
  // Phase-2 outer tracker: macro-pixel (PS) sensors on TBPS L1-L3 and the inner TEDD rings, strip (2S)
  // sensors on TB2S L4-L6 and the outer rings; the pixel detector for the control layout. pitchX is the
  // pitch of the measured coordinate, lenY the length of the unmeasured one; both are read out with a
  // uniform position over their cell, so the variances are pitchX^2/12 and lenY^2/12.
  enum Sensor { kPixel, kPS, kSS };
  struct SensorGeom {
    double pitchX, lenY;
  };
  constexpr SensorGeom kGeom[3] = {{0.0100, 0.0150}, {0.0100, 0.1467}, {0.0090, 5.0250}};

  struct Barrel {
    double r, halfZ;
    Sensor sensor;
    bool tilted;  // TBPS: the modules beyond the flat central rings point at the interaction region
  };
  struct Disc {
    double z, rMin, rMax;
    Sensor sensor;  // kSS above rSwitch, kPS below (outer tracker); the pixel discs are all kPixel
    double rSwitch;
  };
  const std::vector<Barrel> kBarrels = {{3.0, 25., kPixel, false},
                                        {6.8, 25., kPixel, false},
                                        {10.2, 25., kPixel, false},
                                        {16.0, 25., kPixel, false},
                                        {23.0, 120., kPS, true},
                                        {36.0, 120., kPS, true},
                                        {51.0, 120., kPS, true},
                                        {68.0, 120., kSS, false},
                                        {86.0, 120., kSS, false},
                                        {108.0, 120., kSS, false}};
  const std::vector<Disc> kDiscs = {{33., 4.5, 25., kPixel, 1e9},
                                    {40., 4.5, 25., kPixel, 1e9},
                                    {48., 4.5, 25., kPixel, 1e9},
                                    {58., 4.5, 25., kPixel, 1e9},
                                    {70., 4.5, 25., kPixel, 1e9},
                                    {85., 4.5, 25., kPixel, 1e9},
                                    {103., 4.5, 25., kPixel, 1e9},
                                    {126., 4.5, 25., kPixel, 1e9},
                                    {131., 22., 112., kSS, 60.},
                                    {155., 22., 112., kSS, 60.},
                                    {178., 22., 112., kSS, 60.},
                                    {201., 22., 112., kSS, 60.},
                                    {225., 22., 112., kSS, 60.},
                                    {250., 22., 112., kSS, 60.}};

  //!< One recorded crossing: the true point, the module's measurement frame and its sensor.
  struct Hit {
    double p[3];
    double ex[3], ey[3];  // measured and unmeasured directions, global unit vectors
    Sensor sensor;
    bool isStub;
  };
  struct Track {
    Hit hit[kNPix];
    bool ok = false;
  };

  //!< Measurement frame of a module at (r, z, phi): the measured direction is always the azimuthal one;
  //!< the unmeasured one is the beam direction for a flat barrel module, the radial one for a disc, and
  //!< the direction perpendicular to the line from the interaction region for a pointing (tilted) module.
  void frameAt(double x, double y, double z, bool barrel, bool tilted, double* ex, double* ey) {
    const double r = std::hypot(x, y);
    const double cs = (r > 0.) ? x / r : 1., sn = (r > 0.) ? y / r : 0.;
    ex[0] = -sn;
    ex[1] = cs;
    ex[2] = 0.;
    if (!barrel) {  // disc: the strips/pixel columns run radially
      ey[0] = cs;
      ey[1] = sn;
      ey[2] = 0.;
      return;
    }
    if (!tilted || std::abs(z) < 15.) {  // flat barrel module: along the beam
      ey[0] = ey[1] = 0.;
      ey[2] = 1.;
      return;
    }
    const double n = std::hypot(r, z);  // pointing module: perpendicular to (r, z) in the r-z plane
    ey[0] = -z * cs / n;
    ey[1] = -z * sn / n;
    ey[2] = r / n;
  }

  //!< Highland planar angle for a thickness x = X/X0 at momentum p (pion), the form the fit models.
  double theta0Of(double x, double p) {
    if (!(x > 0.))
      return 0.;
    constexpr double kMassPion = 0.13957;
    const double beta = p / std::sqrt(p * p + kMassPion * kMassPion);
    return 13.6e-3 / (beta * p) * std::sqrt(x) * (1. + 0.038 * std::log(x));
  }

  //!< RK4 in a constant solenoid field from the origin, scattering in the material map when rng != nullptr;
  //!< records the crossings of the surfaces above, skipping the pixel detector for the outer-tracker layout.
  //!< `xLogTotal` > 0 evaluates Highland's logarithm at the whole thickness the particle crosses, as the
  //!< PDG formula prescribes, instead of at each lump separately; `xTot` returns that thickness.
  Track makeTrack(const blMaterialMap::Map* rho,
                  double pT,
                  double eta,
                  int q,
                  std::mt19937_64* rng,
                  int nWanted,
                  bool otOnly,
                  double xLogTotal = 0.,
                  double* xTot = nullptr) {
    Track tk;
    const double tanl = std::sinh(eta);
    const double p = pT * std::cosh(eta);
    const double invn = 1. / std::sqrt(1. + tanl * tanl);
    std::array<double, 3> pos = {0., 0., 0.};
    std::array<double, 3> dir = {invn, 0., tanl * invn};  // phi0 = 0
    const double ds = 0.2;
    int nHit = 0;
    double xCluster = 0., xSum = 0.;
    std::normal_distribution<double> gauss(0., 1.);
    for (int step = 0; step < 6000 && nHit < nWanted; ++step) {
      const auto prev = pos;
      auto deriv = [&](const std::array<double, 3>& t) {
        return std::array<double, 3>{(q / p) * t[1] * kBField, -(q / p) * t[0] * kBField, 0.};
      };
      std::array<double, 3> k1p = dir, k1t = deriv(dir), p2{}, t2{}, p3{}, t3{}, p4{}, t4{};
      for (int i = 0; i < 3; ++i) {
        p2[i] = pos[i] + 0.5 * ds * k1p[i];
        t2[i] = dir[i] + 0.5 * ds * k1t[i];
      }
      auto k2p = t2, k2t = deriv(t2);
      for (int i = 0; i < 3; ++i) {
        p3[i] = pos[i] + 0.5 * ds * k2p[i];
        t3[i] = dir[i] + 0.5 * ds * k2t[i];
      }
      auto k3p = t3, k3t = deriv(t3);
      for (int i = 0; i < 3; ++i) {
        p4[i] = pos[i] + ds * k3p[i];
        t4[i] = dir[i] + ds * k3t[i];
      }
      auto k4p = t4, k4t = deriv(t4);
      for (int i = 0; i < 3; ++i) {
        pos[i] += ds / 6. * (k1p[i] + 2. * k2p[i] + 2. * k3p[i] + k4p[i]);
        dir[i] += ds / 6. * (k1t[i] + 2. * k2t[i] + 2. * k3t[i] + k4t[i]);
      }
      double dn = 0.;
      for (int i = 0; i < 3; ++i)
        dn += dir[i] * dir[i];
      dn = std::sqrt(dn);
      for (int i = 0; i < 3; ++i)
        dir[i] /= dn;

      // one Highland kink per crossed material lump of the map (its air is charged in lumps of the same
      // size, so that none of its total is lost)
      {
        const double rr = std::hypot(pos[0], pos[1]);
        const double dens = blMaterialMap::rhoAt(*rho, float(rr), float(pos[2]));
        xCluster += dens * ds;
        xSum += dens * ds;
        constexpr double kDense = 1.e-3, kLump = 2.e-3;
        if (rng != nullptr && xCluster > 0. && (dens < kDense || xCluster > kLump)) {
          const double th0 = (xLogTotal > 0.) ? theta0Of(xCluster, p) * (1. + 0.038 * std::log(xLogTotal)) /
                                                    (1. + 0.038 * std::log(xCluster))
                                              : theta0Of(xCluster, p);
          xCluster = 0.;
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

      auto record = [&](double x, double y, double z, bool barrel, bool tilted, Sensor s) {
        Hit& h = tk.hit[nHit];
        h.p[0] = x;
        h.p[1] = y;
        h.p[2] = z;
        h.sensor = s;
        h.isStub = (s != kPixel);
        frameAt(x, y, z, barrel, tilted, h.ex, h.ey);
        ++nHit;
      };
      const double r0 = std::hypot(prev[0], prev[1]), r1 = std::hypot(pos[0], pos[1]);
      for (auto const& bl : kBarrels) {
        if (nHit >= nWanted)
          break;
        if (otOnly && bl.sensor == kPixel)
          continue;
        if ((r0 - bl.r) * (r1 - bl.r) < 0.) {
          const double f = (bl.r - r0) / (r1 - r0);
          const double z = prev[2] + f * (pos[2] - prev[2]);
          if (std::abs(z) < bl.halfZ)
            record(prev[0] + f * (pos[0] - prev[0]), prev[1] + f * (pos[1] - prev[1]), z, true, bl.tilted, bl.sensor);
        }
      }
      for (auto const& dc : kDiscs) {
        if (nHit >= nWanted)
          break;
        if (otOnly && dc.sensor == kPixel)
          continue;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          const double zt = sgn * dc.z;
          if ((prev[2] - zt) * (pos[2] - zt) < 0.) {
            const double f = (zt - prev[2]) / (pos[2] - prev[2]);
            const double x = prev[0] + f * (pos[0] - prev[0]), y = prev[1] + f * (pos[1] - prev[1]);
            const double r = std::hypot(x, y);
            if (r > dc.rMin && r < dc.rMax)
              record(x, y, zt, false, false, (r > dc.rSwitch) ? kSS : kPS);
          }
        }
      }
      if (r1 > 112. || std::abs(pos[2]) > 265.)
        break;
    }
    tk.ok = (nHit == nWanted);
    if (xTot != nullptr)
      *xTot = xSum;
    return tk;
  }

  // --------------------------------------------------------------------------------- the fit, on device
  template <int NH>
  struct FitKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  const blMaterialMap::Map* rho,
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
          // variants 0 and 1 read the shipped error block, 2 and 3 the one without the scale factors
          const float* g = geIn + std::size_t(2 * t + (v < 2 ? 0 : 1)) * 6 * NH + 6 * c;
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = g[r];
        }
        Eigen::Vector4d ff;
        bld::fastFit(acc, hits, ff);
        double* mine = scratch + std::size_t(j) * std::size_t(bld::kLegacyFitScratchDoubles<NH>);
        bld::PreparedBrokenLineDataMap<NH, 1> data(mine);
        bld::LegacyFitWorkspaceMap<NH, 1> fitWs(mine + bld::kPreparedDataDoubles<NH>);
        bld::karimaki_circle_fit circle;
        ::riemannFit::LineFit line;
        bld::prepareBrokenLineData(
            acc, hits, ff, kBField, rho, data, fitWs, /*fitCorrections=*/true, /*elossGaps=*/true);
        if (v == 0)
          data.xx0Total = 0.;  // back to Highland's logarithm at each node's own thickness
        bld::lineFit(acc, hits_ge, ff, kBField, data, line, fitWs, /*fitCorrections=*/true);
        bld::circleFit(
            acc, hits, hits_ge, ff, kBField, data, circle, fitWs, /*fitCorrections=*/true, /*elossGaps=*/true);
        double* o = out + std::size_t(j) * kOut;
        o[0] = circle.par(1);                                    // d0
        o[1] = line.par(1);                                      // z0
        o[2] = circle.par(0);                                    // phi
        o[3] = line.par(0);                                      // cot(theta)
        o[4] = kBField / alpaka::math::abs(acc, circle.par(2));  // pt
        o[5] = circle.cov(1, 1);
        o[6] = line.cov(1, 1);
        o[7] = circle.cov(0, 0);
        o[8] = line.cov(0, 0);
        o[9] = circle.chi2 + line.chi2;
      }
    }
  };

  struct Config {
    double pT, eta;
  };

  //!< spread of a fitted parameter over the replicas, divided by the error the fit declares for it
  struct Stat {
    double mean = 0., sigma = 0., decl = 0., ratio = 0.;
  };
  //!< Core width, the estimator the in-situ gate uses: the +-2 sigma truncated r.m.s., iterated and
  //!< rescaled by the truncation factor of a Gaussian (E[x^2 | |x| < 2 sigma] = 0.77374 sigma^2).
  double coreSigma(const std::vector<double>& v) {
    if (v.size() < 20)
      return 0.;
    double mu = 0.;
    for (double x : v)
      mu += x;
    mu /= double(v.size());
    double sg = 0.;
    for (double x : v)
      sg += (x - mu) * (x - mu);
    sg = std::sqrt(sg / double(v.size()));
    for (int it = 0; it < 6 && sg > 0.; ++it) {
      double s = 0., ss = 0.;
      int n = 0;
      for (double x : v)
        if (std::abs(x - mu) < 2. * sg) {
          s += x;
          ss += x * x;
          ++n;
        }
      if (n < 10)
        break;
      mu = s / n;
      sg = std::sqrt(std::max(0., ss / n - mu * mu) / 0.77374);
    }
    return sg;
  }

  Stat spreadOverDeclared(const double* out, const std::vector<int>& idx, int v, int par, int varCol) {
    std::vector<double> vals;
    double sd = 0.;
    for (int t : idx) {
      const double* o = out + (std::size_t(t) * kNVar + v) * kOut;
      if (!(o[varCol] > 0.))
        continue;
      vals.push_back(o[par]);
      sd += o[varCol];
    }
    Stat st;
    if (vals.size() < 20)
      return st;
    st.sigma = coreSigma(vals);
    st.decl = std::sqrt(sd / double(vals.size()));
    st.ratio = (st.decl > 0.) ? st.sigma / st.decl : 0.;
    return st;
  }

  //!< Generates, reads out and fits one hit layout; prints the pull-sigma table and returns the worst
  //!< outer-tracker reads for the assertions.
  template <int NH>
  void runLayout(Queue& queue,
                 const blMaterialMap::Map* rho,
                 bool otOnly,
                 bool gaussianReadout,
                 int fineY,  // 1 = the unmeasured coordinate of the disc hits is read out finely
                 const char* label,
                 double& worstMeasHonest,
                 double& worstMeasProd,
                 double& worstFullHonest) {
    std::vector<Config> cfgs;
    for (double pT : {1., 3., 10.})
      for (double aeta : {0.3, 0.8, 1.2, 1.7, 2.2})
        cfgs.push_back({pT, aeta});

    std::mt19937_64 rng(20260913);
    std::normal_distribution<double> gauss(0., 1.);
    std::uniform_real_distribution<double> flat(-0.5, 0.5);

    std::vector<Track> tracks;
    std::vector<double> xTotOf(cfgs.size(), 0.);
    std::vector<std::array<std::vector<int>, 2>> index(cfgs.size());  // [cfg][scatter on/off] -> replicas
    for (std::size_t c = 0; c < cfgs.size(); ++c) {
      makeTrack(rho, cfgs[c].pT, cfgs[c].eta, +1, nullptr, NH, otOnly, 0., &xTotOf[c]);
      for (int sc = 0; sc < 2; ++sc) {
        for (int r = 0; r < kRep; ++r) {
          Track tk;
          for (int try_ = 0; try_ < 8 && !tk.ok; ++try_)
            tk = makeTrack(rho, cfgs[c].pT, cfgs[c].eta, +1, (sc == 1) ? &rng : nullptr, NH, otOnly, xTotOf[c]);
          if (!tk.ok)
            continue;
          index[c][sc].push_back(int(tracks.size()));
          tracks.push_back(tk);
        }
      }
      REQUIRE(index[c][0].size() > kRep / 2);
      REQUIRE(index[c][1].size() > kRep / 2);
    }
    const int nTracks = int(tracks.size());

    // readout: a uniform position over the measured pitch and over the unmeasured cell length; the error
    // block the fit is handed is that cell's variance, with the production scale factors on the measured
    // coordinate of a stub in copy 0 and without them in copy 1
    std::vector<double> hitsHost(std::size_t(nTracks) * 3 * NH);
    std::vector<float> geHost(std::size_t(nTracks) * 2 * 6 * NH, 0.f);
    for (int t = 0; t < nTracks; ++t) {
      for (int i = 0; i < NH; ++i) {
        const Hit& h = tracks[t].hit[i];
        SensorGeom g = kGeom[h.sensor];
        if (fineY == 1 && h.isStub && std::abs(h.ey[2]) < 0.5)  // the disc rings, whose strips run radially
          g.lenY = g.pitchX;
        const double dx = gaussianReadout ? g.pitchX / std::sqrt(12.) * gauss(rng) : g.pitchX * flat(rng);
        const double dy = gaussianReadout ? g.lenY / std::sqrt(12.) * gauss(rng) : g.lenY * flat(rng);
        for (int r = 0; r < 3; ++r)
          hitsHost[std::size_t(t) * 3 * NH + 3 * i + r] = h.p[r] + dx * h.ex[r] + dy * h.ey[r];
        const double varX = g.pitchX * g.pitchX / 12., varY = g.lenY * g.lenY / 12.;
        // BrokenLineFitKernels: 2S 0.4624 (barrel) / 0.8464 (endcap), PS 0.64 / 0.9025, by |z| < 118 cm
        double f = 1.;
        if (h.isStub) {
          const bool barrelHit = std::abs(h.p[2]) < 118.;
          f = (h.sensor == kSS) ? (barrelHit ? 0.4624 : 0.8464) : (barrelHit ? 0.64 : 0.9025);
        }
        for (int copy = 0; copy < 2; ++copy) {
          const double vx = (copy == 0) ? varX * f : varX;
          float* ge = geHost.data() + std::size_t(2 * t + copy) * 6 * NH + 6 * i;
          const double c3[3][3] = {{h.ex[0], h.ex[1], h.ex[2]}, {h.ey[0], h.ey[1], h.ey[2]}, {0., 0., 0.}};
          auto comp = [&](int a, int b) { return vx * c3[0][a] * c3[0][b] + varY * c3[1][a] * c3[1][b]; };
          ge[0] = float(comp(0, 0));
          ge[1] = float(comp(0, 1));
          ge[2] = float(comp(1, 1));
          ge[3] = float(comp(0, 2));
          ge[4] = float(comp(1, 2));
          ge[5] = float(comp(2, 2));
        }
      }
    }

    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(hitsHost.size());
    std::copy(hitsHost.begin(), hitsHost.end(), hits_h.data());
    auto ge_h = cms::alpakatools::make_host_buffer<float[], Platform>(geHost.size());
    std::copy(geHost.begin(), geHost.end(), ge_h.data());
    auto rho_h = cms::alpakatools::make_host_buffer<blMaterialMap::Map, Platform>();
    *rho_h.data() = *rho;
    auto out_h = cms::alpakatools::make_host_buffer<double[], Platform>(std::size_t(nTracks) * kNVar * kOut);
    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, hitsHost.size());
    auto ge_d = cms::alpakatools::make_device_buffer<float[]>(queue, geHost.size());
    auto rho_d = cms::alpakatools::make_device_buffer<blMaterialMap::Map>(queue);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, std::size_t(nTracks) * kNVar * kOut);
    auto scr_d = cms::alpakatools::make_device_buffer<double[]>(
        queue, std::size_t(nTracks) * kNVar * std::size_t(bld::kLegacyFitScratchDoubles<NH>));
    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ge_d, ge_h);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>((nTracks * kNVar + 63) / 64, 64),
                        FitKernel<NH>{},
                        hits_d.data(),
                        ge_d.data(),
                        rho_d.data(),
                        nTracks,
                        scr_d.data(),
                        out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::wait(queue);
    const double* out = out_h.data();

    printf("\n== fast-BL declared covariance, %s, %d hits ==\n", label, NH);
    printf(
        "  pt  eta  X/X0 |        d0 core pull sigma        |        z0 core pull sigma        |  cot core pull\n"
        "               | log@node  log@total  no factors | log@node  log@total  no factors | log@node  @total\n");
    for (std::size_t c = 0; c < cfgs.size(); ++c) {
      const auto& sc = index[c][1];
      const Stat d0n = spreadOverDeclared(out, sc, 0, 0, 5), d0t = spreadOverDeclared(out, sc, 1, 0, 5);
      const Stat d0f = spreadOverDeclared(out, sc, 2, 0, 5);
      const Stat z0n = spreadOverDeclared(out, sc, 0, 1, 6), z0t = spreadOverDeclared(out, sc, 1, 1, 6);
      const Stat z0f = spreadOverDeclared(out, sc, 2, 1, 6);
      const Stat con = spreadOverDeclared(out, sc, 0, 3, 8), cot = spreadOverDeclared(out, sc, 1, 3, 8);
      printf("%4.0f %4.1f %6.3f | %8.3f %10.3f %11.3f | %8.3f %10.3f %11.3f | %8.3f %8.3f\n",
             cfgs[c].pT,
             cfgs[c].eta,
             xTotOf[c],
             d0n.ratio,
             d0t.ratio,
             d0f.ratio,
             z0n.ratio,
             z0t.ratio,
             z0f.ratio,
             con.ratio,
             cot.ratio);
      if (otOnly) {
        worstMeasHonest = std::max(worstMeasHonest, std::max(std::abs(d0f.ratio - 1.), std::abs(z0f.ratio - 1.)));
        worstMeasProd = std::max(worstMeasProd, std::max(std::abs(d0n.ratio - 1.), std::abs(z0n.ratio - 1.)));
        worstFullHonest = std::max(worstFullHonest, std::max(std::abs(d0t.ratio - 1.), std::abs(z0t.ratio - 1.)));
      }
    }
  }

}  // namespace

TEST_CASE("fast BrokenLine stub covariance on the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  auto rhoTable = std::make_unique<blMaterialMap::Map>();
  // the fixture's physics is the D121 (T35) map, loaded from the shipped binary file
  blMaterialMap::readFile(
      edm::FileInPath("RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap_T35_BP2030v3_v1.bin").fullPath(),
      *rhoTable);
  const blMaterialMap::Map* rho = rhoTable.get();

  for (auto const& device : devices) {
    auto queue = Queue(device);
    printf("\n%s\n", alpaka::getName(device).c_str());
    double mh = 0., mp = 0., fh = 0., dummy = 0.;
    runLayout<kNOt>(queue,
                    rho,
                    /*otOnly=*/true,
                    /*gaussianReadout=*/false,
                    /*fineY=*/0,
                    "outer-tracker only (the displaced layout)",
                    mh,
                    mp,
                    fh);
    double gh = 0., gp = 0., gf = 0.;
    runLayout<kNOt>(queue,
                    rho,
                    /*otOnly=*/true,
                    /*gaussianReadout=*/true,
                    /*fineY=*/0,
                    "outer-tracker only, Gaussian readout of the same variances (diagnostic)",
                    gh,
                    gp,
                    gf);
    double fd = 0., fp = 0., ff2 = 0.;
    runLayout<kNOt>(queue,
                    rho,
                    /*otOnly=*/true,
                    /*gaussianReadout=*/false,
                    /*fineY=*/1,
                    "outer-tracker only, discs with a fine radial coordinate (diagnostic)",
                    fd,
                    fp,
                    ff2);
    runLayout<kNPix>(queue,
                     rho,
                     /*otOnly=*/false,
                     /*gaussianReadout=*/false,
                     /*fineY=*/0,
                     "pixel-seeded (control)",
                     dummy,
                     dummy,
                     dummy);
    printf(
        "\n  worst outer-tracker |sigma - 1|: logarithm at each node %.3f, at the track total %.3f,"
        " and without the stub scale factors %.3f\n",
        mp,
        fh,
        mh);
    // Highland's logarithm at the track's total thickness is what makes the declared covariance of an
    // outer-tracker-only track match its spread; at each node's own thickness the kinks are under-charged.
    CHECK(fh < mp);
    CHECK(fh < 0.25);
  }
}
