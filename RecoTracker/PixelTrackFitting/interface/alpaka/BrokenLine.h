#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BrokenLine_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BrokenLine_h

#include <limits>

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BandedSolve.h"  // bandFactor / bandSolveInPlace
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"       // Geant4 rho(r,z) -> segment X/X0

//#define BL_DEEPDEBUG

// The factorized fast-BL circle/line fit solves its bordered-pentadiagonal normal matrix with the O(N) root-free
// LDLt band factor + scalar Schur border elimination (Blobel NIM A566 eq.8) and selected-element unit-column
// solves, rather than forming the dense (N+1)^2 / N^2 inverse.
//
// `fitCorrections` (a runtime bool threaded from the producer parameter useFitCorrections through HelixFit and
// Kernel_BLFit into prepareBrokenLineData / lineFit / circleFit) selects the fast-BL scattering/covariance
// model. With it on, the corrections apply as one package (not independently selectable):
//   - material from the Geant4 tracker map (segmentXX0 with trapezoid quadrature: exact segment length, each
//     layer counted once) with the ENDPOINT PARTITION in prepareBrokenLineData (each gap's material split
//     between its two END nodes so that the gap's total AND its first moment are both reproduced), the
//     rigid-node guard that removes the kink term of a node with no assigned material, and a beamline->first-hit
//     material integral for the innermost scattering term;
//   - Highland's theta0 with geometry factor 1.0, the pion 1/beta factor and no pt cap in multScatt;
//   - the circle reference frame built on the first hit at least kBaseMin away (mref) instead of hit 1;
//   - the covariance blend in circleFit (measurement covariance from the full-circle Fisher inverse in the
//     Karimaki-consistent basis, grafted with the broken line's full multiple-scattering part), written as a
//     full 3x3 so the emitted matrix is one model throughout;
//   - the IONIZATION ENERGY LOSS of the material the map just measured, as a deterministic offset of the
//     circle's residuals (the elossCurvPerXX0 argument of circleFit, built by the caller from the SAME Landau
//     law the GBL refit ladder uses -- generalBrokenLine::elossTypicalColumn, one evaluation per track).
//     The fit models ONE constant curvature; the real one grows along the path as the track loses momentum,
//     so the fitted curvature comes out as a path average and the published pT is systematically LOW. The
//     offset is the double integral of that known curvature growth, u_eloss(s) = -int int dkappa, subtracted
//     from the measured radial residuals so the fitted Delta-kappa is the curvature AT THE REFERENCE POINT,
//     i.e. the production pT. It is a VALUE correction only: the covariance is untouched, exactly as in the
//     GBL's Deloss accumulator. It is charge-even in every ingredient (unsigned curvature, unsigned column,
//     the charge-normalised sTransverse), so the two charges stay exact mirrors with no mirroring code.
// With it off every one of these is bypassed and the fit is upstream's exactly: flat material |Delta s|*0.06/16
// per gap charged at its arrival node (the first gap for the innermost term), theta0 with geometry factor 0.7,
// beta = 1 and the 20 GeV pt cap, hit 1 as the reference, and the broken-line covariance emitted unblended.

namespace ALPAKA_ACCELERATOR_NAMESPACE::brokenline {

  using namespace cms::alpakatools;
  using namespace ::riemannFit;

  //!< Karimäki's parameters: (phi, d, k=1/R)
  /*!< covariance matrix: \n
    |cov(phi,phi)|cov( d ,phi)|cov( k ,phi)| \n
    |cov(phi, d )|cov( d , d )|cov( k , d )| \n
    |cov(phi, k )|cov( d , k )|cov( k , k )| \n
    as defined in Karimäki V., 1990, Effective circle fitting for particle trajectories, 
    Nucl. Instr. and Meth. A305 (1991) 187.
  */
  using karimaki_circle_fit = riemannFit::CircleFit;

  // Per-lane scratch layout for the fast-BL fit (the CA main fit). The fit's O(N) state -- prepared-data
  // vectors, the shared band block and the solve/workspace vectors -- lives in a per-lane slice of the
  // fit-solver scratch buffer instead of on the Kernel_BLFit stack frame (which the driver reserves as
  // frame x max-resident-threads of device memory). It is addressed in an [element][lane] layout (element e
  // of a lane at base[e*S], S = the launch's concurrent-fit count), so a warp reads consecutive lanes at
  // consecutive addresses; the raw band arrays are indexed with that stride and the helper vectors are
  // strided Eigen maps. The fit functions are storage-generic (the host implementation in
  // interface/BrokenLine.h keeps owned, stride-1 structs). Scalars and O(1) objects stay on the stack.

  //!< doubles per lane for the prepared-data map (radii 2n | sTransverse n | sTotal n | zInSZplane n |
  //!< varBeta n | matXX0 n)
  template <int n>
  constexpr int kPreparedDataDoubles = 7 * n;

  //!< Legacy-fit workspace = a shared bordered-pentadiagonal band block + the O(N) helper vectors.
  //!<   band block (6n+1): Mb 3n (lower-packed pentadiagonal, shared by line and circle) | cbor n (circle
  //!<                      border column) | corner 1 | w n (Schur vector) | ms n (solve scratch)
  //!<   helpers (10n+2):   pointsSZ 2n | zVec n | lineWeights n | lineRu n | lineU n | circleWeights n |
  //!<                      circleRu n+1 | circleU n+1 | varBetaOff n
  //!< The zVec slot carries TWO disjoint lifetimes and is mapped twice (zVec / gapXX0, same storage, no extra
  //!< doubles): zVec is written and read inside prepareBrokenLineData only (hit z -> pointsSZ, dead by the
  //!< time sTotal/zInSZplane are extracted); gapXX0 is written at the END of prepareBrokenLineData and read
  //!< in circleFit. lineFit touches neither, so the second lifetime spans prepare -> line -> circle intact.
  template <int n>
  constexpr int kLegacyBandBlockDoubles = 6 * n + 1;
  template <int n>
  constexpr int kLegacyHelperDoubles = 10 * n + 2;
  template <int n>
  constexpr int kLegacyWorkspaceDoubles = kLegacyBandBlockDoubles<n> + kLegacyHelperDoubles<n>;

  //!< total legacy-fit doubles per lane: [ prepared-data | workspace ].
  //!< The launch-side scratch quota (BrokenLineFit.dev.cc) must be >= this at maxN.
  template <int n>
  constexpr int kLegacyFitScratchDoubles = kPreparedDataDoubles<n> + kLegacyWorkspaceDoubles<n>;

  // [element][lane] scratch maps: element e of a lane is at base[e*S] (S = the launch's concurrent-fit
  // count, so a warp reads consecutive lanes at consecutive addresses). S is compile-time, so the Eigen
  // stride is baked into the type (as for the input-buffer Map3xNdS); S = 1 degenerates to per-lane
  // contiguous storage.
  template <int n, uint32_t S>
  using LegacyMapVecNd = Eigen::Map<riemannFit::VectorNd<n>, Eigen::Unaligned, Eigen::InnerStride<S>>;
  template <int n, uint32_t S>
  using LegacyMapVecNplus1d = Eigen::Map<riemannFit::VectorNplusONEd<n>, Eigen::Unaligned, Eigen::InnerStride<S>>;
  template <int n, uint32_t S>
  using LegacyMap2xNd = Eigen::Map<riemannFit::Matrix2xNd<n>, Eigen::Unaligned, Eigen::Stride<2 * S, S>>;

  //!< Map-backed prepared data over a lane's [element][lane] scratch block (stride S).
  template <int n, uint32_t S>
  struct PreparedBrokenLineDataMap {
    static constexpr int kN = n;
    int qCharge;
    LegacyMap2xNd<n, S> radii;
    LegacyMapVecNd<n, S> sTransverse;
    LegacyMapVecNd<n, S> sTotal;
    LegacyMapVecNd<n, S> zInSZplane;
    LegacyMapVecNd<n, S> varBeta;
    LegacyMapVecNd<n, S> matXX0;
    double innerXX0;
    ALPAKA_FN_ACC explicit PreparedBrokenLineDataMap(double* p)
        : radii(p),
          sTransverse(p + std::size_t(2 * n) * S),
          sTotal(p + std::size_t(3 * n) * S),
          zInSZplane(p + std::size_t(4 * n) * S),
          varBeta(p + std::size_t(5 * n) * S),
          matXX0(p + std::size_t(6 * n) * S) {}
  };

  //!< Per-lane fit workspace, map-backed over a lane's [element][lane] scratch
  //!< block (stride S). Band arrays are raw double* (indexed with laneStride in the raw-scalar band
  //!< solves); helpers are strided Eigen maps. Offsets follow the kLegacyBandBlockDoubles +
  //!< kLegacyHelperDoubles layout above.
  template <int n, uint32_t S>
  struct LegacyFitWorkspaceMap {
    static constexpr int kN = n;
    static constexpr int laneStride = int(S);
    double* bandMb;
    double* bandCbor;
    double* bandCorner;
    double* bandW;
    double* bandMs;
    LegacyMap2xNd<n, S> pointsSZ;
    LegacyMapVecNd<n, S> zVec;
    // Second mapping of zVec's n slots (see the layout note above): the running per-node material column
    // [X/X0] from hit 0, written by prepareBrokenLineData and overwritten in place by circleFit with the
    // dE/dx residual offset.
    LegacyMapVecNd<n, S> gapXX0;
    LegacyMapVecNd<n, S> lineWeights;
    LegacyMapVecNd<n, S> lineRu;
    LegacyMapVecNd<n, S> lineU;
    LegacyMapVecNd<n, S> circleWeights;
    LegacyMapVecNplus1d<n, S> circleRu;
    LegacyMapVecNplus1d<n, S> circleU;
    LegacyMapVecNd<n, S> varBetaOff;
    ALPAKA_FN_ACC explicit LegacyFitWorkspaceMap(double* p)
        : bandMb(p),
          bandCbor(p + std::size_t(3 * n) * S),
          bandCorner(p + std::size_t(4 * n) * S),
          bandW(p + std::size_t(4 * n + 1) * S),
          bandMs(p + std::size_t(5 * n + 1) * S),
          pointsSZ(p + std::size_t(6 * n + 1) * S),
          zVec(p + std::size_t(8 * n + 1) * S),
          gapXX0(p + std::size_t(8 * n + 1) * S),  // deliberately zVec's storage; disjoint lifetimes
          lineWeights(p + std::size_t(9 * n + 1) * S),
          lineRu(p + std::size_t(10 * n + 1) * S),
          lineU(p + std::size_t(11 * n + 1) * S),
          circleWeights(p + std::size_t(12 * n + 1) * S),
          circleRu(p + std::size_t(13 * n + 1) * S),
          circleU(p + std::size_t(14 * n + 2) * S),
          varBetaOff(p + std::size_t(15 * n + 3) * S) {}
  };

  //!< legacy-fit scratch is 233 doubles/lane at n=10 (7n prepared + 6n+1 band block + 10n+2 helpers).
  static_assert(kLegacyFitScratchDoubles<10> == 233, "legacy-fit scratch quota != 233 at n=10");

  /*!
    \brief Computes the Coulomb multiple scattering variance of the planar angle.
    
    \param length length of the track in the material.
    \param bField magnetic field in Gev/cm/c.
    \param radius radius of curvature (needed to evaluate p).
    \param layer denotes which of the four layers of the detector is the endpoint of the 
   *             multiple scattered track. For example, if Layer=3, then the particle has 
   *             just gone through the material between the second and the third layer.
    
    \todo add another Layer variable to identify also the start point of the track, 
   *      so if there are missing hits or multiple hits, the part of the detector that 
   *      the particle has traversed can be exactly identified.
    
    \warning the formula used here assumes beta=1, and so neglects the dependence 
   *         of theta_0 on the mass of the particle at fixed momentum.
    
    \return the variance of the planar angle ((theta_0)^2 /3).
  */
  // X/X0 integrated along the straight segment (r0,z0)->(r1,z1) through the Geant4 density map.
  // r,z from the actual hit positions [cm]; the march samples at ~0.5 cm, one sample per cell on the
  // reader's 0.5 cm radial lattice.
  //
  // `trapezoid` selects the quadrature weights over the SAME samples at the SAME positions.
  //   false: every one of the nseg samples carries the full weight dl = L/(nseg-1), so the rule integrates
  //     a length L*nseg/(nseg-1) (2L at the nseg floor of 2) and gives the two endpoint samples -- which sit
  //     ON the detector layers -- full weight in BOTH adjacent segments, i.e. every layer is counted twice.
  //   true: trapezoid weights (half-weight endpoints), which integrate L exactly and count each layer's
  //     bin once across the segment sum. Free: same sample count, same rhoAt lookups.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double segmentXX0(
      const TAcc& acc, const float* rho, double r0, double z0, double r1, double z1, bool trapezoid = false) {
    const double L = alpaka::math::sqrt(acc, (r1 - r0) * (r1 - r0) + (z1 - z0) * (z1 - z0));
    int nseg = int(2. * L);
    if (nseg < 2)
      nseg = 2;
    const double dl = L / (nseg - 1);
    double xx0 = 0.;
    for (int k = 0; k < nseg; ++k) {
      const double f = double(k) / (nseg - 1);
      // the endpoint half-weight is applied to dl itself; every other sample carries dl unscaled.
      const double w = (trapezoid && (k == 0 || k == nseg - 1)) ? 0.5 * dl : dl;
      xx0 += blMaterialMap::rhoAt(rho, float(r0 + f * (r1 - r0)), float(z0 + f * (z1 - z0))) * w;
    }
    return xx0;  // dimensionless X/X0
  }

  // ENDPOINT PARTITION of a segment's material between its two EXISTING end nodes -- the zero-extra-node
  // two-moment model. Same march, same sample positions and the same rhoAt lookups as segmentXX0: the ONLY
  // added work is one multiply-add per sample and one divide per segment.
  //
  // With q(l) the X/X0 density along (r0,z0)->(r1,z1) and d(l) = distance from the sample to the ARRIVAL end
  // (r1,z1):   W = int q dl,  S1 = int q d dl.  Charging  fDep*W  at the DEPARTURE end and  (1-fDep)*W  at the
  // arrival end, with
  //     fDep = S1 / (W * L) = <d> / L   in [0,1],
  // reproduces the segment's total material AND its first moment about either end EXACTLY (the lever of the
  // departure share about the arrival end is fDep*W*L = S1). It is exact to the second moment too whenever the
  // material sits at the two ends (modules with services between, the worst case for a single kink), and
  // otherwise reproduces the far-end offset variance as S1*L against the true S2, where a single kink at the
  // arrival node produces it as zero.
  //
  // Returns W, the same value segmentXX0 returns for the same `trapezoid` (same samples, same weights, same
  // accumulation order), so the partition never changes a segment's material total -- only where it is charged.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double segmentXX0Endpoint(const TAcc& acc,
                                                           const float* rho,
                                                           double r0,
                                                           double z0,
                                                           double r1,
                                                           double z1,
                                                           double& fDep,
                                                           bool trapezoid = false) {
    const double L = alpaka::math::sqrt(acc, (r1 - r0) * (r1 - r0) + (z1 - z0) * (z1 - z0));
    int nseg = int(2. * L);
    if (nseg < 2)
      nseg = 2;
    const double dl = L / (nseg - 1);
    double W = 0., S1overL = 0.;
    for (int k = 0; k < nseg; ++k) {
      const double f = double(k) / (nseg - 1);
      // the endpoint half-weight is applied to dl itself; every other sample carries dl unscaled.
      const double w = (trapezoid && (k == 0 || k == nseg - 1)) ? 0.5 * dl : dl;
      const double q = blMaterialMap::rhoAt(rho, float(r0 + f * (r1 - r0)), float(z0 + f * (z1 - z0))) * w;
      W += q;
      S1overL += q * (1. - f);  // (1-f) == d/L, so this accumulates S1/L directly (no division by L, L>0 free)
    }
    fDep = (W > 0.) ? S1overL / W : 0.;  // in [0,1] by construction: every sample contributes (1-f) in [0,1]
    return W;                            // total X/X0, the same value as segmentXX0(..., trapezoid)
  }

  // Coulomb MS planar-angle variance from a segment's X/X0 (radLen), fed in directly from the
  // material map (no length*inv_X0). Angle handling unchanged: the 1/(1+slope^2) here and the
  // (1+slope^2) projection applied later in circleFit; radLen already carries the path angle.
  //
  // `pionBeta` selects the momentum factor. false: upstream's multScatt exactly -- geometry factor 0.7,
  // beta == 1 (so theta0 does not depend on the mass, see the \warning at the declaration above) and pt
  // capped at 20 GeV by min(20., bField*radius). true: Highland's theta0 = 13.6 MeV/(beta c p) with geometry
  // factor 1.0, the pion 1/beta and no cap. p^2 beta^2 = p^4/(p^2 + m_pi^2).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double multScatt(const TAcc& acc,
                                                  const double radLen,
                                                  const double bField,
                                                  const double radius,
                                                  double slope,
                                                  bool pionBeta = false) {
    if (radLen <= 0.)
      return 0.;
    if (pionBeta) {
      // Corrections package: full Highland theta0^2 (geometry factor 1.0), pion 1/beta, no pt cap.
      constexpr double fact = riemannFit::sqr(13.6 / 1000.);
      constexpr double kPionMass = 0.13957;  // GeV (PDG)
      const double p2 = riemannFit::sqr(bField * radius) * (1. + riemannFit::sqr(slope));
      const double p2b2 = p2 * p2 / (p2 + riemannFit::sqr(kPionMass));
      return fact / p2b2 * radLen * riemannFit::sqr(1. + 0.038 * log(radLen));
    }
    // Corrections off: upstream's multScatt exactly (geometry factor 0.7, pt capped at 20 GeV). Any deviation
    // here shifts the emitted covariances and moves the fixed high-purity selector working point.
    auto pt2 = alpaka::math::min(acc, 20., bField * radius);
    pt2 *= pt2;
    constexpr double fact = 0.7 * riemannFit::sqr(13.6 / 1000.);
    return fact / (pt2 * (1. + riemannFit::sqr(slope))) * radLen * riemannFit::sqr(1. + 0.038 * log(radLen));
  }

  /*!
    \brief Computes the 2D rotation matrix that transforms the line y=slope*x into the line y=0.
    
    \param slope tangent of the angle of rotation.
    
    \return 2D rotation matrix.
  */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE riemannFit::Matrix2d rotationMatrix(const TAcc& acc, double slope) {
    riemannFit::Matrix2d rot;
    rot(0, 0) = 1. / alpaka::math::sqrt(acc, 1. + riemannFit::sqr(slope));
    rot(0, 1) = slope * rot(0, 0);
    rot(1, 0) = -rot(0, 1);
    rot(1, 1) = rot(0, 0);
    return rot;
  }

  /*!
    \brief Changes the Karimäki parameters (and consequently their covariance matrix) under a 
   *       translation of the coordinate system, such that the old origin has coordinates (x0,y0) 
   *       in the new coordinate system. The formulas are taken from Karimäki V., 1990, Effective 
   *       circle fitting for particle trajectories, Nucl. Instr. and Meth. A305 (1991) 187.
    
    \param circle circle fit in the old coordinate system. circle.par(0) is phi, circle.par(1) is d and circle.par(2) is rho. 
    \param x0 x coordinate of the translation vector.
    \param y0 y coordinate of the translation vector.
    \param jacobian passed by reference in order to save stack.
  */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void translateKarimaki(
      const TAcc& acc, karimaki_circle_fit& circle, double x0, double y0, riemannFit::Matrix3d& jacobian) {
    // Avoid multiple access to the circle.par vector.
    using scalar = typename std::remove_reference<decltype(circle.par(0))>::type;
    scalar phi = circle.par(0);
    scalar dee = circle.par(1);
    scalar rho = circle.par(2);

    // Avoid repeated trig. computations
    scalar sinPhi = alpaka::math::sin(acc, phi);
    scalar cosPhi = alpaka::math::cos(acc, phi);

    // Intermediate computations for the circle parameters
    scalar deltaPara = x0 * cosPhi + y0 * sinPhi;
    scalar deltaOrth = x0 * sinPhi - y0 * cosPhi + dee;
    scalar tempSmallU = 1 + rho * dee;
    scalar tempC = -rho * y0 + tempSmallU * cosPhi;
    scalar tempB = rho * x0 + tempSmallU * sinPhi;
    scalar tempA = 2. * deltaOrth + rho * (riemannFit::sqr(deltaOrth) + riemannFit::sqr(deltaPara));
    scalar tempU = alpaka::math::sqrt(acc, 1. + rho * tempA);

    // Intermediate computations for the error matrix transform
    scalar xi = 1. / (riemannFit::sqr(tempB) + riemannFit::sqr(tempC));
    scalar tempV = 1. + rho * deltaOrth;
    scalar lambda = (0.5 * tempA) / (riemannFit::sqr(1. + tempU) * tempU);
    scalar mu = 1. / (tempU * (1. + tempU)) + rho * lambda;
    scalar zeta = riemannFit::sqr(deltaOrth) + riemannFit::sqr(deltaPara);
    jacobian << xi * tempSmallU * tempV, -xi * riemannFit::sqr(rho) * deltaOrth, xi * deltaPara,
        2. * mu * tempSmallU * deltaPara, 2. * mu * tempV, mu * zeta - lambda * tempA, 0, 0, 1.;

    // translated circle parameters
    // phi
    circle.par(0) = alpaka::math::atan2(acc, tempB, tempC);
    // d
    circle.par(1) = tempA / (1 + tempU);
    // rho after translation. It is invariant, so noop
    // circle.par(2)= rho;

    // translated error matrix
    circle.cov = jacobian * circle.cov * jacobian.transpose();
  }

  /*!
    \brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.
    
    \param hits hits coordinates.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param results PreparedBrokenLineDataMap to be filled (see its description above).
    \param fitWs caller-provided workspace holding the O(N) transients (pointsSZ, zVec): a scratch-backed
           LegacyFitWorkspaceMap in the kernel. The fill expressions do not depend on the storage backing.
    \param fitCorrections selects the scattering/material model (see the file head): on, each gap's material
           comes from the tracker material map, split between its two end nodes, and a node with no material
           carries no kink; off, upstream's flat |Delta s|*0.06/16 per gap charged in full at its arrival node.
    \param elossGaps additionally records into fitWs.gapXX0(g) the running plain material column from hit 0
           through node g+1 [X/X0] (the gap totals accumulated, not the endpoint-partitioned matXX0), which
           circleFit's dE/dx offset (elossCurvPerXX0) needs. Costs one add and one store per gap inside the
           material march, only under fitCorrections. Default false: a caller that wants only the
           extrapolation covariance (ExtPredCoeff) does not pay for it.
  */
  template <alpaka::concepts::Acc TAcc, typename M3xN, typename V4, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void __attribute__((always_inline)) prepareBrokenLineData(const TAcc& acc,
                                                                                           const M3xN& hits,
                                                                                           const V4& fast_fit,
                                                                                           const double bField,
                                                                                           const float* rho,
                                                                                           TData& results,
                                                                                           TWs& fitWs,
                                                                                           bool fitCorrections = false,
                                                                                           bool elossGaps = false) {
    riemannFit::Vector2d dVec;
    riemannFit::Vector2d eVec;

    int mId = 1;

    if constexpr (n > 3) {
      riemannFit::Vector2d middle = 0.5 * (hits.block(0, n - 1, 2, 1) + hits.block(0, 0, 2, 1));
      auto d1 = (hits.block(0, n / 2, 2, 1) - middle).squaredNorm();
      auto d2 = (hits.block(0, n / 2 - 1, 2, 1) - middle).squaredNorm();
      mId = d1 < d2 ? n / 2 : n / 2 - 1;
    }

    dVec = hits.block(0, mId, 2, 1) - hits.block(0, 0, 2, 1);
    eVec = hits.block(0, n - 1, 2, 1) - hits.block(0, mId, 2, 1);
    results.qCharge = riemannFit::cross2D(acc, dVec, eVec) > 0 ? -1 : 1;

    const double slope = -results.qCharge / fast_fit(3);

    riemannFit::Matrix2d rotMat = rotationMatrix(acc, slope);

    // calculate radii and s
    results.radii = hits.block(0, 0, 2, n) - fast_fit.head(2) * riemannFit::MatrixXd::Constant(1, n, 1);
    eVec = -fast_fit(2) * fast_fit.head(2) / fast_fit.head(2).norm();
    for (u_int i = 0; i < n; i++) {
      dVec = results.radii.block(0, i, 2, 1);
      results.sTransverse(i) =
          results.qCharge * fast_fit(2) *
          alpaka::math::atan2(acc, riemannFit::cross2D(acc, dVec, eVec), dVec.dot(eVec));  // calculates the arc length
    }
    // zVec/pointsSZ storage comes from the caller's workspace (a Map over scratch for the kernel callers,
    // an owned Matrix otherwise); the fill expressions below (copy/zero/cwise stores) are the same either way.
    auto& zVec = fitWs.zVec;
    zVec = hits.block(2, 0, 1, n).transpose();
#ifdef BL_DEEPDEBUG
    for (u_int i = 0; i < n; i++) {
      printf("Point %d, x, %f, y: %f, z: %f, s: %f\n",
             i,
             hits.block(0, 0, 2, n)(0, i),
             hits.block(0, 0, 2, n)(1, i),
             zVec(i),
             results.sTransverse(i));
    }
#endif

    //calculate sTotal and zVec
    auto& pointsSZ = fitWs.pointsSZ;
    pointsSZ = riemannFit::Matrix2xNd<n>::Zero();
    for (u_int i = 0; i < n; i++) {
      pointsSZ(0, i) = results.sTransverse(i);
      pointsSZ(1, i) = zVec(i);
      pointsSZ.block(0, i, 2, 1) = rotMat * pointsSZ.block(0, i, 2, 1);
    }
    results.sTotal = pointsSZ.block(0, 0, 1, n).transpose();
    results.zInSZplane = pointsSZ.block(1, 0, 1, n).transpose();
#ifdef BL_DEEPDEBUG
    for (u_int i = 0; i < n; i++) {
      printf("Point %d, rot_s: %f, rot_z: %f\n", i, results.sTotal(i), results.zInSZplane(i));
    }
#endif
    // material X/X0 per segment from the Geant4 map. hits(0..2,i) = global x,y,z.
    auto rOf = [&](u_int j) { return alpaka::math::sqrt(acc, hits(0, j) * hits(0, j) + hits(1, j) * hits(1, j)); };
    // matXX0 slot g is the material charged at the kink of NODE g+1 (varBeta(i) reads slot i-1, and
    // generalBrokenLine::prepareGblData reads matXX0(i-1) at hit i).
    //   fitCorrections off: slot g = all of gap g, charged at its arrival node (total reproduced, first
    //        moment modelled as zero).
    //   fitCorrections on: slot g = the endpoint-partitioned share of node g+1 = (1-fDep) of gap g plus fDep of
    //        gap g+1 (segmentXX0Endpoint), so every gap's total and first moment are reproduced. Node 0 has
    //        no kink (varBeta(0) = 0), so gap 0's departure share is dropped; the last gap's departure share
    //        lands on node n-2.
    if (fitCorrections) {
      for (u_int i = 0; i < n; i++)
        results.matXX0(i) = 0.;
      // Running plain column for the dE/dx offset (elossGaps): gapXX0(g) = sum of the gap totals 0..g, i.e. the
      // material between hit 0 and node g+1, from the same map samples the partition uses.
      double xx0Run = 0.;
      for (u_int g = 0; g < n - 1; g++) {
        double fDep = 0.5;  // overwritten by segmentXX0Endpoint (uniform density => lever L/2)
        double xx0;
        xx0 = segmentXX0Endpoint(acc, rho, rOf(g), hits(2, g), rOf(g + 1), hits(2, g + 1), fDep, true);
        results.matXX0(g) += (1. - fDep) * xx0;  // arrival share   -> node g+1 -> slot g
        if (g > 0)
          results.matXX0(g - 1) += fDep * xx0;  // departure share -> node g   -> slot g-1
        if (elossGaps) {
          xx0Run += xx0;
          fitWs.gapXX0(g) = xx0Run;
        }
      }
      if (elossGaps)
        fitWs.gapXX0(n - 1) = 0.;  // no gap n-1; the walk never reads it
    } else {
      // Corrections OFF: upstream's material model exactly -- each gap charged as |Delta sTotal| times the
      // flat inverse radiation length 0.06/16 (no material-map lookup). Reproduces upstream's
      // multScatt(sTotal(i+1) - sTotal(i), ...) arguments bit-for-bit.
      constexpr double kInvX0 = 0.06 / 16.;
      for (u_int i = 0; i < n - 1; i++) {
        results.matXX0(i) = alpaka::math::abs(acc, results.sTotal(i + 1) - results.sTotal(i)) * kInvX0;
      }
      results.matXX0(n - 1) = 0.;
    }
    // beamline -> first hit material (beam pipe + everything upstream of the innermost fitted hit). The map is
    // always integrated from the beamline, also for tracks whose first hit is not beam-pipe-connected (OT-only
    // stub tracks, first hit at r ~ 23 cm): that is the CKF's prompt-origin material assumption, so pulls are
    // directly comparable, and without it the extrapolated PCA covariance of the (dominant) tracks that did
    // traverse the pixel volume would carry no upstream scattering or dE/dx. Genuinely displaced tracks are
    // over-covered, as in the CKF; prepareGblFitData applies the same rule.
    const bool useInner = true;
    if (fitCorrections) {
      results.innerXX0 = useInner ? segmentXX0(acc, rho, 0., 0., rOf(0), hits(2, 0), fitCorrections) : 0.;
    } else {
      // Corrections OFF: upstream adds the innermost multiple-scattering term from the FIRST GAP's
      // length -- multScatt(sTotal(1) - sTotal(0), ...) -- not from a beamline material integral.
      constexpr double kInvX0 = 0.06 / 16.;
      results.innerXX0 = alpaka::math::abs(acc, results.sTotal(1) - results.sTotal(0)) * kInvX0;
    }
    //calculate varBeta
    results.varBeta(0) = results.varBeta(n - 1) = 0;
    for (u_int i = 1; i < n - 1; i++) {
      if (fitCorrections) {
        // Slot i-1 holds the material assigned to node i (endpoint partition above): one thin scatterer per
        // node, Highland applied once to that node's total assigned thickness. Charging every gap at both of
        // its end nodes with its full variance (the branch below) would count each gap twice.
        results.varBeta(i) = multScatt(acc, results.matXX0(i - 1), bField, fast_fit(2), slope, true);
        if (!(results.varBeta(i) > 0.)) {
          // A node with no assigned material carries no kink information: its kink degree of freedom is absent,
          // not infinitely precise. A very large finite varBeta makes every 1/varBeta consumer (the band, the
          // border and both chi2 sums) evaluate to effectively 0 -- the mirror of the `if (th2 > 0.)` guard in
          // generalBrokenLine::prepareGblData -- and stays large under circleFit's (1+slope^2) and kMSoff scalings.
          results.varBeta(i) = std::numeric_limits<float>::max();
        }
      } else {
        results.varBeta(i) = multScatt(acc, results.matXX0(i), bField, fast_fit(2), slope) +
                             multScatt(acc, results.matXX0(i - 1), bField, fast_fit(2), slope);
      }
    }
  }

  /*!
    \brief Computes the n-by-n band matrix obtained minimizing the Broken Line's cost function w.r.t u. 
   *       This is the whole matrix in the case of the line fit and the main n-by-n block in the case 
   *       of the circle fit.
    
    \param weights weights of the first part of the cost function, the one with the measurements 
   *         and not the angles (\sum_{i=1}^n w*(y_i-u_i)^2).
    \param sTotal total distance traveled by the particle from the pre-fitted closest approach.
    \param varBeta kink angles' variance.
    
    \return the n-by-n matrix of the linear system
  */
  // Builds the symmetric band matrix directly in the caller-provided `out` (an n-by-n block of the bordered
  // band matrix, or lineFit's iMat), with no intermediate n-by-n temporary; `out` never aliases the inputs.
  // The diagonal is halved then doubled back (`out(i,i) = t + t`) and the strictly-upper/-lower entries are
  // stored as `t + 0.0` / `0.0 + t` on purpose: this reproduces, element for element, the arithmetic of
  // c + c.transpose(), and under IEEE-754 signed zero `x + 0.0` is not bit-identical to `x` at -O2 without
  // -fno-signed-zeros, so simplifying these operations away would change the stored values.
  // weights/sTotal/varBeta may be any statically-sized Eigen vector (an owned VectorNd<n>, or an
  // Eigen::Map over per-lane scratch, which is what the kernel callers pass); n comes from the row count.
  template <alpaka::concepts::Acc TAcc, typename VW, typename VS, typename VB, typename MOut, int n = VW::RowsAtCompileTime>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void matrixC_u(
      const TAcc& acc, const VW& weights, const VS& sTotal, const VB& varBeta, MOut&& out) {
    for (u_int r = 0; r < n; r++)
      for (u_int c = 0; c < n; c++)
        out(r, c) = 0.;
    for (u_int i = 0; i < n; i++) {
      out(i, i) = weights(i);
      if (i > 1)
        out(i, i) += 1. / (varBeta(i - 1) * riemannFit::sqr(sTotal(i) - sTotal(i - 1)));
      if (i > 0 && i < n - 1)
        out(i, i) += (1. / varBeta(i)) * riemannFit::sqr((sTotal(i + 1) - sTotal(i - 1)) /
                                                         ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
      if (i < n - 2)
        out(i, i) += 1. / (varBeta(i + 1) * riemannFit::sqr(sTotal(i + 1) - sTotal(i)));

      if (i > 0 && i < n - 1)
        out(i, i + 1) =
            1. / (varBeta(i) * (sTotal(i + 1) - sTotal(i))) *
            (-(sTotal(i + 1) - sTotal(i - 1)) / ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
      if (i < n - 2)
        out(i, i + 1) +=
            1. / (varBeta(i + 1) * (sTotal(i + 1) - sTotal(i))) *
            (-(sTotal(i + 2) - sTotal(i)) / ((sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i))));

      if (i < n - 2)
        out(i, i + 2) = 1. / (varBeta(i + 1) * (sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i)));

      out(i, i) *= 0.5;
    }
    for (u_int i = 0; i < n; i++) {
      const double td = out(i, i);
      out(i, i) = td + td;
      for (u_int j = i + 1; j < n; j++) {
        const double t = out(i, j);
        out(i, j) = t + 0.;
        out(j, i) = 0. + t;
      }
    }
  }

  // By-value overload, returning the dense n-by-n band: the CPP_DUMP diagnostic block is its only caller
  // (the fits themselves build the band lower-packed, through matrixC_uBandLower). Delegates to the
  // fill-in-place form so the band arithmetic has a single definition.
  template <alpaka::concepts::Acc TAcc, int n>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE riemannFit::MatrixNd<n> matrixC_u(const TAcc& acc,
                                                                   const riemannFit::VectorNd<n>& weights,
                                                                   const riemannFit::VectorNd<n>& sTotal,
                                                                   const riemannFit::VectorNd<n>& varBeta) {
    riemannFit::MatrixNd<n> out;
    matrixC_u(acc, weights, sTotal, varBeta, out);
    return out;
  }

  // Build the symmetric pentadiagonal band as matrixC_u (b = 2), written directly in the lower-packed layout
  // generalBrokenLine::bandFactor<2>/bandSolveInPlace<2> consume: Mb[i*3 + p] = M(i, i-p), p = 0..min(i,2)
  // (row stride 3 = kBand+1, lstride 1). The per-element arithmetic is matrixC_u's, so the assembled M is
  // identical (the dense build's halve/double diagonal round trip is value-identical for the diagonal). Only the
  // packed slots bandFactor<2> reads are written; the three structurally-absent slots (row 0 offsets 1,2 and
  // row 1 offset 2) are never read by the solver and are intentionally left unwritten.
  template <typename VW, typename VS, typename VB, int n = VW::RowsAtCompileTime>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void matrixC_uBandLower(
      const VW& weights, const VS& sTotal, const VB& varBeta, double* Mb, int lstride) {
    for (u_int i = 0; i < n; i++) {
      double d = weights(i);
      if (i > 1)
        d += 1. / (varBeta(i - 1) * riemannFit::sqr(sTotal(i) - sTotal(i - 1)));
      if (i > 0 && i < n - 1)
        d += (1. / varBeta(i)) * riemannFit::sqr((sTotal(i + 1) - sTotal(i - 1)) /
                                                 ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
      if (i < n - 2)
        d += 1. / (varBeta(i + 1) * riemannFit::sqr(sTotal(i + 1) - sTotal(i)));
      Mb[(i * 3) * lstride] = d;  // M(i,i)

      if (i + 1 < n) {  // M(i, i+1) = M(i+1, i), stored lower-packed at row i+1 offset 1
        double o1 = 0.;
        if (i > 0 && i < n - 1)
          o1 = 1. / (varBeta(i) * (sTotal(i + 1) - sTotal(i))) *
               (-(sTotal(i + 1) - sTotal(i - 1)) / ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
        if (i < n - 2)
          o1 += 1. / (varBeta(i + 1) * (sTotal(i + 1) - sTotal(i))) *
                (-(sTotal(i + 2) - sTotal(i)) / ((sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i))));
        Mb[((i + 1) * 3 + 1) * lstride] = o1;
      }
      if (i + 2 < n)  // M(i, i+2) = M(i+2, i), stored lower-packed at row i+2 offset 2
        Mb[((i + 2) * 3 + 2) * lstride] =
            1. / (varBeta(i + 1) * (sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i)));
    }
  }

  // Build the circle border column c(i,n) into cbor[i] and return the corner
  // c(n,n). Verbatim the circleFit border build (matrixC_u border block), with c_uMat(i,n) -> cbor[i] and
  // c_uMat(n,n) -> the returned scalar; the symmetric c_uMat(n,i) copy is implicit (never stored separately).
  template <typename VS, typename VB, int n = VS::RowsAtCompileTime>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double circleBorderLower(const VS& sTransverse,
                                                          const VB& varBeta,
                                                          double* cbor,
                                                          int lstride) {
    double corner = 0.;
    for (u_int i = 0; i < n; i++) {
      double b = 0.;
      if (i > 0 && i < n - 1)
        b += -(sTransverse(i + 1) - sTransverse(i - 1)) * (sTransverse(i + 1) - sTransverse(i - 1)) /
             (2. * varBeta(i) * (sTransverse(i + 1) - sTransverse(i)) * (sTransverse(i) - sTransverse(i - 1)));
      if (i > 1)
        b += (sTransverse(i) - sTransverse(i - 2)) / (2. * varBeta(i - 1) * (sTransverse(i) - sTransverse(i - 1)));
      if (i < n - 2)
        b += (sTransverse(i + 2) - sTransverse(i)) / (2. * varBeta(i + 1) * (sTransverse(i + 1) - sTransverse(i)));
      cbor[i * lstride] = b;
      if (i > 0 && i < n - 1)
        corner += riemannFit::sqr(sTransverse(i + 1) - sTransverse(i - 1)) / (4. * varBeta(i));
    }
    return corner;
  }

  /*!
    \brief A very fast helix fit.
    
    \param hits the measured hits.
    
    \return (X0,Y0,R,tan(theta)).
    
    \warning sign of theta is (intentionally, for now) mistaken for negative charges.
  */

  template <alpaka::concepts::Acc TAcc, typename M3xN, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fastFit(const TAcc& acc, const M3xN& hits, V4& result) {
    constexpr uint32_t n = M3xN::ColsAtCompileTime;

    int mId = 1;

    if constexpr (n > 3) {
      riemannFit::Vector2d middle = 0.5 * (hits.block(0, n - 1, 2, 1) + hits.block(0, 0, 2, 1));
      auto d1 = (hits.block(0, n / 2, 2, 1) - middle).squaredNorm();
      auto d2 = (hits.block(0, n / 2 - 1, 2, 1) - middle).squaredNorm();
      mId = d1 < d2 ? n / 2 : n / 2 - 1;
    }

    const riemannFit::Vector2d a = hits.block(0, mId, 2, 1) - hits.block(0, 0, 2, 1);
    const riemannFit::Vector2d b = hits.block(0, n - 1, 2, 1) - hits.block(0, mId, 2, 1);
    const riemannFit::Vector2d c = hits.block(0, 0, 2, 1) - hits.block(0, n - 1, 2, 1);

    auto tmp = 0.5 / riemannFit::cross2D(acc, c, a);
    result(0) = hits(0, 0) - (a(1) * c.squaredNorm() + c(1) * a.squaredNorm()) * tmp;
    result(1) = hits(1, 0) + (a(0) * c.squaredNorm() + c(0) * a.squaredNorm()) * tmp;
    // check Wikipedia for these formulas

    result(2) = alpaka::math::sqrt(acc, a.squaredNorm() * b.squaredNorm() * c.squaredNorm()) /
                (2. * alpaka::math::abs(acc, riemannFit::cross2D(acc, b, a)));
    // Using Math Olympiad's formula R=abc/(4A)

    const riemannFit::Vector2d d = hits.block(0, 0, 2, 1) - result.head(2);
    const riemannFit::Vector2d e = hits.block(0, n - 1, 2, 1) - result.head(2);

    result(3) = result(2) * atan2(riemannFit::cross2D(acc, d, e), d.dot(e)) / (hits(2, n - 1) - hits(2, 0));
    // ds/dz slope between last and first point
#ifdef BL_DEEPDEBUG
    printf("FastFit results(x,y,R,tan(theta): %e, %e, %e, %e\n", result(0), result(1), result(2), result(3));
#endif
  }

  /*!
    \brief Performs the Broken Line fit in the curved track case (that is, the fit 
   *       parameters are the interceptions u and the curvature correction \Delta\kappa).
    
    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineDataMap.
    \param circle_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (phi, d, k); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    
    \param fitCorrections selects the fast-BL scattering/covariance package (see the file head): the reference
           hit of the circle frame, the momentum factor multScatt uses, and whether the covariance blend runs
           (with corrections off it is skipped and the emitted covariance is upstream's).
    \param elossCurvPerXX0 the track's ionization curvature growth per unit material column,
           kappa_0 * (dE/dX)_eff / p  [1/cm per X/X0], or 0 to run without any dE/dx term. The caller builds it
           once per track from generalBrokenLine::elossTypicalColumn at the track's total column (Kernel_BLFit),
           so the Landau law is evaluated once and not per node. Non-zero requires prepareBrokenLineData to have
           been called with elossGaps = true (it reads fitWs.gapXX0); zero leaves upstream's arithmetic intact.

    \details The function implements the steps 2 and 3 of the Broken Line fit
   *         with the curvature correction.\n
   * The step 2 is the least square fit, done by imposing the minimum constraint on 
   * the cost function and solving the consequent linear system. It determines the 
   * fitted parameters u and \Delta\kappa and their covariance matrix.
   * The step 3 is the correction of the fast pre-fitted parameters for the innermost 
   * part of the track. It is first done in a comfortable coordinate system (the one 
   * in which the first hit is the origin) and then the parameters and their 
   * covariance matrix are transformed to the original coordinate system.
  */
  template <alpaka::concepts::Acc TAcc,
            typename M3xN,
            typename M6xN,
            typename V4,
            typename TData,
            typename TWs,
            int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void circleFit(const TAcc& acc,
                                                const M3xN& hits,
                                                const M6xN& hits_ge,
                                                const V4& fast_fit,
                                                const double bField,
                                                TData& data,
                                                karimaki_circle_fit& circle_results,
                                                TWs& fitWs,
                                                bool fitCorrections = false,
                                                double elossCurvPerXX0 = 0.) {
    circle_results.qCharge = data.qCharge;
    auto& radii = data.radii;
    const auto& sTransverse = data.sTransverse;
    auto& zInSZplane = data.zInSZplane;
    auto& varBeta = data.varBeta;
    const double slope = -circle_results.qCharge / fast_fit(3);
    varBeta *= 1. + riemannFit::sqr(slope);  // the kink angles are projected!

    for (u_int i = 0; i < n; i++) {
      zInSZplane(i) = radii.block(0, i, 2, 1).norm() - fast_fit(2);
    }

    // IONIZATION ENERGY LOSS, deterministic part (fit-corrections package; elossCurvPerXX0 == 0 skips the block).
    // The fit models one circle: u(s), the radial offset from the reference circle, obeys u'' = -(kappa(s) -
    // kappa_ref) with Delta-kappa a CONSTANT, while the real curvature grows along the path as the track loses
    // momentum, kappa(s) = kappa_0 (1 + dE(s)/p); the least-squares Delta-kappa then lands on a path average of
    // kappa and the published pT is low. The known growth is removed as the GBL refit removes it (the Deloss
    // accumulator): u_eloss(s) = -int_0^s ds' int_0^s' dkappa, dkappa(x) = elossCurvPerXX0 * X(x), is subtracted
    // from the measured residuals, so the fitted curvature is the one at the anchor, node 0 (u_eloss = u_eloss'
    // = 0 there; anchoring at the PCA differs by an affine function of s, which the u_i absorb, so Delta-kappa is
    // invariant and only (phi, d0) move, at the sub-micron level). Node 0's column is innerXX0, not zero. The
    // column of a gap is taken at its midpoint (trapezoid rule for the double integral). No transcendental here:
    // the caller paid the one Landau evaluation; gapXX0 (zVec's slot) is rewritten in place.
    const bool eloss = (elossCurvPerXX0 > 0.);
    auto& uEloss = fitWs.gapXX0;  // in: running column from hit 0 per node; out: the dE/dx residual offset
    if (eloss) {
      double xPrev = data.innerXX0;             // material column at node 0 (beamline -> hit 0)
      double xCur = data.innerXX0 + uEloss(0);  // ... at node 1; read BEFORE slot 0 is overwritten
      double du = 0., dup = 0.;
      uEloss(0) = 0.;  // the anchor
      for (u_int k = 0; k + 1 < n; ++k) {
        const double ds = sTransverse(k + 1) - sTransverse(k);
        const double dk = elossCurvPerXX0 * 0.5 * (xPrev + xCur);  // mid-gap curvature increment [1/cm]
        du += dup * ds - 0.5 * dk * ds * ds;
        dup -= dk * ds;
        xPrev = xCur;
        if (k + 2 < n)
          xCur = data.innerXX0 + uEloss(k + 1);  // next node's column, read BEFORE the store below
        uEloss(k + 1) = du;
      }
    }

    riemannFit::Matrix2d vMat;  // covariance matrix
    // weightsVec is workspace-backed (a Map over scratch for the kernel callers, an owned vector
    // otherwise); filled below by scalar stores only.
    auto& weightsVec = fitWs.circleWeights;  // weights
    riemannFit::Matrix2d rotMat;             // rotation matrix point by point
    for (u_int i = 0; i < n; i++) {
      vMat(0, 0) = hits_ge.col(i)[0];               // x errors
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];  // cov_xy
      vMat(1, 1) = hits_ge.col(i)[2];               // y errors
      rotMat = rotationMatrix(acc, -radii(0, i) / radii(1, i));
      weightsVec(i) =
          1. / ((rotMat * vMat * rotMat.transpose())(1, 1));  // compute the orthogonal weight point by point
    }

    auto& r_uVec = fitWs.circleRu;  // workspace-backed storage; filled by scalar stores only
    // The normal equations are M u = r with r(i) = w(i) * (measured offset), r(n) = 0: the kink term does not
    // involve the measurements, so the dE/dx offset enters at exactly one place, the measured offset. Two loops
    // rather than one predicated loop, so that the no-eloss path is upstream's instruction stream verbatim.
    r_uVec(n) = 0;
    if (eloss) {
      for (u_int i = 0; i < n; i++) {
        r_uVec(i) = weightsVec(i) * (zInSZplane(i) - uEloss(i));
      }
    } else {
      for (u_int i = 0; i < n; i++) {
        r_uVec(i) = weightsVec(i) * zInSZplane(i);
      }
    }

    // Solve the (N+1)x(N+1) bordered-pentadiagonal circle normal matrix with the O(N) root-free LDLt band
    // factor + scalar Schur border elimination (Blobel NIM A566 eq.8), rather than forming the dense (N+1)^2
    // inverse: Mb = lower-packed pentadiagonal band, cbor = border column, corner = (n,n) entry, w = M^-1 cbor.
    // The band block is shared with lineFit (which uses Mb only) and reused in place by the MS-off re-fit below;
    // this is valid because lineFit precedes circleFit and the primary uVec + covariance are read out before the
    // MS-off rebuild. laneStride is the [element][lane] stride of the raw band arrays (1 for owned storage).
    static_assert(kLegacyBandBlockDoubles<n> == 6 * n + 1,
                  "band block [Mb 3n | cbor n | corner 1 | w n | ms n] shared by lineFit (Mb only) and circleFit; "
                  "the shared Mb requires lineFit to run (and read out) before circleFit rebuilds it");
    double schur = 0.;  // held from the primary solve to the covariance extraction below
    auto& uVec = fitWs.circleU;
    {
      const int ls = fitWs.laneStride;
      double* const Mb = fitWs.bandMb;
      double* const cbor = fitWs.bandCbor;
      double* const wcol = fitWs.bandW;
      double* const msc = fitWs.bandMs;
      matrixC_uBandLower(weightsVec, sTransverse, varBeta, Mb, ls);
      fitWs.bandCorner[0] = circleBorderLower(sTransverse, varBeta, cbor, ls);
      generalBrokenLine::bandFactor<2>(Mb, n, ls);
      for (u_int i = 0; i < n; i++)
        wcol[i * ls] = cbor[i * ls];
      generalBrokenLine::bandSolveInPlace<2>(Mb, wcol, n, ls);  // wcol = M^-1 cbor
      schur = fitWs.bandCorner[0];
      for (u_int i = 0; i < n; i++)
        schur -= cbor[i * ls] * wcol[i * ls];  // schur = corner - cbor^T M^-1 cbor
      // uVec = A^-1 r_uVec: border x_n = (r_n - w^T rb)/schur, band x = M^-1 rb - w x_n.
      double dot = 0.;
      for (u_int i = 0; i < n; i++) {
        msc[i * ls] = r_uVec(i);
        dot += wcol[i * ls] * r_uVec(i);
      }
      generalBrokenLine::bandSolveInPlace<2>(Mb, msc, n, ls);  // msc = M^-1 rb
      const double xN = (r_uVec(n) - dot) / schur;
      for (u_int i = 0; i < n; i++)
        uVec(i) = msc[i * ls] - wcol[i * ls] * xN;
      uVec(n) = xN;
    }

    // compute (phi, d_ca, k) in the system in which the midpoint of the first two corrected hits is the origin...
    // Corrections package: build the circle reference frame on hit 0 and the first hit at least kBaseMin
    // (transverse) away from it instead of upstream's hit 1: same-layer overlap / fishbone pairs and consecutive
    // disk hits sit on chords far shorter than the track, which inflates the propagated (phi,d0) covariance.
    // The Schur-complement extraction below works for any reference index. Corrections off: mref == 1.
    u_int mref = 1;
    if (fitCorrections) {
      constexpr double kBaseMin2 = 2.0 * 2.0;  // cm^2: minimum transverse baseline (squared)
      mref = n - 1;
      for (u_int j = 1; j < n; ++j) {
        double dx = hits(0, j) - hits(0, 0), dy = hits(1, j) - hits(1, 0);
        if (dx * dx + dy * dy > kBaseMin2) {
          mref = j;
          break;
        }
      }
    }

    radii.block(0, 0, 2, 1) /= radii.block(0, 0, 2, 1).norm();
    radii.block(0, mref, 2, 1) /= radii.block(0, mref, 2, 1).norm();

    riemannFit::Vector2d dVec = hits.block(0, 0, 2, 1) + (-zInSZplane(0) + uVec(0)) * radii.block(0, 0, 2, 1);
    riemannFit::Vector2d eVec =
        hits.block(0, mref, 2, 1) + (-zInSZplane(mref) + uVec(mref)) * radii.block(0, mref, 2, 1);
    auto eMinusd = eVec - dVec;
    auto eMinusd2 = eMinusd.squaredNorm();
    auto tmp1 = 1. / eMinusd2;
    auto tmp2 = alpaka::math::sqrt(acc, riemannFit::sqr(fast_fit(2)) - 0.25 * eMinusd2);

    circle_results.par << atan2(eMinusd(1), eMinusd(0)), circle_results.qCharge * (tmp2 - fast_fit(2)),
        circle_results.qCharge * (1. / fast_fit(2) + uVec(n));
    const auto par0 = circle_results.par;  // pre-translate params, reused by the MS-off re-extraction below

    tmp2 = 1. / tmp2;

    riemannFit::Matrix3d jacobian;
    jacobian << (radii(1, 0) * eMinusd(0) - eMinusd(1) * radii(0, 0)) * tmp1,
        (radii(1, mref) * eMinusd(0) - eMinusd(1) * radii(0, mref)) * tmp1, 0,
        circle_results.qCharge * (eMinusd(0) * radii(0, 0) + eMinusd(1) * radii(1, 0)) * tmp2,
        circle_results.qCharge * (eMinusd(0) * radii(0, mref) + eMinusd(1) * radii(1, mref)) * tmp2, 0, 0, 0,
        circle_results.qCharge;

    // Raw (0, mref, n)^2 = (phi, d, k) block of A^-1. Selected-element extraction via the Schur formula: for band
    // indices g,h in {0,mref}, A^-1(g,h) = M^-1(g,h) + w(g) w(h)/schur; A^-1(g,n) = -w(g)/schur; A^-1(n,n) =
    // 1/schur; the two M^-1 columns come from unit-column band solves (e_0, e_mref).
    {
      const int ls = fitWs.laneStride;
      double* const Mb = fitWs.bandMb;
      double* const wcol = fitWs.bandW;
      double* const msc = fitWs.bandMs;
      const double invS = 1. / schur;
      for (u_int i = 0; i < n; i++)
        msc[i * ls] = 0.;
      msc[0] = 1.;
      generalBrokenLine::bandSolveInPlace<2>(Mb, msc, n, ls);  // msc = M^-1 e_0
      const double mi00 = msc[0], mim0 = msc[mref * ls];       // M^-1(0,0), M^-1(mref,0)
      for (u_int i = 0; i < n; i++)
        msc[i * ls] = 0.;
      msc[mref * ls] = 1.;
      generalBrokenLine::bandSolveInPlace<2>(Mb, msc, n, ls);  // msc = M^-1 e_mref
      const double mi0m = msc[0], mimm = msc[mref * ls];       // M^-1(0,mref), M^-1(mref,mref)
      const double w0 = wcol[0], wm = wcol[mref * ls];
      circle_results.cov << mi00 + w0 * w0 * invS, mi0m + w0 * wm * invS, -w0 * invS, mim0 + wm * w0 * invS,
          mimm + wm * wm * invS, -wm * invS, -w0 * invS, -wm * invS, invS;
    }
    const riemannFit::Matrix3d jac2hit = jacobian;  // save the 2-hit Jacobian (translateKarimaki overwrites jacobian)

    circle_results.cov = jacobian * circle_results.cov * jacobian.transpose();

    //...Translate in the system in which the first corrected hit is the origin, adding the m.s. correction...

    translateKarimaki(acc, circle_results, 0.5 * eMinusd(0), 0.5 * eMinusd(1), jacobian);
    // innermost MS: add ONLY the upstream (beam pipe -> first hit) material here. The matXX0(0) (first hit -> hit
    // segment) scattering is already folded into the fit covariance via varBeta(1), so it must not be added again.
    circle_results.cov(0, 0) +=
        (1 + riemannFit::sqr(slope)) * multScatt(acc, data.innerXX0, bField, fast_fit(2), slope, fitCorrections);

    //...And translate back to the original system

    translateKarimaki(acc, circle_results, dVec(0), dVec(1), jacobian);

    // Seam-free covariance blend for the (phi,d0) block:  C_blend = C_F + (C_L - C_L^MSoff).
    // C_F = measurement cov from the 3-parameter-circle Fisher inverse (incl. the d0-kappa coupling); C_L = the
    // broken-line cov just computed (measurement + correlated kink MS + innerXX0); C_L^MSoff = the same cov
    // re-built with the MS kinks frozen (varBeta -> ~0), re-extracted through the SAME 2-hit Jacobian and the
    // SAME two translateKarimaki frames, without the innerXX0 add. (C_L - C_L^MSoff) is the broken line's full
    // MS part; grafting it onto C_F substitutes the full-circle measurement cov for the 2-hit one. High pt:
    // C_blend -> C_F; low pt: C_blend ~ C_L. Gated on fitCorrections, since off the covariance must be upstream's.
    if (fitCorrections) {
      const double cph = alpaka::math::cos(acc, circle_results.par(0));
      const double sph = alpaka::math::sin(acc, circle_results.par(0));
      riemannFit::Matrix3d fisher = riemannFit::Matrix3d::Zero();
      for (u_int i = 0; i < n; i++) {
        const double xx = hits_ge.col(i)[0], xy = hits_ge.col(i)[1], yy = hits_ge.col(i)[2];
        const double sv2 = sph * sph * xx - 2. * sph * cph * xy + cph * cph * yy;  // track-perpendicular (rphi) var
        if (!(sv2 > 0.))
          continue;
        const double u = hits(0, i) * cph + hits(1, i) * sph;  // along-track from origin
        const double w = 1. / sv2;
        // Residual derivatives d v / d (phi, d0, rho) in the Karimaki convention {u, -1, -u^2/2}, the basis the
        // blend's other operand C_L is expressed in. (The {u, +1, +u^2/2} basis is its D = diag(1,-1,-1)
        // mirror: same diagonal, opposite sign of the (phi,d0) and (phi,rho) entries that cov(1) = covPhiDxy
        // carries to the consumers.)
        constexpr double gs = -1.;  // Karimaki-convention basis
        const double g[3] = {u, gs, gs * 0.5 * u * u};
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            fisher(a, b) += w * g[a] * g[b];
      }
      if (alpaka::math::abs(acc, fisher.determinant()) > 0.) {
        riemannFit::Matrix3d fcov;
        math::cholesky::invert(fisher, fcov);
        // C_L^MSoff: re-build the band with the kinks frozen (varBeta -> ~0 => rigid), re-invert, re-extract.
        constexpr double kMSoff = 1.e-6;
        auto& varBetaOff = fitWs.varBetaOff;
        varBetaOff = varBeta * kMSoff;
        // MS-off re-fit: rebuild the bordered band with varBetaOff in place (valid because the primary uVec and
        // cov were read out above), refactor + Schur, extract the same {0,mref,n}^2 block via two unit-column
        // solves (no parameter solve: offFit reads only cov), then the SAME 2-hit Jacobian and the SAME two
        // translates, with no innerXX0 add.
        karimaki_circle_fit offFit;
        offFit.qCharge = circle_results.qCharge;
        offFit.par = par0;
        {
          const int ls = fitWs.laneStride;
          double* const Mb = fitWs.bandMb;
          double* const cbor = fitWs.bandCbor;
          double* const wcol = fitWs.bandW;
          double* const msc = fitWs.bandMs;
          matrixC_uBandLower(weightsVec, sTransverse, varBetaOff, Mb, ls);
          const double cornerOff = circleBorderLower(sTransverse, varBetaOff, cbor, ls);
          generalBrokenLine::bandFactor<2>(Mb, n, ls);
          for (u_int i = 0; i < n; i++)
            wcol[i * ls] = cbor[i * ls];
          generalBrokenLine::bandSolveInPlace<2>(Mb, wcol, n, ls);
          double schurOff = cornerOff;
          for (u_int i = 0; i < n; i++)
            schurOff -= cbor[i * ls] * wcol[i * ls];
          const double invS = 1. / schurOff;
          for (u_int i = 0; i < n; i++)
            msc[i * ls] = 0.;
          msc[0] = 1.;
          generalBrokenLine::bandSolveInPlace<2>(Mb, msc, n, ls);  // msc = M^-1 e_0
          const double mi00 = msc[0], mim0 = msc[mref * ls];
          for (u_int i = 0; i < n; i++)
            msc[i * ls] = 0.;
          msc[mref * ls] = 1.;
          generalBrokenLine::bandSolveInPlace<2>(Mb, msc, n, ls);  // msc = M^-1 e_mref
          const double mi0m = msc[0], mimm = msc[mref * ls];
          const double w0 = wcol[0], wm = wcol[mref * ls];
          offFit.cov << mi00 + w0 * w0 * invS, mi0m + w0 * wm * invS, -w0 * invS, mim0 + wm * w0 * invS,
              mimm + wm * wm * invS, -wm * invS, -w0 * invS, -wm * invS, invS;
        }
        offFit.cov = jac2hit * offFit.cov * jac2hit.transpose();
        riemannFit::Matrix3d jtmp;
        translateKarimaki(acc, offFit, 0.5 * eMinusd(0), 0.5 * eMinusd(1), jtmp);
        translateKarimaki(acc, offFit, dVec(0), dVec(1), jtmp);
        // Blend: write the full 3x3 from the one model, the (phi,d0) 2x2 and the kappa row/column (already
        // computed). C_L = circle_results.cov; all differences are read before any write.
        const double msPhiPhi = circle_results.cov(0, 0) - offFit.cov(0, 0);
        const double msPhiD0 = circle_results.cov(0, 1) - offFit.cov(0, 1);
        const double msD0D0 = circle_results.cov(1, 1) - offFit.cov(1, 1);
        const double msPhiK = circle_results.cov(0, 2) - offFit.cov(0, 2);
        const double msD0K = circle_results.cov(1, 2) - offFit.cov(1, 2);
        const double msKK = circle_results.cov(2, 2) - offFit.cov(2, 2);
        circle_results.cov(0, 0) = fcov(0, 0) + msPhiPhi;
        circle_results.cov(0, 1) = fcov(0, 1) + msPhiD0;
        circle_results.cov(1, 0) = fcov(1, 0) + msPhiD0;
        circle_results.cov(1, 1) = fcov(1, 1) + msD0D0;
        circle_results.cov(0, 2) = fcov(0, 2) + msPhiK;
        circle_results.cov(2, 0) = fcov(0, 2) + msPhiK;
        circle_results.cov(1, 2) = fcov(1, 2) + msD0K;
        circle_results.cov(2, 1) = fcov(1, 2) + msD0K;
        circle_results.cov(2, 2) = fcov(2, 2) + msKK;
      }
    }

    // compute chi2. The model at node i is uVec(i) + uEloss(i), so the measurement term carries the same offset
    // the right-hand side did; otherwise the chi2 would grow by exactly the spiral the fit was told to expect.
    // The kink term is not offset: uEloss is a deterministic curvature evolution, not a scatter.
    circle_results.chi2 = 0;
    for (u_int i = 0; i < n; i++) {
      circle_results.chi2 +=
          weightsVec(i) * riemannFit::sqr(eloss ? (zInSZplane(i) - uEloss(i) - uVec(i)) : (zInSZplane(i) - uVec(i)));
      if (i > 0 && i < n - 1)
        circle_results.chi2 +=
            riemannFit::sqr(uVec(i - 1) / (sTransverse(i) - sTransverse(i - 1)) -
                            uVec(i) * (sTransverse(i + 1) - sTransverse(i - 1)) /
                                ((sTransverse(i + 1) - sTransverse(i)) * (sTransverse(i) - sTransverse(i - 1))) +
                            uVec(i + 1) / (sTransverse(i + 1) - sTransverse(i)) +
                            (sTransverse(i + 1) - sTransverse(i - 1)) * uVec(n) / 2) /
            varBeta(i);
    }
  }

  /*!
    \brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).
    
    \param hits hits coordinates.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineDataMap.
    \param line_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (cot(theta), Zip); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    
    \param fitCorrections selects the fast-BL scattering/covariance package (see the file head); here it
           selects the momentum factor of the innermost-material multScatt call.

    \details The function implements the steps 2 and 3 of the Broken Line fit without
   *        the curvature correction.\n
   * The step 2 is the least square fit, done by imposing the minimum constraint 
   * on the cost function and solving the consequent linear system. It determines 
   * the fitted parameters u and their covariance matrix.
   * The step 3 is the correction of the fast pre-fitted parameters for the innermost 
   * part of the track. It is first done in a comfortable coordinate system (the one 
   * in which the first hit is the origin) and then the parameters and their covariance 
   * matrix are transformed to the original coordinate system.
   */
  template <alpaka::concepts::Acc TAcc, typename V4, typename M6xN, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void lineFit(const TAcc& acc,
                                              const M6xN& hits_ge,
                                              const V4& fast_fit,
                                              const double bField,
                                              const TData& data,
                                              riemannFit::LineFit& line_results,
                                              TWs& fitWs,
                                              bool fitCorrections = false) {
    const auto& radii = data.radii;
    const auto& sTotal = data.sTotal;
    const auto& zInSZplane = data.zInSZplane;
    const auto& varBeta = data.varBeta;

    const double slope = -data.qCharge / fast_fit(3);
#ifdef BL_DEEPDEBUG
    printf("Slope: %e, charge: %d, curvature: %e\n", slope, data.qCharge, fast_fit(3));
#endif
    riemannFit::Matrix2d rotMat = rotationMatrix(acc, slope);

    riemannFit::Matrix3d vMat = riemannFit::Matrix3d::Zero();  // covariance matrix XYZ
    riemannFit::Matrix2x3d jacobXYZtosZ =
        riemannFit::Matrix2x3d::Zero();  // jacobian for computation of the error on s (xyz -> sz)
    // weights is workspace-backed (a Map over scratch for the kernel callers, an owned vector otherwise);
    // zero-filled then set by scalar stores below.
    auto& weights = fitWs.lineWeights;
    weights = riemannFit::VectorNd<n>::Zero();
    for (u_int i = 0; i < n; i++) {
      vMat(0, 0) = hits_ge.col(i)[0];               // x errors
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];  // cov_xy
      vMat(0, 2) = vMat(2, 0) = hits_ge.col(i)[3];  // cov_xz
      vMat(1, 1) = hits_ge.col(i)[2];               // y errors
      vMat(2, 1) = vMat(1, 2) = hits_ge.col(i)[4];  // cov_yz
      vMat(2, 2) = hits_ge.col(i)[5];               // z errors
      auto tmp = 1. / radii.block(0, i, 2, 1).norm();
      jacobXYZtosZ(0, 0) = data.qCharge * radii(1, i) * tmp;
      jacobXYZtosZ(0, 1) = -data.qCharge * radii(0, i) * tmp;
      jacobXYZtosZ(1, 2) = 1.;
      weights(i) = 1. / ((rotMat * jacobXYZtosZ * vMat * jacobXYZtosZ.transpose() * rotMat.transpose())(
                            1, 1));  // compute the orthogonal weight point by point
    }

    auto& r_u = fitWs.lineRu;  // workspace-backed storage; filled by scalar stores only
    for (u_int i = 0; i < n; i++) {
      r_u(i) = weights(i) * zInSZplane(i);
    }
#ifdef CPP_DUMP
    std::cout << "CU4\n" << matrixC_u(weights, sTotal, varBeta) << std::endl;
#endif
    // Solve the NxN pentadiagonal line normal matrix with the O(N) root-free LDLt band factor (no border: the
    // line has no curvature parameter), reusing the shared band block's Mb. lineFit runs before circleFit, so Mb
    // is free here; circleFit rebuilds it afterwards. laneStride is the raw band arrays' [element][lane] stride.
    auto& uVec = fitWs.lineU;
    {
      const int ls = fitWs.laneStride;
      double* const Mb = fitWs.bandMb;
      matrixC_uBandLower(weights, sTotal, varBeta, Mb, ls);
      generalBrokenLine::bandFactor<2>(Mb, n, ls);
      for (u_int i = 0; i < n; i++)  // load the rhs, then solve in place: uVec = M^-1 r_u
        uVec(i) = r_u(i);
      generalBrokenLine::bandSolveInPlace<2>(Mb, uVec.data(), n, ls);
    }

    // line parameters in the system in which the first hit is the origin and with axis along SZ
    line_results.par << (uVec(1) - uVec(0)) / (sTotal(1) - sTotal(0)), uVec(0);
    auto idiff = 1. / (sTotal(1) - sTotal(0));
    // innermost MS: add ONLY innerXX0 (upstream). matXX0(0) is already in the fit via varBeta(1), so it must not
    // be added again here. This is the only MS correction for the line fit. The line cov reads only M^-1(0,0),
    // M^-1(0,1), M^-1(1,1), from two unit-column band solves (e_0, e_1).
    {
      const int ls = fitWs.laneStride;
      double* const Mb = fitWs.bandMb;
      double* const col = fitWs.bandMs;
      for (u_int i = 0; i < n; i++)
        col[i * ls] = 0.;
      col[0] = 1.;
      generalBrokenLine::bandSolveInPlace<2>(Mb, col, n, ls);  // col = M^-1 e_0
      const double i00 = col[0];
      for (u_int i = 0; i < n; i++)
        col[i * ls] = 0.;
      col[1 * ls] = 1.;
      generalBrokenLine::bandSolveInPlace<2>(Mb, col, n, ls);  // col = M^-1 e_1
      const double i01 = col[0], i11 = col[1 * ls];
      line_results.cov << (i00 - 2 * i01 + i11) * riemannFit::sqr(idiff) +
                              multScatt(acc, data.innerXX0, bField, fast_fit(2), slope, fitCorrections),
          (i01 - i00) * idiff, (i01 - i00) * idiff, i00;
    }

    // translate to the original SZ system
    riemannFit::Matrix2d jacobian;
    jacobian(0, 0) = 1.;
    jacobian(0, 1) = 0;
    jacobian(1, 0) = -sTotal(0);
    jacobian(1, 1) = 1.;
    line_results.par(1) += -line_results.par(0) * sTotal(0);
    line_results.cov = jacobian * line_results.cov * jacobian.transpose();

    // rotate to the original sz system
    auto tmp = rotMat(0, 0) - line_results.par(0) * rotMat(0, 1);
    jacobian(1, 1) = 1. / tmp;
    jacobian(0, 0) = jacobian(1, 1) * jacobian(1, 1);
    jacobian(0, 1) = 0;
    jacobian(1, 0) = line_results.par(1) * rotMat(0, 1) * jacobian(0, 0);
    line_results.par(1) = line_results.par(1) * jacobian(1, 1);
    line_results.par(0) = (rotMat(0, 1) + line_results.par(0) * rotMat(0, 0)) * jacobian(1, 1);
    line_results.cov = jacobian * line_results.cov * jacobian.transpose();

    // compute chi2
    line_results.chi2 = 0;
    for (u_int i = 0; i < n; i++) {
      line_results.chi2 += weights(i) * riemannFit::sqr(zInSZplane(i) - uVec(i));
      if (i > 0 && i < n - 1)
        line_results.chi2 += riemannFit::sqr(uVec(i - 1) / (sTotal(i) - sTotal(i - 1)) -
                                             uVec(i) * (sTotal(i + 1) - sTotal(i - 1)) /
                                                 ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))) +
                                             uVec(i + 1) / (sTotal(i + 1) - sTotal(i))) /
                             varBeta(i);
    }
  }

  /*!
    \brief Helix fit by three step:
    -fast pre-fit (see Fast_fit() for further info); \n
    -circle fit of the hits projected in the transverse plane by Broken Line algorithm (see BL_Circle_fit() for further info); \n
    -line fit of the hits projected on the (pre-fitted) cilinder surface by Broken Line algorithm (see BL_Line_fit() for further info); \n
    Points must be passed ordered (from inner to outer layer).
    
    \param hits Matrix3xNd hits coordinates in this form: \n
    |x1|x2|x3|...|xn| \n
    |y1|y2|y3|...|yn| \n
    |z1|z2|z3|...|zn|
    \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
    |(x1,x1)|(x2,x1)|(x3,x1)|(x4,x1)|.|(y1,x1)|(y2,x1)|(y3,x1)|(y4,x1)|.|(z1,x1)|(z2,x1)|(z3,x1)|(z4,x1)| \n
    |(x1,x2)|(x2,x2)|(x3,x2)|(x4,x2)|.|(y1,x2)|(y2,x2)|(y3,x2)|(y4,x2)|.|(z1,x2)|(z2,x2)|(z3,x2)|(z4,x2)| \n
    |(x1,x3)|(x2,x3)|(x3,x3)|(x4,x3)|.|(y1,x3)|(y2,x3)|(y3,x3)|(y4,x3)|.|(z1,x3)|(z2,x3)|(z3,x3)|(z4,x3)| \n
    |(x1,x4)|(x2,x4)|(x3,x4)|(x4,x4)|.|(y1,x4)|(y2,x4)|(y3,x4)|(y4,x4)|.|(z1,x4)|(z2,x4)|(z3,x4)|(z4,x4)| \n
    .       .       .       .       . .       .       .       .       . .       .       .       .       . \n
    |(x1,y1)|(x2,y1)|(x3,y1)|(x4,y1)|.|(y1,y1)|(y2,y1)|(y3,x1)|(y4,y1)|.|(z1,y1)|(z2,y1)|(z3,y1)|(z4,y1)| \n
    |(x1,y2)|(x2,y2)|(x3,y2)|(x4,y2)|.|(y1,y2)|(y2,y2)|(y3,x2)|(y4,y2)|.|(z1,y2)|(z2,y2)|(z3,y2)|(z4,y2)| \n
    |(x1,y3)|(x2,y3)|(x3,y3)|(x4,y3)|.|(y1,y3)|(y2,y3)|(y3,x3)|(y4,y3)|.|(z1,y3)|(z2,y3)|(z3,y3)|(z4,y3)| \n
    |(x1,y4)|(x2,y4)|(x3,y4)|(x4,y4)|.|(y1,y4)|(y2,y4)|(y3,x4)|(y4,y4)|.|(z1,y4)|(z2,y4)|(z3,y4)|(z4,y4)| \n
    .       .       .    .          . .       .       .       .       . .       .       .       .       . \n
    |(x1,z1)|(x2,z1)|(x3,z1)|(x4,z1)|.|(y1,z1)|(y2,z1)|(y3,z1)|(y4,z1)|.|(z1,z1)|(z2,z1)|(z3,z1)|(z4,z1)| \n
    |(x1,z2)|(x2,z2)|(x3,z2)|(x4,z2)|.|(y1,z2)|(y2,z2)|(y3,z2)|(y4,z2)|.|(z1,z2)|(z2,z2)|(z3,z2)|(z4,z2)| \n
    |(x1,z3)|(x2,z3)|(x3,z3)|(x4,z3)|.|(y1,z3)|(y2,z3)|(y3,z3)|(y4,z3)|.|(z1,z3)|(z2,z3)|(z3,z3)|(z4,z3)| \n
    |(x1,z4)|(x2,z4)|(x3,z4)|(x4,z4)|.|(y1,z4)|(y2,z4)|(y3,z4)|(y4,z4)|.|(z1,z4)|(z2,z4)|(z3,z4)|(z4,z4)|
    \param bField magnetic field in the center of the detector in Gev/cm/c, in order to perform the p_t calculation.
    
    \warning see BL_Circle_fit(), BL_Line_fit() and Fast_fit() warnings.
    
    \bug see BL_Circle_fit(), BL_Line_fit() and Fast_fit() bugs.
    
    \return (phi,Tip,p_t,cot(theta)),Zip), their covariance matrix and the chi2's of the circle and line fits.
  */

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::brokenline

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BrokenLine_h
