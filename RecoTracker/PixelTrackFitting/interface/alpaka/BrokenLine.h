#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BrokenLine_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BrokenLine_h

#include <cmath>
#include <limits>
#include <numbers>

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "FWCore/Utilities/interface/isFinite.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BandedSolve.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"

//#define BL_DEEPDEBUG

// The BrokenLine fit of the CA tracks: Blobel's broken line [B06] with Karimaki's circle parameters (phi, d, kappa)
// [K91]. The fit finds the offsets u_i of the trajectory from the pre-fit (fastFit) at the hits, with a kink of known
// variance at each interior hit. A root-free LDL^T band factorization (BandedSolve.h) solves the pentadiagonal normal
// matrices. The circle's matrix has a border column for the curvature ([B06] Eq. 8). The run-time switch
// fitCorrections (producer parameter useFitCorrections) adds the material and field maps, the energy loss ([PDG],
// [Bi88], [SP71]), curvilinear kink rows, the line fit on the fitted circle and the transport to the track's perigee.
// [B06] V. Blobel, NIM A566 (2006) 14; [B07] V. Blobel, Broken-line fits: The Manual (2007), Sec. 2.1; [BKM11] V.
// Blobel, C. Kleinwort, F. Meier, Comput. Phys. Commun. 182 (2011) 1760; [K12] C. Kleinwort, NIM A673 (2012) 107;
// [K91] V. Karimaki, NIM A305 (1991) 187; [PDG] Review of Particle Physics 2026, Sec. 34 (Passage of particles through
// matter), with Highland's formula in Eq. 34.16; [Bi88] H. Bichsel, Rev. Mod. Phys. 60 (1988) 663; [SP71] R. M.
// Sternheimer, R. F. Peierls, Phys. Rev. B3 (1971) 3681; [SF10] A. Strandlie, R. Fruhwirth, Rev. Mod. Phys. 82 (2010)
// 1419; [FS21] R. Fruhwirth, A. Strandlie, Pattern Recognition, Tracking and Vertex Reconstruction in Particle
// Detectors, Springer (2021); [Hi02] N. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM (2002).

namespace ALPAKA_ACCELERATOR_NAMESPACE::brokenline {

  using namespace cms::alpakatools;
  using namespace ::riemannFit;

  // the ionization column of a path (int rho_e dl and its two log-means), as the walk accumulates it
  using ElossColumn = blMaterialMap::ElossColumn;

  //!< Karimäki's parameters: (phi, d, k=1/R)
  /*!< covariance matrix: \n
    |cov(phi,phi)|cov( d ,phi)|cov( k ,phi)| \n
    |cov(phi, d )|cov( d , d )|cov( k , d )| \n
    |cov(phi, k )|cov( d , k )|cov( k , k )| \n
    as defined in Karimäki V., 1990, Effective circle fitting for particle trajectories, 
    Nucl. Instr. and Meth. A305 (1991) 187.
  */
  using karimaki_circle_fit = riemannFit::CircleFit;

  // The per-lane scratch: the fit's O(N) state (the prepared data, the band block shared by the two fits and the
  // helper vectors) lives in the lane's slice of the fit scratch buffer; scalars and O(1) objects stay on the stack.
  // Element e of a lane is at lane0[e*S] (LaneSlot), S the element stride of the buffers (riemannFit::stride), so a
  // warp reads consecutive lanes at consecutive addresses. BLLaneBuffers (BrokenLineFitKernels.h) allocates
  // kFitScratchDoubles<maxHitsOnTrackForFullFit> per lane. The blocks, in the order of FitScratchLayout:
  //   prepared data (7n): radii 2n | sTransverse n | sTotal n | zInSZplane n | varBeta n | matXX0 n
  //   band block (6n+1):  bandM 3n (lower-packed pentadiagonal) | bandBorder n (circle border column) | bandCorner 1 |
  //                       bandW n (M^-1 border) | bandMs n (solve column)
  //   helpers (10n+2):    pointsSZ 2n | zVec n | lineWeights n | lineRu n | lineU n | circleWeights n |
  //                       circleRu n+1 | circleU n+1 | circleInvCol0 n
  // The slots are reused, which holds only in the lane's call order: a stage overwrites only what no later stage reads.
  // With fitCorrections off the order is prepareBrokenLineData, lineFit, circleFit: circleFit overwrites zInSZplane
  // with the circle's residuals and projects varBeta in place, so lineFit, which reads both, comes first.
  // With fitCorrections on the order is prepareBrokenLineData, lineOnPreFitArcs, lineDipSteps (with the coarse-
  // coordinate rule then coarseCoordinateRule and, where it moves a hit, preFitGeometry and reprojectDipShift),
  // circleFit, abscissaeOnFittedCircle, lineFit. The accessors of the views below name the later quantities of a slot.
  // sTotal and zInSZplane keep their names: the pre-fit's, zInSZplane then the circle's residuals, both then the
  // abscissae and offsets on the fitted circle. lineRu holds the line's dip-angle steps (dipSteps) from lineDipSteps to
  // circleFit, which reads them before its right-hand side: in between coarseCoordinateRule writes circleRu,
  // preFitGeometry the geometry of the prepared data (radii, sTransverse, sTotal, zInSZplane), reprojectDipShift lineU,
  // and circleFit before its right-hand side zInSZplane, matXX0, pointsSZ row 0, zVec and circleWeights. Among the slots
  // read up to the end of the fit are pointsSZ row 1 (the kink covariance c_g of each gap), circleWeights, bandW,
  // circleInvCol0 and circleInvCol1.
  template <int n>
  struct FitScratchLayout {
    // the prepared data
    static constexpr int radii = 0;
    static constexpr int sTransverse = radii + 2 * n;
    static constexpr int sTotal = sTransverse + n;
    static constexpr int zInSZplane = sTotal + n;
    static constexpr int varBeta = zInSZplane + n;
    static constexpr int matXX0 = varBeta + n;
    // the band block
    static constexpr int bandM = matXX0 + n;
    static constexpr int bandBorder = bandM + 3 * n;
    static constexpr int bandCorner = bandBorder + n;
    static constexpr int bandW = bandCorner + 1;
    static constexpr int bandMs = bandW + n;
    // the helpers
    static constexpr int pointsSZ = bandMs + n;
    static constexpr int zVec = pointsSZ + 2 * n;
    static constexpr int lineWeights = zVec + n;
    static constexpr int lineRu = lineWeights + n;
    static constexpr int lineU = lineRu + n;
    static constexpr int circleWeights = lineU + n;
    static constexpr int circleRu = circleWeights + n;
    static constexpr int circleU = circleRu + n + 1;
    static constexpr int circleInvCol0 = circleU + n + 1;
    static constexpr int total = circleInvCol0 + n;
  };

  // total fit doubles per lane
  template <int n>
  constexpr int kFitScratchDoubles = FitScratchLayout<n>::total;

  // One lane's slot in an [element][lane] buffer of element stride S: element e of the lane is at lane0[e*S].
  template <uint32_t S>
  struct LaneSlot {
    double* lane0;
    ALPAKA_FN_HOST_ACC LaneSlot(double* base, uint32_t lane) : lane0(base + lane) {}
    ALPAKA_FN_HOST_ACC double* element(int e) const { return lane0 + std::size_t(e) * S; }
  };

  // the scratch maps of the per-lane layout above, S at compile time (S = 1 is contiguous storage)
  template <int n, uint32_t S>
  using MapVecNd = Eigen::Map<riemannFit::VectorNd<n>, Eigen::Unaligned, Eigen::InnerStride<S>>;
  template <int n, uint32_t S>
  using MapVecNplus1d = Eigen::Map<riemannFit::VectorNplusONEd<n>, Eigen::Unaligned, Eigen::InnerStride<S>>;
  template <int n, uint32_t S>
  using Map2xNd = Eigen::Map<riemannFit::Matrix2xNd<n>, Eigen::Unaligned, Eigen::Stride<2 * S, S>>;

  // The precision test of a solve (fitCorrections): a pivot d_j of the LDL^T of a symmetric M has no significant digit
  // when d_j/M_jj <= gamma_k = k u/(1 - k u), with u the unit roundoff ([Hi02] Lemma 8.4, Thm. 9.3). k =
  // kPivotRoundings bounds the roundings on a pivot for at most kPivotMaxNodes hits. A non-positive M_jj also fails.
  constexpr int kPivotMaxNodes = 10;
  constexpr int kPivotRoundings = (kPivotMaxNodes - 1) + 2 * 4 + (5 + 3 + 5) + kPivotMaxNodes;
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool significantPivot(double pivot, double diagonal) {
    constexpr double u = 0.5 * std::numeric_limits<double>::epsilon();
    constexpr double tau = kPivotRoundings * u / (1. - kPivotRoundings * u);
    const double ratio = (diagonal > 0.) ? pivot / diagonal : -1.;
    return !edm::isNotFinite(ratio) && ratio > tau;
  }

  // The same test for each pivot D_i = band.M(i, i) of a band factorization, with M_ii saved in diag(i).
  template <typename TBand, typename TColumn>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool significantBandPivots(const TBand& band, const TColumn& diag, int n) {
    bool ok = true;
    for (int i = 0; i < n; ++i) {
      ok = ok && significantPivot(band.M(i, i), diag(i));
    }
    return ok;
  }

  // The band block of a lane's scratch, for the line and the circle fit: the symmetric pentadiagonal M, lower-packed
  // for BandedSolve.h, the circle's border column and corner, w = M^-1 border and the solve column ms.
  template <int n, uint32_t S>
  class BandBlock {
  public:
    MapVecNd<n, S> border;
    MapVecNd<n, S> w;
    MapVecNd<n, S> ms;
    double& corner;

    ALPAKA_FN_ACC explicit BandBlock(LaneSlot<S> const& s)
        : border(s.element(FitScratchLayout<n>::bandBorder)),
          w(s.element(FitScratchLayout<n>::bandW)),
          ms(s.element(FitScratchLayout<n>::bandMs)),
          corner(*s.element(FitScratchLayout<n>::bandCorner)),
          m_(s.element(FitScratchLayout<n>::bandM)) {}

    // M(i, j), j = i-2 .. i: row i of the packed band, offset i - j (32-bit index arithmetic)
    template <typename I>
    ALPAKA_FN_ACC double& M(I i, I j) const {
      static_assert(sizeof(I) == sizeof(int), "32-bit index");
      return m_[bandedSolve::packedBandIndex<2>(i, i - j, I(S))];
    }
    // M = L D L^T in place (bandedSolve::bandFactor<2>)
    ALPAKA_FN_ACC void factor() const { bandedSolve::bandFactor<2>(m_, n, int(S)); }
    // x = M^-1 x with the factors (bandedSolve::bandSolveInPlace<2>), x a column of the lane's scratch
    template <typename X>
    ALPAKA_FN_ACC void solveInPlace(X& x) const {
      static_assert(int(X::InnerStrideAtCompileTime) == int(S), "x is a column of the lane's scratch");
      bandedSolve::bandSolveInPlace<2>(m_, x.data(), n, int(S));
    }
    // x = M^-1 e_k, the column k of M^-1
    template <int k, typename X>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void solveUnitColumn(X& x) const {
      for (int i = 0; i < n; i++) {
        x(i) = 0.;
      }
      x(k) = 1.;
      solveInPlace(x);
    }
    // factor(). With testPivots, it saves the diagonal in diag and sets failed for a pivot without a significant digit.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void factorTestingPivots(MapVecNd<n, S>& diag, bool testPivots, bool& failed) const {
      if (testPivots) {
        for (int i = 0; i < n; i++) {
          diag(i) = M(i, i);
        }
      }
      factor();
      if (testPivots && !significantBandPivots(*this, diag, n)) {
        failed = true;
      }
    }

  private:
    double* m_;
  };

  // The prepared data over a lane's [element][lane] scratch block (stride S).
  template <int n, uint32_t S>
  struct PreparedBrokenLineDataMap {
    static constexpr int kN = n;
    int qCharge;
    Map2xNd<n, S> radii;
    MapVecNd<n, S> sTransverse;
    MapVecNd<n, S> sTotal;
    MapVecNd<n, S> zInSZplane;
    MapVecNd<n, S> varBeta;
    MapVecNd<n, S> matXX0;
    double innerXX0;
    // fitCorrections only: the beam-line segment as two thin scatterers (segmentXX0Moments)
    double innerD1;
    double innerW1;
    // fitCorrections only: the factor of every scattering variance (momentumUncertaintyFactor)
    double momentumFactor;
    // fitCorrections only: innerVariance = H(innerXX0), the 3-D angle variance of the beam-line segment. node0True is
    // the variance of gap 0's departure share at hit 0, node0Cov its covariance with the kink of node 1.
    double innerVariance;
    double node0True;
    double node0Cov;
    // fitCorrections only: the fit cannot be done. The lane then publishes the track as failed.
    bool failed = false;
    ALPAKA_FN_ACC explicit PreparedBrokenLineDataMap(LaneSlot<S> const& s)
        : radii(s.element(L::radii)),
          sTransverse(s.element(L::sTransverse)),
          sTotal(s.element(L::sTotal)),
          zInSZplane(s.element(L::zInSZplane)),
          varBeta(s.element(L::varBeta)),
          matXX0(s.element(L::matXX0)) {}
    // matXX0, fitCorrections: cos A_g of the angle of each gap at the pre-fit circle's centre (circleFit)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& gapCos() { return matXX0; }
    // matXX0, fitCorrections: dS_i/dphi of the abscissae on the fitted circle (lineFitOnFittedCircle)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& dsdPhi() { return matXX0; }

  private:
    using L = FitScratchLayout<n>;
  };

  // Per-lane fit workspace over the lane's scratch: the band block and the helper vectors.
  template <int n, uint32_t S>
  struct FitWorkspaceMap {
    static constexpr int kN = n;
    BandBlock<n, S> band;
    Map2xNd<n, S> pointsSZ;
    MapVecNd<n, S> zVec;
    MapVecNd<n, S> lineWeights;
    MapVecNd<n, S> lineRu;
    MapVecNd<n, S> lineU;
    MapVecNd<n, S> circleWeights;
    MapVecNplus1d<n, S> circleRu;
    MapVecNplus1d<n, S> circleU;
    MapVecNd<n, S> circleInvCol0;
    ALPAKA_FN_ACC explicit FitWorkspaceMap(LaneSlot<S> const& s)
        : band(s),
          pointsSZ(s.element(L::pointsSZ)),
          zVec(s.element(L::zVec)),
          lineWeights(s.element(L::lineWeights)),
          lineRu(s.element(L::lineRu)),
          lineU(s.element(L::lineU)),
          circleWeights(s.element(L::circleWeights)),
          circleRu(s.element(L::circleRu)),
          circleU(s.element(L::circleU)),
          circleInvCol0(s.element(L::circleInvCol0)) {}
    // zVec: the ionization loss [GeV] over each gap, slot n-1 the cumulative loss (prepareBrokenLineData)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& gapEloss() { return zVec; }
    // zVec: the deterministic offset of the circle's residuals, written over the loss (circleFit)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& elossOffset() { return zVec; }
    // zVec: dS_i/drho of the abscissae on the fitted circle (lineFitOnFittedCircle)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& dsdRho() { return zVec; }
    // pointsSZ row 0: sin A_g of the angle of each gap at the pre-fit circle's centre (circleFit)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto gapSin() { return pointsSZ.row(0); }
    // pointsSZ row 0: dS_i/dd of the abscissae on the fitted circle (lineFitOnFittedCircle)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto dsdD() { return pointsSZ.row(0); }
    // lineU: the z' shift c dz_i of the dip-angle row at each node (prepareBrokenLineData), until lineFit's solution
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& dipShift() { return lineU; }
    // lineRu, fitCorrections: the dip-angle step S_g of each gap (lineDipSteps), until circleFit's right-hand side
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& dipSteps() { return lineRu; }
    // lineRu: the line's M^-1 e_0 (lineFit)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& lineInvCol0() { return lineRu; }
    // bandMs: the line's M^-1 e_1 (lineFit)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNd<n, S>& lineInvCol1() { return band.ms; }
    // circleRu: the circle's M^-1 e_1 (circleFit)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE MapVecNplus1d<n, S>& circleInvCol1() { return circleRu; }

  private:
    using L = FitScratchLayout<n>;
  };

  // One lane's scratch: its prepared data and its workspace, built from the scratch buffer and the lane index.
  template <int n, uint32_t S>
  struct LaneScratch {
    PreparedBrokenLineDataMap<n, S> prepared;
    FitWorkspaceMap<n, S> workspace;
    ALPAKA_FN_ACC LaneScratch(double* scratch, uint32_t lane) : LaneScratch(LaneSlot<S>(scratch, lane)) {}
    ALPAKA_FN_ACC explicit LaneScratch(LaneSlot<S> const& s) : prepared(s), workspace(s) {}
  };

  // The model switch and, read only with fitCorrections, the material map, the (Bz, Br) field map and bFieldOrigin,
  // the field [GeV/cm] to which the field map is normalized.
  struct FitContext {
    bool fitCorrections;
    const blMaterialMap::Map* rhoMap;
    const blBFieldMap::Map* bMap;
    double bFieldOrigin;
  };

  // tan(lambda) cos(alpha) = -(cx y - cy x)/(slopeDen r) for the B_r term of the bending field (BLBFieldMap.h), with
  // (cx, cy) the pre-fit circle's centre and slopeDen = (ds_T/dz) |R|. It does not depend on the charge.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double tanLambdaCosAlpha(
      double cx, double cy, double slopeDen, double x, double y, double r) {
    const double den = slopeDen * r;
    return (den != 0.) ? -(cx * y - cy * x) / den : 0.;
  }

  // What segmentWalk returns for a segment: its length L, its moments about the arrival end (d the distance to it)
  // W = int rho dl, S1 = int rho d dl and S2 = int rho d^2 dl, and its ionization column.
  struct SegmentMoments {
    double L = 0., W = 0., S1 = 0., S2 = 0.;
    ElossColumn col;
  };

  // Narrows [tLo, tHi] to the t at which x0 + t (x1 - x0) is in [lo, hi]. A chord with both ends in [lo, hi] keeps it.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void clipChord(double x0, double x1, double lo, double hi, double& tLo, double& tHi) {
    if (x0 >= lo && x0 <= hi && x1 >= lo && x1 <= hi) {
      return;
    }
    const double dx = x1 - x0;
    if (dx == 0.) {
      tHi = tLo;  // the chord is outside [lo, hi]
      return;
    }
    const double ta = (lo - x0) / dx, tb = (hi - x0) / dx;
    const double tEnter = ta < tb ? ta : tb, tExit = ta < tb ? tb : ta;
    tLo = tEnter > tLo ? tEnter : tLo;
    tHi = tExit < tHi ? tExit : tHi;
  }

  // Exact cell walk of the material map along the straight (r, z) chord (r0, z0)->(r1, z1). Each piece between two
  // cell boundaries adds its cell's density times its length to W, S1, S2 and the column. path3D > 0 rescales to the
  // track's 3-D path, k = path3D/L: the moments scale by k, k^2, k^3 and the column by k.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE SegmentMoments segmentWalk(
      const TAcc& acc, const blMaterialMap::Map* map, double r0, double z0, double r1, double z1, double path3D) {
    SegmentMoments moments;
    auto& [L, W, S1, S2, col] = moments;
    const double dr = r1 - r0, dz = z1 - z0;
    L = alpaka::math::sqrt(acc, dr * dr + dz * dz);
    if (!(L > 0.)) {
      return moments;
    }
    // The walk covers the part [tLo, tHi] of the chord inside the map's box: the cells outside it hold no material.
    double tLo = 0., tHi = 1.;
    clipChord(r0, r1, 0., double(blMaterialMap::kNR) * double(blMaterialMap::kDR), tLo, tHi);
    clipChord(z0, z1, -double(blMaterialMap::kZMAX), double(blMaterialMap::kZMAX), tLo, tHi);
    const double rStart = r0 + tLo * dr, zStart = z0 + tLo * dz;
    // The next boundary crossing of each coordinate in the chord parameter t, and the spacing of its crossings.
    double tR = 2., dtR = 1.;
    if (dr != 0.) {
      const double kNext = alpaka::math::floor(acc, rStart / double(blMaterialMap::kDR)) + (dr > 0. ? 1. : 0.);
      tR = (kNext * double(blMaterialMap::kDR) - r0) / dr;
      dtR = double(blMaterialMap::kDR) / alpaka::math::abs(acc, dr);
    }
    double tZ = 2., dtZ = 1.;
    if (dz != 0.) {
      const double kNext = alpaka::math::floor(acc, zStart / double(blMaterialMap::kDZ)) + (dz > 0. ? 1. : 0.);
      tZ = (kNext * double(blMaterialMap::kDZ) - z0) / dz;
      dtZ = double(blMaterialMap::kDZ) / alpaka::math::abs(acc, dz);
    }
    // kMaxCells bounds the pieces of a chord in the map's box and ends the loop for a non-finite input.
    constexpr int kMaxCells = blMaterialMap::kNR + blMaterialMap::kNZ + 1;
    double t = tLo;
    for (int cell = 0; cell < kMaxCells && t < tHi; ++cell) {
      double tn = tR < tZ ? tR : tZ;
      if (tn > tHi) {
        tn = tHi;
      }
      if (tn > t) {
        const double tm = 0.5 * (t + tn);  // midpoint: inside the cell whatever the boundary rounding
        const float rm = float(r0 + tm * dr), zm = float(z0 + tm * dz);
        const int cellIndex = blMaterialMap::cellAt(rm, zm);
        const double rho = blMaterialMap::rhoOfCell(*map, cellIndex);
        const blMaterialMap::DeDx dedx = blMaterialMap::dedxOfCell(*map, cellIndex);
        if (rho > 0.f) {
          const double a = 1. - t, c = 1. - tn;
          W += rho * (a - c) * L;
          S1 += rho * (a * a - c * c) * 0.5 * L * L;
          S2 += rho * (a * a * a - c * c * c) * (1. / 3.) * L * L * L;
          // the same sub-interval, the same cell: the ionization column of the piece
          const double we = double(dedx.rhoE) * (a - c) * L;
          col.e += we;
          col.eLnI += we * double(dedx.lnI);
          col.eLnRho += we * double(dedx.lnRhoE);
        }
      }
      t = tn;
      if (tR <= t) {
        tR += dtR;
      }
      if (tZ <= t) {
        tZ += dtZ;
      }
    }
    if (path3D > 0.) {
      const double k = path3D / L;
      W *= k;
      S1 *= k * k;
      S2 *= k * k * k;
      col = col * k;  // first moments only: the column is charged per unit path
      L = path3D;
    }
    return moments;
  }

  // Two thin scatterers equivalent to the material of the beam-line segment from (r = 0, z0) to (r1, z1) ([BKM11],
  // [K12]). W, S1, S2 are the walk's moments about the (r1, z1) end. One scatterer is at path distance d1 = S2/S1
  // before that end with the fraction w1 = S1^2/(S2 W) of the variance, the other is at the end. Returns W.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double segmentXX0Moments(const TAcc& acc,
                                                          const blMaterialMap::Map* map,
                                                          double z0,
                                                          double r1,
                                                          double z1,
                                                          double& d1,
                                                          double& w1,
                                                          double path3D,
                                                          ElossColumn& col) {
    const auto [L, W, S1, S2, walked] = segmentWalk(acc, map, 0., z0, r1, z1, path3D);
    col = walked;
    d1 = 0.;
    w1 = 0.;
    if (W > 0. && S1 > 0. && S2 > 0.) {
      d1 = S2 / S1;
      w1 = S1 * S1 / (S2 * W);
    }
    return W;
  }

  // The beam-line segment from the closest approach to the beam line, (r, z) = (0, zPca) on the pre-fit line, to hit 0.
  // slope = dz/ds_T of the pre-fit line, sT0 = sTransverse(0), path3D the 3-D path of the segment.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void beamlineSegment(
      const TAcc& acc, double zHit0, double slope, double sT0, double& zPca, double& path3D) {
    zPca = zHit0 - slope * sT0;
    path3D = alpaka::math::abs(acc, sT0) * alpaka::math::sqrt(acc, 1. + slope * slope);
  }

  // 1 + slope^2 = 1/cos^2(lambda): the factor from the variance of a 3-D angle to that of its transverse projection.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double kinkProjection(double slope) { return 1. + riemannFit::sqr(slope); }

  // The pre-fit's total momentum [GeV]: bField [GeV/cm] times the radius, over sin(theta) (slope = cot(theta)).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double totalMomentum(const TAcc& acc, double bField, double radius, double slope) {
    return alpaka::math::sqrt(acc, riemannFit::sqr(bField * radius) * kinkProjection(slope));
  }

  // The number of sub-steps of a field integral on an arc of the circle of centre c and radius R, from the unit radius
  // vector n0 turned by sweep: n = 1 + floor(max(V_r kInvDR, |z1 - z0| kInvDZ)), V_r the total variation of r on the
  // arc. kMaxFieldSteps bounds n for an arc of less than a turn. A longer arc or a non-finite input gets that bound.
  constexpr int kMaxFieldSteps =
      1 + (3 * (blBFieldMap::kNR - 1) > blBFieldMap::kNZ - 1 ? 3 * (blBFieldMap::kNR - 1) : blBFieldMap::kNZ - 1);

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int latticeSteps(
      const TAcc& acc, double cx, double cy, double radius, double n0x, double n0y, double sweep, double z0, double z1) {
    constexpr double kTwoPi = 2. * std::numbers::pi;
    if (edm::isNotFinite(sweep) || !(alpaka::math::abs(acc, sweep) < kTwoPi)) {
      return kMaxFieldSteps;
    }
    auto cross = [](double ax, double ay, double bx, double by) { return ax * by - ay * bx; };
    const double cs = alpaka::math::cos(acc, sweep), sn = alpaka::math::sin(acc, sweep);
    const double n1x = cs * n0x - sn * n0y, n1y = sn * n0x + cs * n0y;
    const double sg = sweep < 0. ? -1. : 1.;
    const bool upToHalfTurn = alpaka::math::abs(acc, sweep) <= 0.5 * kTwoPi;
    // v on the arc: between n0 and n1 in the sense of the arc up to half a turn, off the complementary arc beyond
    auto inside = [&](double vx, double vy) {
      if (upToHalfTurn) {
        return sg * cross(n0x, n0y, vx, vy) > 0. && sg * cross(vx, vy, n1x, n1y) > 0.;
      }
      return !(sg * cross(n1x, n1y, vx, vy) >= 0. && sg * cross(vx, vy, n0x, n0y) >= 0.);
    };
    const double sx = cx + radius * n0x, sy = cy + radius * n0y;
    const double ex = cx + radius * n1x, ey = cy + radius * n1y;
    const double rStart = blBFieldMap::clampR(alpaka::math::sqrt(acc, sx * sx + sy * sy));
    const double rEnd = blBFieldMap::clampR(alpaka::math::sqrt(acc, ex * ex + ey * ey));
    double variation = 0.;
    const double cNorm = alpaka::math::sqrt(acc, cx * cx + cy * cy);
    if (cNorm > 0.) {
      const double ax = cx / cNorm, ay = cy / cNorm;  // the direction of the apex
      const double rApex = blBFieldMap::clampR(cNorm + radius),
                   rPerigee = blBFieldMap::clampR(alpaka::math::abs(acc, cNorm - radius));
      const bool hasApex = inside(ax, ay), hasPerigee = inside(-ax, -ay);
      // the extreme met first lies within half a turn of n0 in the sense of the arc
      const bool apexFirst = sg * cross(n0x, n0y, ax, ay) > 0.;
      double r = rStart;
      if (hasApex && hasPerigee) {
        const double first = apexFirst ? rApex : rPerigee, second = apexFirst ? rPerigee : rApex;
        variation = alpaka::math::abs(acc, first - r) + alpaka::math::abs(acc, second - first);
        r = second;
      } else if (hasApex) {
        variation = alpaka::math::abs(acc, rApex - r);
        r = rApex;
      } else if (hasPerigee) {
        variation = alpaka::math::abs(acc, rPerigee - r);
        r = rPerigee;
      }
      variation += alpaka::math::abs(acc, rEnd - r);
    } else {
      variation = alpaka::math::abs(acc, rEnd - rStart);
    }
    const double xR = variation * blBFieldMap::kInvDR;
    const double xZ = alpaka::math::abs(acc, blBFieldMap::clampZ(z1) - blBFieldMap::clampZ(z0)) * blBFieldMap::kInvDZ;
    const double x = xR < xZ ? xZ : xR;
    if (edm::isNotFinite(x) || !(x < double(kMaxFieldSteps - 1))) {
      return kMaxFieldSteps;
    }
    return 1 + int(x);
  }

  // 1 - cos x from cos x and sin x, without the cancellation of 1 - cos x at small x.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double oneMinusCos(double c, double s) {
    return (c >= 0.) ? s * s / (1. + c) : 1. - c;
  }

  // tan(x/2) from cos x and sin x, in the form that does not cancel for the sign of cos x.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double tanHalf(double c, double s) { return (c >= 0.) ? s / (1. + c) : (1. - c) / s; }

  // The exact step of (u, u', int u ds) over an arc h on a circle of radius R, for u'' + u/R^2 = -f with f linear from
  // f0 to f1. With x = h/R, the homogeneous part rotates (u, R u') by x. The source term subtracts w0 f0 + w1 f1 from
  // u, v0 f0 + v1 f1 from u' and iw0 f0 + iw1 f1 from int u ds.
  struct CurvilinearStep {
    double c = 1., s = 0., omc = 0.;  // cos x, sin x, 1 - cos x
    double w0 = 0., w1 = 0., v0 = 0., v1 = 0., iw0 = 0., iw1 = 0.;
    double radius = 0.;
    // integral: also compute the weights of int u ds
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC CurvilinearStep(const TAcc& acc, double h, double r, bool integral = true) : radius(r) {
      const double x = h / r;
      if (x == 0.) {
        return;
      }
      c = alpaka::math::cos(acc, x);
      s = alpaka::math::sin(acc, x);
      omc = oneMinusCos(c, s);
      w1 = r * r * (1. - s / x);
      w0 = r * r * omc - w1;
      v1 = r * omc / x;
      v0 = r * s - v1;
      if (!integral) {
        return;
      }
      iw1 = r * r * r * (0.5 * x - omc / x);
      iw0 = r * r * r * (x - s) - iw1;
    }
    // advances (u, up) by this step
    ALPAKA_FN_ACC void apply(double& u, double& up, double f0, double f1) const {
      const double uNew = c * u + radius * s * up - (w0 * f0 + w1 * f1);
      const double upNew = -s * u / radius + c * up - (v0 * f0 + v1 * f1);
      u = uNew;
      up = upNew;
    }
    // advances (u, up) and the running integral of u by this step
    ALPAKA_FN_ACC void apply(double& u, double& up, double& integral, double f0, double f1) const {
      const double uNew = c * u + radius * s * up - (w0 * f0 + w1 * f1);
      const double upNew = -s * u / radius + c * up - (v0 * f0 + v1 * f1);
      integral += radius * s * u + radius * radius * omc * up - (iw0 * f0 + iw1 * f1);
      u = uNew;
      up = upNew;
    }
  };

  // The off model's scattering variance for radLen radiation lengths: Highland's width with the factor 0.7 and
  // beta = 1, at p^2 = pT^2 (1 + slope^2) with pT capped at 20 GeV.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double multScatt(
      const TAcc& acc, const double radLen, const double bField, const double radius, double slope) {
    auto pt2 = alpaka::math::min(acc, 20., bField * radius);
    pt2 *= pt2;
    constexpr double fact = 0.7 * riemannFit::sqr(13.6 / 1000.);
    return fact / (pt2 * (1. + riemannFit::sqr(slope))) * radLen * riemannFit::sqr(1. + 0.038 * log(radLen));
  }

  // The variance of the projected scattering angle for x = X/X0 radiation lengths, Highland's formula ([PDG] Eq. 34.16)
  //     H(x) = (13.6 MeV/(beta p))^2 x (1 + 0.038 ln(x/beta^2))^2,
  // for a pion of momentum p, p2 = p^2 [GeV^2]. Zero for x <= 0.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double highlandCombined(const TAcc& acc, double x, double p2) {
    if (!(x > 0.)) {
      return 0.;
    }
    constexpr double kHighland = 13.6 / 1000.;  // GeV
    constexpr double kHighlandLog = 0.038;
    constexpr double m = blMaterialMap::elossMedium::kPionMass;
    const double p2b2 = p2 * p2 / (p2 + m * m);
    const double xLog = x * (p2 + m * m) / p2;
    const double log1 = 1. + kHighlandLog * alpaka::math::log(acc, xLog);
    return kHighland * kHighland / p2b2 * x * log1 * log1;
  }

  // The factor of every scattering variance for the pre-fit's momentum uncertainty (fitCorrections),
  // 1 + s^2 (p^2 + m^2 beta^2 (5 + 3 p^2 s^2)), the "exact formula" of MultipleScatteringUpdator.cc in
  // TrackingTools/MaterialEffects. m is the pion mass and s^2 = var(1/p) = cos^2(lambda) var(kappa)/B^2, with
  // var(kappa) of the circle through hits 0, mId and n-1. A degenerate triangle returns 1.
  template <alpaka::concepts::Acc TAcc, typename M3xN, typename M6xN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double momentumUncertaintyFactor(const TAcc& acc,
                                                                  const M3xN& hits,
                                                                  const M6xN& hits_ge,
                                                                  int mId,
                                                                  int n,
                                                                  double slope,
                                                                  double bField,
                                                                  double radius) {
    const int idx[3] = {0, mId, n - 1};
    const double p0x = hits(0, 0), p0y = hits(1, 0);
    const double p1x = hits(0, mId), p1y = hits(1, mId);
    const double p2x = hits(0, n - 1), p2y = hits(1, n - 1);
    const double ax = p1x - p0x, ay = p1y - p0y, bx = p2x - p1x, by = p2y - p1y, cx = p2x - p0x, cy = p2y - p0y;
    const double s = ax * cy - ay * cx;
    const double la2 = ax * ax + ay * ay, lb2 = bx * bx + by * by, lc2 = cx * cx + cy * cy;
    if (!(s != 0.) || !(la2 > 0.) || !(lb2 > 0.) || !(lc2 > 0.)) {
      return 1.;
    }
    const double kappa = 2. * s / alpaka::math::sqrt(acc, la2 * lb2 * lc2);
    // the gradient of kappa with respect to each of the three points
    const double g[3][2] = {
        {kappa * ((p1y - p2y) / s + ax / la2 + cx / lc2), kappa * (-(p1x - p2x) / s + ay / la2 + cy / lc2)},
        {kappa * ((p2y - p0y) / s - ax / la2 + bx / lb2), kappa * (-(p2x - p0x) / s - ay / la2 + by / lb2)},
        {kappa * ((p0y - p1y) / s - bx / lb2 - cx / lc2), kappa * (-(p0x - p1x) / s - by / lb2 - cy / lc2)}};
    double varK = 0.;
    for (int j = 0; j < 3; ++j) {
      const double cxx = hits_ge.col(idx[j])[0], cxy = hits_ge.col(idx[j])[1], cyy = hits_ge.col(idx[j])[2];
      varK += g[j][0] * (cxx * g[j][0] + cxy * g[j][1]) + g[j][1] * (cxy * g[j][0] + cyy * g[j][1]);
    }
    constexpr double m = blMaterialMap::elossMedium::kPionMass;
    const double cos2 = 1. / (1. + slope * slope);
    const double s2 = cos2 * varK / (bField * bField);  // var(1/p)
    const double p2 = bField * radius * bField * radius * (1. + slope * slope);
    const double beta2 = p2 / (p2 + m * m);
    return 1. + s2 * (p2 + m * m * beta2 * (5. + 3. * p2 * s2));
  }

  // out = jac * mid * jac^T. mid and out may be the same matrix.
  template <typename TJac, typename TMid, typename TOut>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void similarity(const TJac& jac, const TMid& mid, TOut& out) {
    constexpr int rows = TJac::RowsAtCompileTime, cols = TJac::ColsAtCompileTime;
    Eigen::Matrix<double, rows, cols> jacMid;
    jacMid.noalias() = jac * mid;
    Eigen::Matrix<double, rows, rows, Eigen::RowMajor> result;
    result.noalias() = jacMid * jac.transpose();
    out = result;
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
    \brief Changes the Karimäki parameters (phi, d, rho) of a circle under a translation of the coordinate system, so
   *       that the old origin is at (x0, y0) in the new system ([K91]).

    \param in the circle in the old coordinate system: (phi, d, rho).
    \param x0 x coordinate of the translation vector.
    \param y0 y coordinate of the translation vector.
    \param jac the derivative d(out)/d(phi, d, rho) at a fixed translation vector.
    \param out when given, the circle in the new coordinate system. It may be the input itself.
    \param jacXY when given, the derivative d(out)/d(x0, y0).
  */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void translateKarimaki(const TAcc& acc,
                                                        const riemannFit::Vector3d& in,
                                                        double x0,
                                                        double y0,
                                                        riemannFit::Matrix3d& jac,
                                                        riemannFit::Vector3d* out = nullptr,
                                                        Eigen::Matrix<double, 3, 2>* jacXY = nullptr) {
    const double phi = in(0), d = in(1), rho = in(2);
    const double sn = alpaka::math::sin(acc, phi), cs = alpaka::math::cos(acc, phi);
    const double along = x0 * cs + y0 * sn;
    const double across = x0 * sn - y0 * cs + d;
    const double u = 1. + rho * d;
    const double cc = -rho * y0 + u * cs;
    const double bb = rho * x0 + u * sn;
    const double aa = 2. * across + rho * (across * across + along * along);
    const double uu = alpaka::math::sqrt(acc, 1. + rho * aa);
    const double xi = 1. / (bb * bb + cc * cc);
    const double v = 1. + rho * across;
    const double lambda = 0.5 * aa / ((1. + uu) * (1. + uu) * uu);
    const double mu = 1. / (uu * (1. + uu)) + rho * lambda;
    const double zeta = across * across + along * along;
    if (out != nullptr) {
      *out << alpaka::math::atan2(acc, bb, cc), aa / (1. + uu), rho;
    }
    jac << xi * u * v, -xi * rho * rho * along, xi * along, 2. * mu * u * along, 2. * mu * v, mu * zeta - lambda * aa,
        0., 0., 1.;
    if (jacXY != nullptr) {
      const double dddA = 1. / (1. + uu) - rho * aa / (2. * uu * (1. + uu) * (1. + uu));
      *jacXY << rho * cc * xi, rho * bb * xi, dddA * (2. * sn + 2. * rho * (across * sn + along * cs)),
          dddA * (-2. * cs + 2. * rho * (-across * cs + along * sn)), 0., 0.;
    }
  }

  // The kink row of interior node i of the curvilinear broken line (fitCorrections), with A and B the angles at the
  // pre-fit circle's centre (radius R) of the gaps before and after node i ([BKM11] Eq. 7, exact circle Jacobians):
  //     vartheta_i = u_{i-1}/(R sin A) - u_i (cot A + cot B)/R + u_{i+1}/(R sin B)
  //                  + Delta-kappa R (tan(A/2) + tan(B/2)).
  struct CurvilinearKinkRow {
    double before, at, after, dKappa;
    // the coefficient of u_{i-1+a}, a = 0, 1, 2
    ALPAKA_FN_ACC double u(int a) const { return a == 0 ? before : (a == 1 ? at : after); }
  };
  ALPAKA_FN_ACC ALPAKA_FN_INLINE CurvilinearKinkRow
  curvilinearKinkRow(double radius, double cA, double sA, double cB, double sB) {
    return CurvilinearKinkRow{1. / (radius * sA),
                              -(cA / sA + cB / sB) / radius,
                              1. / (radius * sB),
                              radius * (tanHalf(cA, sA) + tanHalf(cB, sB))};
  }

  // The straight kink row of interior node i at the abscissae s ([B06] Eq. 7).
  struct StraightKinkRow {
    double ra, rb, rc;
  };
  template <typename VS>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE StraightKinkRow straightKinkRow(const VS& s, int i) {
    const double a = s(i) - s(i - 1), b = s(i + 1) - s(i);
    return StraightKinkRow{1. / a, -(a + b) / (a * b), 1. / b};
  }

  // The dip-angle row (fitCorrections). The radial field turns the dip angle, dlambda/ds = -(q/p) B_r sin(alpha). The
  // z displacement dz = int sec^2(lambda_0) delta-lambda ds_T from the pre-fit PCA goes to dipShift as
  // cos(lambda) dz_i. lineOnPreFitArcs, coarseCoordinateRule and abscissaeOnFittedCircle subtract it from z'.
  template <alpaka::concepts::Acc TAcc, typename M3xN, typename V4, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void radialFieldDipShift(const TAcc& acc,
                                                          const M3xN& hits,
                                                          const V4& fast_fit,
                                                          const double bField,
                                                          const double slope,
                                                          const riemannFit::Matrix2d& rotMat,
                                                          const FitContext& ctx,
                                                          TData& results,
                                                          TWs& fitWs) {
    const double cx = fast_fit(0), cy = fast_fit(1), rad = fast_fit(2);
    const double sec2 = kinkProjection(slope);
    const double pTot = bField * rad * alpaka::math::sqrt(acc, sec2);  // bField = the effective bending field
    const double qbp = double(results.qCharge) / pTot;
    // dlambda per unit transverse arc at a point of the reference circle
    auto lambdaRate = [&](double x, double y, double z) {
      const double r = alpaka::math::sqrt(acc, x * x + y * y);
      if (!(r > 0.)) {
        return 0.;
      }
      const double brNorm = blBFieldMap::brAt(*ctx.bMap, r, z);
      const double sinAlpha = -double(results.qCharge) * (x * (x - cx) + y * (y - cy)) / (rad * r);
      return -qbp * (ctx.bFieldOrigin * brNorm) * sinAlpha * alpaka::math::sqrt(acc, sec2);
    };
    // Start at the pre-fit circle's point closest to the beam line, with z on the pre-fit line (beamlineSegment).
    const double cNorm = alpaka::math::sqrt(acc, cx * cx + cy * cy);
    double zPca = 0., path0 = 0.;
    beamlineSegment(acc, hits(2, 0), slope, results.sTransverse(0), zPca, path0);
    double xPrev = (cNorm > 0.) ? cx * (1. - rad / cNorm) : 0.;
    double yPrev = (cNorm > 0.) ? cy * (1. - rad / cNorm) : 0.;
    double zPrev = zPca, dLambda = 0., dz = 0.;
    double ratePrev = lambdaRate(xPrev, yPrev, zPrev);
    // the unit radius vector at the start of the segment: the PCA's, then the previous hit's
    auto unit = [&](double& x, double& y) {
      const double norm = alpaka::math::sqrt(acc, x * x + y * y);
      if (norm > 0.) {
        x /= norm;
        y /= norm;
      }
    };
    double nPrevX = xPrev - cx, nPrevY = yPrev - cy;
    unit(nPrevX, nPrevY);
    for (int i = 0; i < n; i++) {
      double nCurX = hits(0, i) - cx, nCurY = hits(1, i) - cy;
      unit(nCurX, nCurY);
      const double dsGap = (i == 0)
                               ? results.sTransverse(0)
                               : rad * alpaka::math::atan2(acc,
                                                           double(results.qCharge) * (nCurX * nPrevY - nCurY * nPrevX),
                                                           nPrevX * nCurX + nPrevY * nCurY);
      const int nSub =
          latticeSteps(acc, cx, cy, rad, nPrevX, nPrevY, -double(results.qCharge) * dsGap / rad, zPrev, hits(2, i));
      const double ds = dsGap / double(nSub);
      // the radius vector turns by -q ds/R per sub-step (the arc convention of sTransverse)
      const double dPsi = -double(results.qCharge) * ds / rad;
      const double cPsi = alpaka::math::cos(acc, dPsi), sPsi = alpaka::math::sin(acc, dPsi);
      double rx = xPrev - cx, ry = yPrev - cy;
      const double dzSub = (hits(2, i) - zPrev) / double(nSub);
      double zSub = zPrev;
      for (int k = 0; k < nSub; ++k) {
        const double rxNew = rx * cPsi - ry * sPsi, ryNew = rx * sPsi + ry * cPsi;
        rx = rxNew;
        ry = ryNew;
        zSub += dzSub;
        const double rateCur =
            (k + 1 == nSub) ? lambdaRate(hits(0, i), hits(1, i), hits(2, i)) : lambdaRate(cx + rx, cy + ry, zSub);
        dz += sec2 * (dLambda * ds + ds * ds * (ratePrev / 3. + rateCur / 6.));
        dLambda += 0.5 * (ratePrev + rateCur) * ds;
        ratePrev = rateCur;
      }
      fitWs.dipShift()(i) = rotMat(1, 1) * dz;  // rotMat(1,1) = cos(lambda): z shift -> sz-plane offset
      xPrev = hits(0, i);
      yPrev = hits(1, i);
      zPrev = hits(2, i);
      nPrevX = nCurX;
      nPrevY = nCurY;
    }
  }

  // The material and the kink variances (fitCorrections). Highland's variance H uses the cumulative depth from the beam
  // line, as ApplyRadX0Correction of SteppingHelixPropagator. Gap g, with depth X_g and material W_g, has the rate
  // K_g = (H(X_g + W_g) - H(X_g))/W_g. Its variance splits between its end nodes with the same total and first moment
  // ([B07] Sec. 2.1), the share fDep = S1/(W_g L) to the departure node. The kink covariance of each gap goes to
  // pointsSZ row 1 and its ionization loss to gapEloss. Every variance carries momentumFactor.
  template <alpaka::concepts::Acc TAcc,
            typename M3xN,
            typename M6xN,
            typename V4,
            typename TData,
            typename TWs,
            int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void materialWithCorrections(const TAcc& acc,
                                                              const blMaterialMap::Map* map,
                                                              const M3xN& hits,
                                                              const M6xN& hits_ge,
                                                              const V4& fast_fit,
                                                              const int mId,
                                                              const double slope,
                                                              const double bField,
                                                              TData& results,
                                                              TWs& fitWs) {
    auto rOf = [&](int j) { return alpaka::math::sqrt(acc, hits(0, j) * hits(0, j) + hits(1, j) * hits(1, j)); };
    double zPca = 0., path0 = 0.;
    beamlineSegment(acc, hits(2, 0), slope, results.sTransverse(0), zPca, path0);
    // The beam-line segment starts the depth and the ionization column for every track (prompt-origin assumption).
    ElossColumn colRun;
    results.innerXX0 =
        segmentXX0Moments(acc, map, zPca, rOf(0), hits(2, 0), results.innerD1, results.innerW1, path0, colRun);
    const double pTot = totalMomentum(acc, bField, fast_fit(2), slope);
    const double p2 = pTot * pTot;
    results.momentumFactor = momentumUncertaintyFactor(acc, hits, hits_ge, mId, n, slope, bField, fast_fit(2));
    const double factor = results.momentumFactor;
    double depth = results.innerXX0;
    for (int g = 0; g < n - 1; g++) {
      const double path = alpaka::math::abs(acc, results.sTotal(g + 1) - results.sTotal(g));
      const auto [L, W, S1, S2, colGap] = segmentWalk(acc, map, rOf(g), hits(2, g), rOf(g + 1), hits(2, g + 1), path);
      const double fDep = (W > 0.) ? S1 / (W * L) : 0.;
      const double rate = (W > 0.) ? (highlandCombined(acc, depth + W, p2) - highlandCombined(acc, depth, p2)) / W : 0.;
      depth += W;
      if (g + 2 < n) {
        results.varBeta(g + 1) = rate * (1. - fDep) * W;  // arrival share, node g+1
      }
      if (g > 0) {
        results.varBeta(g) += rate * fDep * W;  // departure share, node g
      }
      // a gap of zero length (coincident hits in r and z) has no material and no kink covariance
      const double cov = (L > 0.) ? rate * (S1 / L - S2 / (L * L)) * factor : 0.;
      fitWs.pointsSZ(1, g) = cov;
      if (g == 0) {
        results.node0True = (L > 0.) ? rate * S2 / (L * L) * factor : 0.;
        results.node0Cov = cov;
      }
      // The loss over gap g: the loss law on the column to node g plus the share fDep of the gap's own column.
      fitWs.gapEloss()(g) = blMaterialMap::elossTypicalColumn(acc, pTot, colRun + colGap * fDep);
      colRun += colGap;
    }
    // Slot n-1 holds the cumulative loss at the outermost node. A loss that reaches the kinetic energy fails the fit.
    fitWs.gapEloss()(n - 1) = blMaterialMap::elossTypicalColumn(acc, pTot, colRun);
    constexpr double m = blMaterialMap::elossMedium::kPionMass;
    if (!(fitWs.gapEloss()(n - 1) < alpaka::math::sqrt(acc, pTot * pTot + m * m) - m)) {
      results.failed = true;
    }
    results.innerVariance = highlandCombined(acc, results.innerXX0, p2) * factor;
  }

  // The off model: the material of gap g is matXX0(g) = |Delta sTotal| 0.06/16 per cm, the flat inverse radiation
  // length of RiemannFit.h. varBeta(i) reads slots i - 1 and i. innerXX0 is the material of the first gap.
  template <alpaka::concepts::Acc TAcc, typename TData, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void materialOffModel(const TAcc& acc, TData& results) {
    constexpr double kInvX0 = 0.06 / 16.;
    for (int i = 0; i < n - 1; i++) {
      results.matXX0(i) = alpaka::math::abs(acc, results.sTotal(i + 1) - results.sTotal(i)) * kInvX0;
    }
    results.matXX0(n - 1) = 0.;
    results.innerXX0 = alpaka::math::abs(acc, results.sTotal(1) - results.sTotal(0)) * kInvX0;
  }

  // The geometry of the hits on the pre-fit circle (fastFit). qCharge is the sense of the turn from hit 0 over mId to
  // hit n-1. radii are the hits minus the centre. sTransverse is the arc from the circle's point closest to the beam
  // line. sTotal = c (s_T + slope z) and zInSZplane = c (z - slope s_T), c = cos(lambda). It writes only these, so
  // moveCoarseCoordinates (BrokenLineFitKernels.h) can run it again. Returns mId.
  template <alpaka::concepts::Acc TAcc, typename M3xN, typename V4, typename TData, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int preFitGeometry(const TAcc& acc,
                                                    const M3xN& hits,
                                                    const V4& fast_fit,
                                                    TData& results) {
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
    results.radii = hits.template block<2, n>(0, 0).colwise() - fast_fit.head(2);
    eVec = -fast_fit(2) * fast_fit.head(2) / fast_fit.head(2).norm();
    for (int i = 0; i < n; i++) {
      dVec = results.radii.block(0, i, 2, 1);
      results.sTransverse(i) =
          results.qCharge * fast_fit(2) *
          alpaka::math::atan2(acc, riemannFit::cross2D(acc, dVec, eVec), dVec.dot(eVec));  // calculates the arc length
    }
#ifdef BL_DEEPDEBUG
    for (int i = 0; i < n; i++) {
      printf("Point %d, x, %f, y: %f, z: %f, s: %f\n",
             i,
             hits.block(0, 0, 2, n)(0, i),
             hits.block(0, 0, 2, n)(1, i),
             hits(2, i),
             results.sTransverse(i));
    }
#endif

    // calculate sTotal and zInSZplane
    for (int i = 0; i < n; i++) {
      riemannFit::Vector2d pointSZ(results.sTransverse(i), hits(2, i));
      pointSZ = rotMat * pointSZ;
      results.sTotal(i) = pointSZ(0);
      results.zInSZplane(i) = pointSZ(1);
    }
    return mId;
  }

  /*!
    \brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.

    \param hits hits coordinates.
    \param hits_ge hits covariance matrices.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param results PreparedBrokenLineDataMap to be filled.
    \param fitWs the lane's workspace, written only with fitCorrections (dipShift, pointsSZ row 1, gapEloss).
    \param ctx the model (FitContext).
  */
  template <alpaka::concepts::Acc TAcc,
            typename M3xN,
            typename M6xN,
            typename V4,
            typename TData,
            typename TWs,
            int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void __attribute__((always_inline)) prepareBrokenLineData(const TAcc& acc,
                                                                                           const M3xN& hits,
                                                                                           const M6xN& hits_ge,
                                                                                           const V4& fast_fit,
                                                                                           const double bField,
                                                                                           TData& results,
                                                                                           TWs& fitWs,
                                                                                           const FitContext& ctx) {
    const bool fitCorrections = ctx.fitCorrections;
    const int mId = preFitGeometry(acc, hits, fast_fit, results);
    const double slope = -results.qCharge / fast_fit(3);
    const riemannFit::Matrix2d rotMat = rotationMatrix(acc, slope);
    // The dip-angle row, zero when the condition below fails.
    if (fitCorrections) {
      for (int i = 0; i < n; i++) {
        fitWs.dipShift()(i) = 0.;
      }
    }
    if (fitCorrections && ctx.bFieldOrigin > 0. && bField > 0. && fast_fit(2) > 0.) {
      radialFieldDipShift(acc, hits, fast_fit, bField, slope, rotMat, ctx, results, fitWs);
    }
#ifdef BL_DEEPDEBUG
    for (u_int i = 0; i < n; i++) {
      printf("Point %d, rot_s: %f, rot_z: %f\n", i, results.sTotal(i), results.zInSZplane(i));
    }
#endif
    if (fitCorrections) {
      materialWithCorrections(acc, ctx.rhoMap, hits, hits_ge, fast_fit, mId, slope, bField, results, fitWs);
    } else {
      materialOffModel(acc, results);
    }
    //calculate varBeta
    results.varBeta(0) = results.varBeta(n - 1) = 0;
    for (u_int i = 1; i < n - 1; i++) {
      if (fitCorrections) {
        results.varBeta(i) *= results.momentumFactor;
        // A node without material means that the map does not cover the track. The fit cannot be done.
        if (!(results.varBeta(i) > 0.)) {
          results.failed = true;
        }
      } else {
        results.varBeta(i) = multScatt(acc, results.matXX0(i), bField, fast_fit(2), slope) +
                             multScatt(acc, results.matXX0(i - 1), bField, fast_fit(2), slope);
      }
    }
  }

  // The n-by-n band matrix of the straight broken line, for the line fit and the circle fit with fitCorrections off:
  // the measurement weights plus the kink terms with the variances varBeta at the abscissae sTotal, lower-packed in M.
  template <typename VW, typename VS, typename VB, typename TBand, int n = VW::RowsAtCompileTime>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void matrixC_uBandLower(const VW& weights,
                                                         const VS& sTotal,
                                                         const VB& varBeta,
                                                         const TBand& band) {
    for (u_int i = 0; i < n; i++) {
      double d = weights(i);
      if (i > 1)
        d += 1. / (varBeta(i - 1) * riemannFit::sqr(sTotal(i) - sTotal(i - 1)));
      if (i > 0 && i < n - 1)
        d += (1. / varBeta(i)) * riemannFit::sqr((sTotal(i + 1) - sTotal(i - 1)) /
                                                 ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
      if (i < n - 2)
        d += 1. / (varBeta(i + 1) * riemannFit::sqr(sTotal(i + 1) - sTotal(i)));
      band.M(i, i) = d;

      if (i + 1 < n) {  // M(i, i+1) = M(i+1, i), stored lower-packed at row i+1 offset 1
        double o1 = 0.;
        if (i > 0 && i < n - 1) {
          o1 = 1. / (varBeta(i) * (sTotal(i + 1) - sTotal(i))) *
               (-(sTotal(i + 1) - sTotal(i - 1)) / ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
        }
        if (i < n - 2) {
          o1 += 1. / (varBeta(i + 1) * (sTotal(i + 1) - sTotal(i))) *
                (-(sTotal(i + 2) - sTotal(i)) / ((sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i))));
        }
        band.M(i + 1, i) = o1;
      }
      if (i + 2 < n) {  // M(i, i+2) = M(i+2, i), stored lower-packed at row i+2 offset 2
        band.M(i + 2, i) = 1. / (varBeta(i + 1) * (sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i)));
      }
    }
  }

  // The border column of the straight circle fit (fitCorrections off) into border, and its corner as the return value.
  template <typename VS, typename VB, typename TColumn, int n = VS::RowsAtCompileTime>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double circleBorderLower(const VS& sTransverse, const VB& varBeta, TColumn& border) {
    double corner = 0.;
    for (int i = 0; i < n; i++) {
      double b = 0.;
      if (i > 0 && i < n - 1)
        b += -(sTransverse(i + 1) - sTransverse(i - 1)) * (sTransverse(i + 1) - sTransverse(i - 1)) /
             (2. * varBeta(i) * (sTransverse(i + 1) - sTransverse(i)) * (sTransverse(i) - sTransverse(i - 1)));
      if (i > 1) {
        b += (sTransverse(i) - sTransverse(i - 2)) / (2. * varBeta(i - 1) * (sTransverse(i) - sTransverse(i - 1)));
      }
      if (i < n - 2)
        b += (sTransverse(i + 2) - sTransverse(i)) / (2. * varBeta(i + 1) * (sTransverse(i + 1) - sTransverse(i)));
      border(i) = b;
      if (i > 0 && i < n - 1) {
        corner += riemannFit::sqr(sTransverse(i + 1) - sTransverse(i - 1)) / (4. * varBeta(i));
      }
    }
    return corner;
  }

  // Row k of the inverse of the bordered matrix A = [[M, b], [b^T, corner]] at the columns {0, 1, n}, by the Schur
  // complement s of the border (w = M^-1 b): (M^-1(k, 0) + w_k w_0/s, M^-1(k, 1) + w_k w_1/s, -w_k/s).
  struct BorderedColumn {
    double row0, row1, rowN;
  };
  ALPAKA_FN_ACC ALPAKA_FN_INLINE BorderedColumn
  borderedInverseColumn(double invK0, double invK1, double wk, double w0, double w1, double invS) {
    return BorderedColumn{invK0 + wk * w0 * invS, invK1 + wk * w1 * invS, -wk * invS};
  }
  // The block of A^-1 at the indices {0, 1, n}: the rows 0 and 1 by borderedInverseColumn from invKJ = M^-1(k, j), then
  // the border row (-w_0/s, -w_1/s, 1/s).
  template <typename TBand>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE riemannFit::Matrix3d borderedInverseBlock(
      const TBand& band, double inv00, double inv01, double inv10, double inv11, double invS) {
    const double w0 = band.w(0), w1 = band.w(1);
    const BorderedColumn a0 = borderedInverseColumn(inv00, inv01, w0, w0, w1, invS);
    const BorderedColumn a1 = borderedInverseColumn(inv10, inv11, w1, w0, w1, invS);
    riemannFit::Matrix3d block;
    block << a0.row0, a0.row1, a0.rowN, a1.row0, a1.row1, a1.rowN, -w0 * invS, -w1 * invS, invS;
    return block;
  }

  // True when the output of a fit with fitCorrections can be stored as float: a finite state, and a 5x5 covariance
  // (upper triangle read) that is positive definite to double precision, every pivot of its LDL^T significant.
  template <typename V5, typename M5>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool publishable(V5 const& state, M5 const& c) {
    bool ok = true;
    // clang and hipcc fold edm::isNotFinite to false under -Ofast. The comparison catches inf and NaN there.
    for (int i = 0; i < 5; ++i) {
      ok = ok && !edm::isNotFinite(state(i)) && std::abs(state(i)) < std::numeric_limits<float>::max();
    }
    double l[5][5] = {};
    double d[5];
    for (int k = 0; k < 5; ++k) {
      double dk = c(k, k);
      for (int j = 0; j < k; ++j) {
        dk -= l[k][j] * l[k][j] * d[j];
      }
      d[k] = dk;
      ok = ok && significantPivot(dk, c(k, k));
      for (int i = k + 1; i < 5; ++i) {
        double t = c(k, i);
        for (int j = 0; j < k; ++j) {
          t -= l[i][j] * l[k][j] * d[j];
        }
        l[i][k] = t / dk;
      }
    }
    return ok;
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

  // What lineFitOnFittedCircle takes from circleFit with fitCorrections. circle_results.cov then holds G Sigma G^T.
  //   jacobian  G = d(phi, d, kappa)/d(u_0, u_1, Delta-kappa) at node 0.
  //   node0     G times the covariance of (u_0, u_1, Delta-kappa) with the angle at hit 0.
  //   schur     corner - border^T M^-1 border, the Schur complement of the circle's border.
  struct CircleEstimate {
    riemannFit::Matrix3d jacobian;
    riemannFit::Vector3d node0;
    double schur;
  };

  // The hits of a fit without the outer-tracker rules: every hit as measured, one sensor per node.
  struct NoCoarseHits {
    static constexpr bool kActive = false;
  };

  // The gaps on the pre-fit circle (fitCorrections): gap g joins node g to node g+1 and spans the angle A_g at the
  // centre. With n_g the unit radial vector of node g, cos A_g = n_g . n_{g+1} and sin A_g = q (n_{g+1} x n_g).
  template <typename M2xN, typename TCos, typename TSin, int n = M2xN::ColsAtCompileTime>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void curvilinearGaps(M2xN& radii, int qCharge, TCos& gapCos, TSin& gapSin) {
    for (int g = 0; g + 1 < n; ++g) {
      const double ra = radii.block(0, g, 2, 1).norm(), rb = radii.block(0, g + 1, 2, 1).norm();
      const double ax = radii(0, g) / ra, ay = radii(1, g) / ra, bx = radii(0, g + 1) / rb, by = radii(1, g + 1) / rb;
      gapCos(g) = ax * bx + ay * by;
      gapSin(g) = double(qCharge) * (bx * ay - by * ax);
    }
  }

  // The deterministic offset of the circle's residuals (fitCorrections). The curvature departs from the fitted constant
  // by dkappa(s) = kappa_0 (dE(s)/(beta p) + B_bend(r, z)/bField - 1), with dE the loss from the beam line and B_bend
  // the local bending field (BLBFieldMap.h). The offset solves u'' + u/R^2 = -dkappa with u = u' = 0 at node 0, and
  // replaces gapEloss in place (elossOffset). eloss and field select the two sources. pTot is the pre-fit momentum.
  template <alpaka::concepts::Acc TAcc, typename M3xN, typename V4, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void deterministicOffset(const TAcc& acc,
                                                          const M3xN& hits,
                                                          const V4& fast_fit,
                                                          const double bField,
                                                          const FitContext& ctx,
                                                          const double pTot,
                                                          const bool eloss,
                                                          const bool field,
                                                          TData& data,
                                                          TWs& fitWs) {
    const auto& radii = data.radii;
    const auto& gapCos = data.gapCos();
    const auto gapSin = fitWs.gapSin();
    constexpr double m = blMaterialMap::elossMedium::kPionMass;
    // kappa_0 / (beta p) = E / (p^2 R) [1/(cm GeV)]
    const double kappaOverP = alpaka::math::sqrt(acc, pTot * pTot + m * m) / (pTot * pTot * fast_fit(2));
    const double cx = fast_fit(0), cy = fast_fit(1), rad = fast_fit(2);
    const double qCharge = double(data.qCharge);
    const double slopeDen = fast_fit(3) * alpaka::math::abs(acc, fast_fit(2));
    const double invRad = 1. / fast_fit(2);
    // B_bend(r,z)/bField - 1 at a point: the local bending field's departure from the effective one.
    auto bDevAt = [&](double x, double y, double z) {
      if (!field) {
        return 0.;
      }
      const double r = alpaka::math::sqrt(acc, x * x + y * y);
      return ctx.bFieldOrigin * blBFieldMap::bBendAt(*ctx.bMap, r, z, tanLambdaCosAlpha(cx, cy, slopeDen, x, y, r)) /
                 bField -
             1.;
    };
    double eCur = eloss ? fitWs.gapEloss()(0) : 0.;  // gap 0's loss, read before slot 0 is overwritten
    double bPrev = bDevAt(hits(0, 0), hits(1, 0), hits(2, 0));
    double u = 0., up = 0.;
    fitWs.elossOffset()(0) = 0.;  // the anchor
    for (int k = 0; k + 1 < n; ++k) {
      const double cA = gapCos(k), sA = gapSin(k);
      const double length = rad * alpaka::math::atan2(acc, sA, cA);  // the arc of gap k
      const double gLoss = kappaOverP * eCur;  // the ionization growth of the curvature charged over this gap [1/cm]
      const double bCur = bDevAt(hits(0, k + 1), hits(1, k + 1), hits(2, k + 1));
      double uh = cA * u + rad * sA * up - gLoss * rad * rad * oneMinusCos(cA, sA);
      double uph = -sA * u / rad + cA * up - gLoss * rad * sA;
      if (field) {
        double n0x = radii(0, k), n0y = radii(1, k);  // the unit radius vector of hit k
        const double norm = alpaka::math::sqrt(acc, n0x * n0x + n0y * n0y);
        n0x /= norm;
        n0y /= norm;
        const double z0 = hits(2, k), z1 = hits(2, k + 1);
        const int nSub = latticeSteps(acc, cx, cy, rad, n0x, n0y, -qCharge * length / rad, z0, z1);
        const double h = length / double(nSub);
        const CurvilinearStep step(acc, h, rad, false);
        const double a = -qCharge * h / rad;  // the radius vector turns by -q h/R per sub-step
        const double ca = alpaka::math::cos(acc, a), sa = alpaka::math::sin(acc, a);
        double rx = rad * n0x, ry = rad * n0y;  // the reference circle's point at hit k
        double uf = 0., upf = 0.;
        double fPrev = invRad * bPrev;
        for (int j = 1; j <= nSub; ++j) {
          const double rxNew = rx * ca - ry * sa, ryNew = rx * sa + ry * ca;
          rx = rxNew;
          ry = ryNew;
          const double z = z0 + (z1 - z0) * double(j) / double(nSub);
          const double fCur = (j == nSub) ? invRad * bCur : invRad * bDevAt(cx + rx, cy + ry, z);
          step.apply(uf, upf, fPrev, fCur);
          fPrev = fCur;
        }
        uh += uf;
        uph += upf;
      }
      u = uh;
      up = uph;
      bPrev = bCur;
      if (eloss && k + 2 < n) {
        eCur = fitWs.gapEloss()(k + 1);  // next gap's loss, read before the store below
      }
      fitWs.elossOffset()(k + 1) = u;
    }
  }

  // The circle offsets of the dip-kink coupling (fitCorrections). At fixed |p|, the line's dip-angle step S_k
  // (dipSteps, S_0 = 0) changes the curvature of gap k by slope S_k / R ([K12]). This adds the solution of
  // u'' + u/R^2 = -slope S_k / R, with u = u' = 0 at node 0, to offsets.
  template <int n, typename V4, typename TCos, typename TSin, typename TSteps, typename TOffsets>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void dipCouplingOffsets(const V4& fast_fit,
                                                         const double slope,
                                                         const TCos& gapCos,
                                                         const TSin& gapSin,
                                                         const TSteps& dipSteps,
                                                         TOffsets& offsets) {
    const double rad = fast_fit(2);
    double u = 0., up = 0.;
    for (int k = 0; k + 1 < n; ++k) {
      const double source = slope * dipSteps(k) / rad;
      const double cA = gapCos(k), sA = gapSin(k);
      const double uh = cA * u + rad * sA * up - source * rad * rad * oneMinusCos(cA, sA);
      const double uph = -sA * u / rad + cA * up - source * rad * sA;
      u = uh;
      up = uph;
      offsets(k + 1) += u;
    }
  }

  // The two sensors of a vector hit at one node of the circle (fitCorrections). A node whose partner has a measurement
  // (offset y_Q, weight 1/a_Q) takes the least-squares combination with the published offset y_P: the weight becomes
  // 1/(c + a_P a_Q/(a_P + a_Q)) and offsets(i) decreases by a_P/(a_P + a_Q) (y_Q - y_P). a_P = sigma_m^2 (n.M)^2, with
  // n the radial direction and M the precise axis, and c = max(1/w_P - a_P, 0) is the common coarse error.
  template <int n, typename TCoarse, typename M2xN, typename VY, typename VW, typename VO>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void combinePartners(
      const TCoarse& coarse, const M2xN& radii, const VY& zInSZplane, VW& weights, VO& offsets) {
    for (int i = 0; i < n; i++) {
      const auto partner = coarse.partnerMeasurement(i);
      if (!(partner.weight > 0.)) {
        continue;
      }
      const auto s = coarse.sensor(i);
      const double rNorm = radii.block(0, i, 2, 1).norm();
      const double nM = (radii(0, i) * s.mx + radii(1, i) * s.my) / rNorm;
      const double aP = s.sig2m * nM * nM, aQ = 1. / partner.weight;
      const double coarseShare = 1. / weights(i) - aP;
      offsets(i) -= aP / (aP + aQ) * (partner.offset - zInSZplane(i));
      weights(i) = 1. / ((coarseShare > 0. ? coarseShare : 0.) + aP * aQ / (aP + aQ));
    }
  }

  // The curvilinear normal matrix of the circle (fitCorrections): M = diag(w, 0) + sum_i row_i row_i^T / D_i, with the
  // rows of curvilinearKinkRow and D_i = varBeta(i) (1 + slope^2). It writes the band, the border and the corner.
  template <int n, uint32_t S, typename VW, typename TCos, typename TSin, typename VB, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void assembleCurvilinearBand(BandBlock<n, S>& band,
                                                              VW& weightsVec,
                                                              TCos& gapCos,
                                                              TSin& gapSin,
                                                              VB& varBeta,
                                                              const double slope,
                                                              const V4& fast_fit) {
    const double projection = kinkProjection(slope);
    for (int i = 0; i < n; i++) {
      band.M(i, i) = weightsVec(i);
      if (i > 0) {
        band.M(i, i - 1) = 0.;
      }
      if (i > 1) {
        band.M(i, i - 2) = 0.;
      }
      band.border(i) = 0.;
    }
    double corner = 0.;
    for (int i = 1; i + 1 < n; i++) {
      const CurvilinearKinkRow row =
          curvilinearKinkRow(fast_fit(2), gapCos(i - 1), gapSin(i - 1), gapCos(i), gapSin(i));
      const double var = varBeta(i) * projection;
      band.M(i - 1, i - 1) += row.before * row.before / var;
      band.M(i, i) += row.at * row.at / var;
      band.M(i + 1, i + 1) += row.after * row.after / var;
      band.M(i, i - 1) += row.at * row.before / var;
      band.M(i + 1, i) += row.after * row.at / var;
      band.M(i + 1, i - 1) += row.after * row.before / var;
      band.border(i - 1) += row.before * row.dKappa / var;
      band.border(i) += row.at * row.dKappa / var;
      band.border(i + 1) += row.after * row.dKappa / var;
      corner += row.dKappa * row.dKappa / var;
    }
    band.corner = corner;
  }

  // Solves the bordered system A x = r of the circle ([B06] Eq. 8) into uVec by the Schur complement of the border.
  // Returns schur. With fitCorrections, a pivot without a significant digit fails the fit.
  template <typename TBand, typename VR, typename VU, typename TData, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double solveBordered(
      TBand& band, VR& r_uVec, VU& uVec, const bool fitCorrections, TData& data) {
    band.factorTestingPivots(band.ms, fitCorrections, data.failed);
    for (int i = 0; i < n; i++) {
      band.w(i) = band.border(i);
    }
    band.solveInPlace(band.w);  // w = M^-1 border
    double schur = band.corner;
    for (int i = 0; i < n; i++) {
      schur -= band.border(i) * band.w(i);  // schur = corner - border^T M^-1 border
    }
    if (fitCorrections && !significantPivot(schur, band.corner)) {
      data.failed = true;
    }
    // uVec = A^-1 r_uVec: border x_n = (r_n - w^T rb)/schur, band x = M^-1 rb - w x_n.
    double dot = 0.;
    for (int i = 0; i < n; i++) {
      band.ms(i) = r_uVec(i);
      dot += band.w(i) * r_uVec(i);
    }
    band.solveInPlace(band.ms);  // ms = M^-1 rb
    const double xN = (r_uVec(n) - dot) / schur;
    for (int i = 0; i < n; i++) {
      uVec(i) = band.ms(i) - band.w(i) * xN;
    }
    uVec(n) = xN;
    return schur;
  }

  // The covariance of the circle's estimate (u_0, u_1, Delta-kappa) under the true kink covariance c_g of each gap
  // (pointsSZ row 1): Sigma = A^-1 - sum_g c_g (1 + slope^2) v_g v_g^T, v_g = A^-1 (r_g/D_g - r_{g+1}/D_{g+1}), with
  // r_i the kink row of node i and D_i = varBeta(i) (1 + slope^2). resp = -(A^-1 r_1)/D_1. circleInvCol0 and
  // circleInvCol1 keep M^-1 e_0 and M^-1 e_1 for propagateCircleIntoLine.
  template <typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void estimateCovariance(TData& data,
                                                         TWs& fitWs,
                                                         const double rad,
                                                         const double projection,
                                                         const double schur,
                                                         riemannFit::Matrix3d& sigma,
                                                         riemannFit::Vector3d& resp) {
    auto& band = fitWs.band;
    const auto& gapCos = data.gapCos();
    const auto gapSin = fitWs.gapSin();
    const auto& varBeta = data.varBeta;
    auto& inv0 = fitWs.circleInvCol0;
    auto& inv1 = fitWs.circleInvCol1();
    const double invS = 1. / schur;
    band.template solveUnitColumn<0>(inv0);  // M^-1 e_0
    band.template solveUnitColumn<1>(inv1);  // M^-1 e_1
    const double w0 = band.w(0), w1 = band.w(1);
    sigma = borderedInverseBlock(band, inv0(0), inv1(0), inv0(1), inv1(1), invS);
    // (A^-1 r_i)_{0,1,n}/D_i for an interior node i
    auto responseOf = [&](int i, riemannFit::Vector3d& out) {
      const CurvilinearKinkRow row = curvilinearKinkRow(rad, gapCos(i - 1), gapSin(i - 1), gapCos(i), gapSin(i));
      out << -row.dKappa * w0 * invS, -row.dKappa * w1 * invS, row.dKappa * invS;
      for (int a = 0; a < 3; a++) {
        const int k = i - 1 + a;
        const double wk = band.w(k);
        out(0) += row.u(a) * (inv0(k) + wk * w0 * invS);
        out(1) += row.u(a) * (inv1(k) + wk * w1 * invS);
        out(2) -= row.u(a) * wk * invS;
      }
      out /= varBeta(i) * projection;
    };
    riemannFit::Vector3d prev = riemannFit::Vector3d::Zero();  // of the departure node of gap g (0 at node 0)
    resp = riemannFit::Vector3d::Zero();
    for (int g = 0; g + 1 < n; g++) {
      riemannFit::Vector3d next = riemannFit::Vector3d::Zero();
      if (g + 2 < n) {
        responseOf(g + 1, next);
        if (g == 0) {
          resp = -next;
        }
      }
      const riemannFit::Vector3d v = prev - next;
      sigma -= (fitWs.pointsSZ(1, g) * projection) * v * v.transpose();
      prev = next;
    }
  }

  /*!
    \brief Performs the Broken Line fit in the curved track case (that is, the fit
   *       parameters are the interceptions u and the curvature correction \Delta\kappa).

    \param hits hits coordinates.
    \param hits_ge hits covariance matrices.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineDataMap.
    \param circle_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (phi, d, k); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    \param estimate (fitCorrections) filled with what lineFitOnFittedCircle propagates (CircleEstimate).
    \param fitWs the lane's workspace (FitWorkspaceMap).
    \param ctx the model (FitContext).
    \param coarse the outer-tracker rules of the lane (NoCoarseHits without them). With fitCorrections, a vector hit
           with a partner measurement combines its two sensors (combinePartners).

    \details With fitCorrections, the fit uses curvilinear kink rows and the residual offsets of the energy loss, the
    field and the dip steps. The perigee parameters come from the state at node 0. Off, they come from the chord
    through the fitted points of hits 0 and 1, with the innermost scattering added at hit 0.
  */
  template <alpaka::concepts::Acc TAcc,
            typename M3xN,
            typename M6xN,
            typename V4,
            typename TData,
            typename TWs,
            typename TCoarse = NoCoarseHits,
            int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void circleFit(const TAcc& acc,
                                                const M3xN& hits,
                                                const M6xN& hits_ge,
                                                const V4& fast_fit,
                                                const double bField,
                                                TData& data,
                                                karimaki_circle_fit& circle_results,
                                                CircleEstimate& estimate,
                                                TWs& fitWs,
                                                const FitContext& ctx,
                                                const TCoarse& coarse = TCoarse{}) {
    const bool fitCorrections = ctx.fitCorrections;
    circle_results.qCharge = data.qCharge;
    auto& radii = data.radii;
    const auto& sTransverse = data.sTransverse;
    auto& zInSZplane = data.zInSZplane;
    auto& varBeta = data.varBeta;
    const double slope = -circle_results.qCharge / fast_fit(3);
    // Project the kink variances on the transverse plane, 1 + slope^2 = 1/cos^2(lambda). With fitCorrections, the line
    // fit runs later on the 3-D variances, so varBeta stays unprojected and each read below multiplies.
    if (!fitCorrections) {
      varBeta *= 1. + riemannFit::sqr(slope);
    }

    for (u_int i = 0; i < n; i++) {
      zInSZplane(i) = radii.block(0, i, 2, 1).norm() - fast_fit(2);
    }

    auto& gapCos = data.gapCos();
    auto gapSin = fitWs.gapSin();
    if (fitCorrections) {
      curvilinearGaps(radii, data.qCharge, gapCos, gapSin);
    }

    const double pTot = totalMomentum(acc, bField, fast_fit(2), slope);
    // The loss needs a momentum and a radius. The field needs a positive field and a radius.
    const bool eloss = fitCorrections && pTot > 0. && fast_fit(2) > 0.;
    const bool field = (fitCorrections && ctx.bFieldOrigin > 0. && bField > 0. && fast_fit(2) > 0.);
    const bool offset = eloss || field;
    if (offset) {
      deterministicOffset(acc, hits, fast_fit, bField, ctx, pTot, eloss, field, data, fitWs);
    }

    riemannFit::Matrix2d vMat;               // covariance matrix
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

    // fitCorrections: add the offsets of the dip-kink coupling, then combine the two sensors of each vector hit.
    if (fitCorrections) {
      if (!offset) {
        for (int i = 0; i < n; i++) {
          fitWs.elossOffset()(i) = 0.;
        }
      }
      dipCouplingOffsets<n>(fast_fit, slope, gapCos, gapSin, fitWs.dipSteps(), fitWs.elossOffset());
      if constexpr (TCoarse::kActive) {
        combinePartners<n>(coarse, radii, zInSZplane, weightsVec, fitWs.elossOffset());
      }
    }

    auto& r_uVec = fitWs.circleRu;
    // r(i) = w(i) (zInSZplane(i) - elossOffset(i)) and r(n) = 0: the offset enters through the measurements only.
    r_uVec(n) = 0;
    if (fitCorrections) {
      for (int i = 0; i < n; i++) {
        r_uVec(i) = weightsVec(i) * (zInSZplane(i) - fitWs.elossOffset()(i));
      }
    } else {
      for (int i = 0; i < n; i++) {
        r_uVec(i) = weightsVec(i) * zInSZplane(i);
      }
    }

    auto& uVec = fitWs.circleU;
    auto& band = fitWs.band;
    if (fitCorrections) {
      assembleCurvilinearBand(band, weightsVec, gapCos, gapSin, varBeta, slope, fast_fit);
    } else {
      matrixC_uBandLower(weightsVec, sTransverse, varBeta, band);
      band.corner = circleBorderLower(sTransverse, varBeta, band.border);
    }
    const double schur = solveBordered(band, r_uVec, uVec, fitCorrections, data);

    if (fitCorrections) {
      const double q = double(data.qCharge);
      const double rad = fast_fit(2);
      const double projection = kinkProjection(slope);
      riemannFit::Matrix3d sigma;
      riemannFit::Vector3d resp;
      estimateCovariance(data, fitWs, rad, projection, schur, sigma, resp);

      // The state at node 0 from the first two fitted offsets ([B06] Sec. 4): the circle through the point P with the
      // direction T, moved to the beam line by translateKarimaki. The covariance is G Sigma G^T.
      const double nNorm0 = radii.block(0, 0, 2, 1).norm();
      const double n0x = radii(0, 0) / nNorm0, n0y = radii(1, 0) / nNorm0;
      const double t0x = q * n0y, t0y = -q * n0x;
      const double cX = gapCos(0), sX = gapSin(0), thX = tanHalf(cX, sX);
      const double u0 = uVec(0), u1 = uVec(1), dk = uVec(n);
      const double du0 = (u1 - u0 * cX) / (rad * sX) + dk * rad * thX;
      const double tx = (1. + u0 / rad) * t0x + du0 * n0x, ty = (1. + u0 / rad) * t0y + du0 * n0y;
      const double px = hits(0, 0) + (u0 - zInSZplane(0)) * n0x, py = hits(1, 0) + (u0 - zInSZplane(0)) * n0y;
      const riemannFit::Vector3d local(alpaka::math::atan2(acc, ty, tx), 0., q * (1. / rad + dk));
      riemannFit::Matrix3d jacT;
      Eigen::Matrix<double, 3, 2> jacXY;
      translateKarimaki(acc, local, px, py, jacT, &circle_results.par, &jacXY);
      const double dT[2][3] = {{t0x / rad - n0x * cX / (rad * sX), n0x / (rad * sX), rad * thX * n0x},
                               {t0y / rad - n0y * cX / (rad * sX), n0y / (rad * sX), rad * thX * n0y}};
      const double tNorm2 = tx * tx + ty * ty;
      riemannFit::Matrix3d dLocal = riemannFit::Matrix3d::Zero();
      for (int k = 0; k < 3; k++) {
        dLocal(0, k) = (tx * dT[1][k] - ty * dT[0][k]) / tNorm2;
      }
      dLocal(2, 2) = q;
      Eigen::Matrix<double, 2, 3> dP = Eigen::Matrix<double, 2, 3>::Zero();
      dP(0, 0) = n0x;
      dP(1, 0) = n0y;
      riemannFit::Matrix3d jacG;
      jacG.noalias() = jacT * dLocal;
      jacG.noalias() += jacXY * dP;
      similarity(jacG, sigma, circle_results.cov);
      estimate.jacobian = jacG;
      estimate.node0 = jacG * (resp * projection * data.node0Cov);
      estimate.schur = schur;
    } else {
      // compute (phi, d_ca, k) in the system in which the midpoint of the first two corrected hits is the origin...

      radii.block(0, 0, 2, 1) /= radii.block(0, 0, 2, 1).norm();
      radii.block(0, 1, 2, 1) /= radii.block(0, 1, 2, 1).norm();

      riemannFit::Vector2d dVec = hits.block(0, 0, 2, 1) + (-zInSZplane(0) + uVec(0)) * radii.block(0, 0, 2, 1);
      riemannFit::Vector2d eVec = hits.block(0, 1, 2, 1) + (-zInSZplane(1) + uVec(1)) * radii.block(0, 1, 2, 1);
      auto eMinusd = eVec - dVec;
      auto eMinusd2 = eMinusd.squaredNorm();
      auto tmp1 = 1. / eMinusd2;
      auto tmp2 = alpaka::math::sqrt(acc, riemannFit::sqr(fast_fit(2)) - 0.25 * eMinusd2);

      circle_results.par << atan2(eMinusd(1), eMinusd(0)), circle_results.qCharge * (tmp2 - fast_fit(2)),
          circle_results.qCharge * (1. / fast_fit(2) + uVec(n));

      tmp2 = 1. / tmp2;

      riemannFit::Matrix3d jacobian;
      jacobian << (radii(1, 0) * eMinusd(0) - eMinusd(1) * radii(0, 0)) * tmp1,
          (radii(1, 1) * eMinusd(0) - eMinusd(1) * radii(0, 1)) * tmp1, 0,
          circle_results.qCharge * (eMinusd(0) * radii(0, 0) + eMinusd(1) * radii(1, 0)) * tmp2,
          circle_results.qCharge * (eMinusd(0) * radii(0, 1) + eMinusd(1) * radii(1, 1)) * tmp2, 0, 0, 0,
          circle_results.qCharge;

      // The block of A^-1 at the indices (0, 1, n), from M^-1 e_0 and M^-1 e_1.
      {
        const double invS = 1. / schur;
        band.template solveUnitColumn<0>(band.ms);            // ms = M^-1 e_0
        const double inv00 = band.ms(0), inv10 = band.ms(1);  // M^-1(0,0), M^-1(1,0)
        band.template solveUnitColumn<1>(band.ms);            // ms = M^-1 e_1
        circle_results.cov = borderedInverseBlock(band, inv00, band.ms(0), inv10, band.ms(1), invS);
      }

      similarity(jacobian, circle_results.cov, circle_results.cov);

      //...Translate in the system in which the first corrected hit is the origin, adding the m.s. correction...

      translateKarimaki(acc, circle_results.par, 0.5 * eMinusd(0), 0.5 * eMinusd(1), jacobian, &circle_results.par);
      similarity(jacobian, circle_results.cov, circle_results.cov);
      // innermost MS: the first gap's material as an angle at hit 0, where node 0 has no kink
      circle_results.cov(0, 0) +=
          (1 + riemannFit::sqr(slope)) * multScatt(acc, data.innerXX0, bField, fast_fit(2), slope);

      //...And translate back to the original system

      translateKarimaki(acc, circle_results.par, dVec(0), dVec(1), jacobian, &circle_results.par);
      similarity(jacobian, circle_results.cov, circle_results.cov);
    }

    // chi2. The model at node i is uVec(i) + elossOffset(i). The offset enters the measurement term only.
    circle_results.chi2 = 0;
    if (fitCorrections) {
      const double projection = kinkProjection(slope);
      for (int i = 0; i < n; i++) {
        circle_results.chi2 += weightsVec(i) * riemannFit::sqr(zInSZplane(i) - fitWs.elossOffset()(i) - uVec(i));
        if (i > 0 && i < n - 1) {
          const CurvilinearKinkRow row =
              curvilinearKinkRow(fast_fit(2), gapCos(i - 1), gapSin(i - 1), gapCos(i), gapSin(i));
          circle_results.chi2 += riemannFit::sqr(row.before * uVec(i - 1) + row.at * uVec(i) + row.after * uVec(i + 1) +
                                                 row.dKappa * uVec(n)) /
                                 (varBeta(i) * projection);
        }
      }
    } else {
      for (int i = 0; i < n; i++) {
        circle_results.chi2 += weightsVec(i) * riemannFit::sqr(zInSZplane(i) - uVec(i));
        if (i > 0 && i < n - 1) {
          circle_results.chi2 +=
              riemannFit::sqr(uVec(i - 1) / (sTransverse(i) - sTransverse(i - 1)) -
                              uVec(i) * (sTransverse(i + 1) - sTransverse(i - 1)) /
                                  ((sTransverse(i + 1) - sTransverse(i)) * (sTransverse(i) - sTransverse(i - 1))) +
                              uVec(i + 1) / (sTransverse(i + 1) - sTransverse(i)) +
                              (sTransverse(i + 1) - sTransverse(i - 1)) * uVec(n) / 2) /
              varBeta(i);
        }
      }
    }
  }

  /*!
    \brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).
    
    \param hits_ge hits covariance matrices.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineDataMap.
    \param line_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (cot(theta), Zip); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    \param fitWs the lane's workspace (FitWorkspaceMap).
    \param ctx the model (FitContext). With fitCorrections, the covariance uses the true kink covariance. Off, it adds
           the innermost multScatt term.

    \details The least-squares fit of u, then (cot(theta), Zip) from the fitted points of hits 0 and 1.
   */
  template <alpaka::concepts::Acc TAcc, typename V4, typename M6xN, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void lineFit(const TAcc& acc,
                                              const M6xN& hits_ge,
                                              const V4& fast_fit,
                                              const double bField,
                                              TData& data,
                                              riemannFit::LineFit& line_results,
                                              TWs& fitWs,
                                              const FitContext& ctx) {
    const bool fitCorrections = ctx.fitCorrections;
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

    auto& r_u = fitWs.lineRu;
    for (u_int i = 0; i < n; i++) {
      r_u(i) = weights(i) * zInSZplane(i);
    }
    auto& uVec = fitWs.lineU;
    auto& band = fitWs.band;
    {
      matrixC_uBandLower(weights, sTotal, varBeta, band);
      band.factorTestingPivots(band.ms, fitCorrections, data.failed);
      for (int i = 0; i < n; i++) {  // load the rhs, then solve in place: uVec = M^-1 r_u
        uVec(i) = r_u(i);
      }
      band.solveInPlace(uVec);
    }

    // line parameters in the system in which the first hit is the origin and with axis along SZ
    line_results.par << (uVec(1) - uVec(0)) / (sTotal(1) - sTotal(0)), uVec(0);
    auto idiff = 1. / (sTotal(1) - sTotal(0));
    // The line covariance from M^-1 e_0 and M^-1 e_1. Off, it adds the innermost MS as an angle at hit 0.
    {
      auto& col = band.ms;
      band.template solveUnitColumn<0>(col);  // M^-1 e_0
      const double i00 = col(0);
      auto& inv0 = fitWs.lineInvCol0();
      if (fitCorrections) {
        for (int i = 0; i < n; i++) {
          inv0(i) = col(i);
        }
      }
      auto& inv1 = fitWs.lineInvCol1();
      band.template solveUnitColumn<1>(inv1);  // M^-1 e_1
      const double i01 = inv1(0), i11 = inv1(1);
      if (fitCorrections) {
        // The covariance of (u_0, u_1) under the true kink covariance, as in estimateCovariance with the straight
        // kink rows and D_i = varBeta(i).
        double s00 = i00, s01 = i01, s11 = i11;
        double prev0 = 0., prev1 = 0.;  // (M^-1 r_g)_{0,1}/D_g of the departure node of gap g (0 at node 0)
        for (int g = 0; g + 1 < n; g++) {
          double next0 = 0., next1 = 0.;
          const int i = g + 1;  // the arrival node of gap g
          if (i + 1 < n) {
            const auto [ra, rb, rc] = straightKinkRow(sTotal, i);
            next0 = (ra * inv0(i - 1) + rb * inv0(i) + rc * inv0(i + 1)) / varBeta(i);
            next1 = (ra * inv1(i - 1) + rb * inv1(i) + rc * inv1(i + 1)) / varBeta(i);
          }
          const double v0 = prev0 - next0, v1 = prev1 - next1;
          const double c = fitWs.pointsSZ(1, g);
          s00 -= c * v0 * v0;
          s01 -= c * v0 * v1;
          s11 -= c * v1 * v1;
          prev0 = next0;
          prev1 = next1;
        }
        // (t, u_0) = ((u_1 - u_0)/ds, u_0)
        line_results.cov << (s00 - 2 * s01 + s11) * riemannFit::sqr(idiff), (s01 - s00) * idiff, (s01 - s00) * idiff,
            s00;
      } else {
        line_results.cov << (i00 - 2 * i01 + i11) * riemannFit::sqr(idiff) +
                                multScatt(acc, data.innerXX0, bField, fast_fit(2), slope),
            (i01 - i00) * idiff, (i01 - i00) * idiff, i00;
      }
    }

    // translate to the original SZ system
    riemannFit::Matrix2d jacobian;
    jacobian(0, 0) = 1.;
    jacobian(0, 1) = 0;
    jacobian(1, 0) = -sTotal(0);
    jacobian(1, 1) = 1.;
    line_results.par(1) += -line_results.par(0) * sTotal(0);
    similarity(jacobian, line_results.cov, line_results.cov);

    // rotate to the original sz system
    auto tmp = rotMat(0, 0) - line_results.par(0) * rotMat(0, 1);
    jacobian(1, 1) = 1. / tmp;
    jacobian(0, 0) = jacobian(1, 1) * jacobian(1, 1);
    jacobian(0, 1) = 0;
    jacobian(1, 0) = line_results.par(1) * rotMat(0, 1) * jacobian(0, 0);
    line_results.par(1) = line_results.par(1) * jacobian(1, 1);
    line_results.par(0) = (rotMat(0, 1) + line_results.par(0) * rotMat(0, 0)) * jacobian(1, 1);
    similarity(jacobian, line_results.cov, line_results.cov);

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

  // The fitted circle (phi, d, rho) with the cosine and sine of phi, as the line on it reads them (fitCorrections).
  struct FittedCircle {
    double phi, d, rho, cPhi, sPhi;
  };

  // The abscissae on the fitted circle (fitCorrections): S_i is the arc from its perigee to its crossing with the ray
  // of hit i (through h_i along the pre-fit radial n_i), in Karimaki's implicit form [K91]. A ray that misses the
  // circle or is tangent to it fails the fit. It writes s_i = c (S_i + slope z_i) into sTotal, z'_i = c (z_i -
  // slope S_i) - c dz_i into zInSZplane and dS_i/d(phi, d, rho) into dsdPhi, dsdD and dsdRho. Returns S_0.
  template <alpaka::concepts::Acc TAcc, typename M3xN, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double abscissaeOnFittedCircle(const TAcc& acc,
                                                                const M3xN& hits,
                                                                const FittedCircle& fitted,
                                                                const double cDip,
                                                                const double slope,
                                                                TData& data,
                                                                TWs& fitWs) {
    const auto& [phiF, dF, rhoF, cPhi, sPhi] = fitted;
    const auto& radii = data.radii;
    auto dsdD = fitWs.dsdD();
    const double p0x = dF * sPhi, p0y = -dF * cPhi;      // the perigee
    double tX = cPhi, tY = sPhi, mX = sPhi, mY = -cPhi;  // the tangent and the normal at the preceding crossing
    double xPrev = p0x, yPrev = p0y;
    double arc = 0., arcNode0 = 0.;
    for (int i = 0; i < n; i++) {
      const double hx = hits(0, i), hy = hits(1, i);
      const double nNorm = radii.block(0, i, 2, 1).norm();
      const double nx = radii(0, i) / nNorm, ny = radii(1, i) / nNorm;
      const double gh =
          0.5 * rhoF * (hx * hx + hy * hy) - (1. + rhoF * dF) * (hx * sPhi - hy * cPhi) + dF * (1. + 0.5 * rhoF * dF);
      const double bh = rhoF * (hx * nx + hy * ny) - (1. + rhoF * dF) * (nx * sPhi - ny * cPhi);
      const double disc = bh * bh - 2. * rhoF * gh;
      if (!(disc >= 0.) || bh == 0.) {
        data.failed = true;
      }
      const double tRay = -2. * gh / (bh + alpaka::math::copysign(acc, alpaka::math::sqrt(acc, disc), bh));
      const double xx = hx + tRay * nx, yy = hy + tRay * ny;
      const double a = (xx - xPrev) * tX + (yy - yPrev) * tY, b = (xx - xPrev) * mX + (yy - yPrev) * mY;
      const double sn = rhoF * a, cs = 1. - rhoF * b;  // the sine and cosine of the turn between the crossings
      arc += (rhoF != 0.) ? alpaka::math::atan2(acc, sn, cs) / rhoF : a;
      const double nrm = alpaka::math::sqrt(acc, sn * sn + cs * cs);
      const double tXn = (cs * tX + sn * mX) / nrm, tYn = (cs * tY + sn * mY) / nrm;
      const double mXn = (cs * mX - sn * tX) / nrm, mYn = (cs * mY - sn * tY) / nrm;
      tX = tXn;
      tY = tYn;
      mX = mXn;
      mY = mYn;
      xPrev = xx;
      yPrev = yy;
      const double den = -ny * tX + nx * tY;  // nu . t
      if (den == 0.) {
        data.failed = true;
      }
      const double relA = (xx - p0x) * cPhi + (yy - p0y) * sPhi, relB = (xx - p0x) * sPhi - (yy - p0y) * cPhi;
      const double cc = tX * cPhi + tY * sPhi, ss = tX * sPhi - tY * cPhi;
      const double nuT0 = -ny * cPhi + nx * sPhi, nuM0 = -ny * sPhi - nx * cPhi;  // nu . t_0, nu . m_0
      const double nuRho =
          (rhoF != 0.) ? ((arc * cc - relA) / rhoF) * nuT0 + ((arc * ss - relB) / rhoF) * nuM0 : 0.5 * arc * arc * nuM0;
      data.dsdPhi()(i) = -((dF + relB) * nuT0 - relA * nuM0) / den;
      dsdD(i) = -nuM0 / den;
      fitWs.dsdRho()(i) = -nuRho / den;
      if (i == 0) {
        arcNode0 = arc;
      }
      const double z = hits(2, i);
      data.sTotal(i) = cDip * (arc + slope * z);
      data.zInSZplane(i) = cDip * (z - slope * arc) - fitWs.dipShift()(i);
    }
    return arcNode0;
  }

  // The circle's estimate propagated into the line (fitCorrections): the blocks of cov5 without the terms of node 0.
  // J = sum_i -c (slope + t) G_i dS_i/d(phi, d, rho), with t the line's slope in (s, z') and G_i the line's gain for
  // measurement i through L = toPerigee. X = sum_i (G_c R_i) D_i G_i^T, with G_c = estimate.jacobian, R_i the circle's
  // response to measurement i and D_i the covariance of hit i's circle and line errors (coarse.crossShare).
  template <typename M6xN, typename TData, typename TWs, typename TCoarse = NoCoarseHits, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void propagateCircleIntoLine(const M6xN& hits_ge,
                                                              const double q,
                                                              const double slope,
                                                              const double cDip,
                                                              const double t,
                                                              const double ds,
                                                              const riemannFit::Matrix2d& toPerigee,
                                                              TData& data,
                                                              const karimaki_circle_fit& circle_results,
                                                              const CircleEstimate& estimate,
                                                              const riemannFit::LineFit& line_results,
                                                              TWs& fitWs,
                                                              Eigen::Matrix<double, 2, 3>& jac,
                                                              Eigen::Matrix<double, 5, 5>& cov5,
                                                              const TCoarse& coarse = TCoarse{}) {
    const auto& radii = data.radii;
    const auto dsdD = fitWs.dsdD();
    const auto& lineInv0 = fitWs.lineInvCol0();
    const auto& lineInv1 = fitWs.lineInvCol1();
    const double invS = 1. / estimate.schur;
    const double w0 = fitWs.band.w(0), w1 = fitWs.band.w(1);  // the circle's w = M^-1 border (circleFit)
    const double lever = -cDip * (slope + t);
    jac = Eigen::Matrix<double, 2, 3>::Zero();
    Eigen::Matrix<double, 2, 3> crossOwn = Eigen::Matrix<double, 2, 3>::Zero();  // X^T
    for (int i = 0; i < n; i++) {
      const double g0 = lineInv0(i) * fitWs.lineWeights(i), g1 = lineInv1(i) * fitWs.lineWeights(i);
      const riemannFit::Vector2d gain = toPerigee * riemannFit::Vector2d((g1 - g0) / ds, g0);
      const riemannFit::Vector3d dsdc(data.dsdPhi()(i), dsdD(i), fitWs.dsdRho()(i));
      jac += (lever * gain) * dsdc.transpose();
      const double wi = fitWs.band.w(i);
      const BorderedColumn ai =
          borderedInverseColumn(fitWs.circleInvCol0(i), fitWs.circleInvCol1()(i), wi, w0, w1, invS);
      const riemannFit::Vector3d response = fitWs.circleWeights(i) * riemannFit::Vector3d(ai.row0, ai.row1, ai.rowN);
      const double rNorm = radii.block(0, i, 2, 1).norm();
      const double nx = radii(0, i) / rNorm, ny = radii(1, i) / rNorm;
      const double gx = -cDip * slope * q * ny, gy = cDip * slope * q * nx, gz = cDip;
      const double cxx = hits_ge.col(i)[0], cxy = hits_ge.col(i)[1], cyy = hits_ge.col(i)[2];
      const double cxz = hits_ge.col(i)[3], cyz = hits_ge.col(i)[4];
      double shared = nx * (cxx * gx + cxy * gy + cxz * gz) + ny * (cxy * gx + cyy * gy + cyz * gz);  // D_i
      if constexpr (TCoarse::kActive) {
        shared = coarse.crossShare(i, nx, ny, gx, gy, gz, shared);
      }
      crossOwn += (shared * gain) * (estimate.jacobian * response).transpose();
    }
    const riemannFit::Matrix3d& covEstimate = circle_results.cov;
    Eigen::Matrix<double, 2, 3> jacCov;
    jacCov.noalias() = jac * covEstimate;
    cov5.topLeftCorner<3, 3>() = covEstimate;
    cov5.block<2, 3>(3, 0) = jacCov + crossOwn;
    cov5.block<3, 2>(0, 3) = cov5.block<2, 3>(3, 0).transpose();
    auto lineBlock = cov5.bottomRightCorner<2, 2>();
    lineBlock = line_results.cov;
    lineBlock.noalias() += jacCov * jac.transpose();
    lineBlock.noalias() += jac * crossOwn.transpose();
    lineBlock.noalias() += crossOwn * jac.transpose();
  }

  // E, the scattering before node 0, added to cov5: per thin scatterer k, turnA = A_k (transverse turn) and turnB =
  // B_k (dip turn) per unit angle. transverse0 and dip0 are those of the scatterer at node 0.
  template <alpaka::concepts::Acc TAcc, typename TData>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void scatteringBeforeNode0(const TAcc& acc,
                                                            TData& data,
                                                            const riemannFit::LineFit& line_results,
                                                            const double arcNode0,
                                                            const double projection,
                                                            const FittedCircle& fitted,
                                                            const riemannFit::Matrix2d& toPerigee,
                                                            Eigen::Matrix<double, 5, 1>& transverse0,
                                                            Eigen::Matrix<double, 5, 1>& dip0,
                                                            Eigen::Matrix<double, 5, 5>& cov5) {
    const auto& [phiF, dF, rhoF, cPhi, sPhi] = fitted;
    const double cot = line_results.par(0);
    const double w1s = (data.innerW1 > 0.) ? data.innerW1 : 0.;
    const double variance[2] = {(1. - w1s) * data.innerVariance + data.node0True, w1s * data.innerVariance};
    const double path[2] = {0., data.innerD1};
    const int nScatterers = (w1s > 0.) ? 2 : 1;
    for (int k = 0; k < nScatterers; k++) {
      const double a = arcNode0 - path[k] / alpaka::math::sqrt(acc, projection);
      const double ca = alpaka::math::cos(acc, rhoF * a), sa = alpaka::math::sin(acc, rhoF * a);
      const double sinOver = (rhoF != 0.) ? sa / rhoF : a;                    // sin(rho a)/rho
      const double omcOver = (rhoF != 0.) ? oneMinusCos(ca, sa) / rhoF : 0.;  // (1 - cos(rho a))/rho
      const double sx = dF * sPhi + sinOver * cPhi + omcOver * sPhi, sy = -dF * cPhi + sinOver * sPhi - omcOver * cPhi;
      const riemannFit::Vector2d tS(ca * cPhi + sa * sPhi, ca * sPhi - sa * cPhi), t0(cPhi, sPhi), m0(sPhi, -cPhi);
      Eigen::Matrix<double, 2, 3> dxdc;  // d(S)/d(phi, d, rho) at fixed arc
      dxdc.col(0) = (dF + omcOver) * t0 - sinOver * m0;
      dxdc.col(1) = m0;
      dxdc.col(2) = (rhoF != 0.)
                        ? riemannFit::Vector2d(((a * ca - sinOver) / rhoF) * t0 + ((a * sa - omcOver) / rhoF) * m0)
                        : riemannFit::Vector2d(0.5 * a * a * m0);
      riemannFit::Matrix3d jacS;
      translateKarimaki(acc, riemannFit::Vector3d(phiF - rhoF * a, 0., rhoF), sx, sy, jacS);
      const riemannFit::Vector3d dcA = jacS.col(0);
      Eigen::Matrix<double, 5, 1> turnA;
      turnA << dcA, 0., cot * tS.dot(dxdc * dcA);
      const riemannFit::Vector2d dl = toPerigee * riemannFit::Vector2d(1., path[k]);
      const riemannFit::Vector3d dcB = jacS.col(2) * (rhoF * cot * dl(0) / (1. + cot * cot));
      Eigen::Matrix<double, 5, 1> turnB;
      turnB << dcB, dl(0), dl(1) + cot * tS.dot(dxdc * dcB);
      cov5 += projection * variance[k] * turnA * turnA.transpose() + variance[k] * turnB * turnB.transpose();
      if (k == 0) {
        transverse0 = turnA;
        dip0 = turnB;
      }
    }
  }

  // The covariances of the two estimates with the angle at hit 0, added to cov5: v for the circle (estimate.node0 and
  // its image through jac), o for the line from r0, r1 = -(M^-1 r_1)_{0,1} node0Cov/D_1.
  template <typename TData, typename TWs>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void node0Correlations(TData& data,
                                                        TWs& fitWs,
                                                        const CircleEstimate& estimate,
                                                        const double q,
                                                        const double ds,
                                                        const riemannFit::Matrix2d& toPerigee,
                                                        const Eigen::Matrix<double, 2, 3>& jac,
                                                        const Eigen::Matrix<double, 5, 1>& transverse0,
                                                        const Eigen::Matrix<double, 5, 1>& dip0,
                                                        Eigen::Matrix<double, 5, 5>& cov5) {
    const auto& lineInv0 = fitWs.lineInvCol0();
    const auto& lineInv1 = fitWs.lineInvCol1();
    const auto [ra, rb, rc] = straightKinkRow(data.sTotal, 1);
    const double r0 = -(ra * lineInv0(0) + rb * lineInv0(1) + rc * lineInv0(2)) / data.varBeta(1) * data.node0Cov;
    const double r1 = -(ra * lineInv1(0) + rb * lineInv1(1) + rc * lineInv1(2)) / data.varBeta(1) * data.node0Cov;
    Eigen::Matrix<double, 5, 1> v, o;
    v << estimate.node0, jac * estimate.node0;
    o << 0., 0., 0., toPerigee * riemannFit::Vector2d((r1 - r0) / ds, r0);
    const Eigen::Matrix<double, 5, 1> qa0 = q * transverse0;
    cov5 += v * qa0.transpose() + qa0 * v.transpose() + o * dip0.transpose() + dip0 * o.transpose();
  }

  // The line z = z0 + cot(theta) S on the fitted circle's arcs S and the 5x5 covariance of (phi, d, kappa, cot(theta),
  // z0) (fitCorrections): [[C, C J^T + X], [J C + X^T, C_l + J C J^T + J X + X^T J^T]] + E + the correlations with the
  // angle at hit 0 ([FS21] Sec. 8.1.2.3, [SF10] Eq. 16). C and C_l are the covariances of the two estimates,
  // J = d(cot, z0)/d(phi, d, kappa), X their covariance through shared hits, E the scattering before node 0. hits and
  // hits_ge are the hits as measured.
  template <alpaka::concepts::Acc TAcc,
            typename M3xN,
            typename M6xN,
            typename V4,
            typename TData,
            typename TWs,
            typename TCoarse = NoCoarseHits,
            int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void lineFitOnFittedCircle(const TAcc& acc,
                                                            const M3xN& hits,
                                                            const M6xN& hits_ge,
                                                            const V4& fast_fit,
                                                            const double bField,
                                                            TData& data,
                                                            const karimaki_circle_fit& circle_results,
                                                            const CircleEstimate& estimate,
                                                            riemannFit::LineFit& line_results,
                                                            TWs& fitWs,
                                                            Eigen::Matrix<double, 5, 5>& cov5,
                                                            const FitContext& ctx,
                                                            const TCoarse& coarse = TCoarse{}) {
    const double q = double(data.qCharge);
    const double slope = -data.qCharge / fast_fit(3);
    const double projection = kinkProjection(slope);
    const double cDip = 1. / alpaka::math::sqrt(acc, projection);
    const double phiF = circle_results.par(0);
    const FittedCircle fitted{
        phiF, circle_results.par(1), circle_results.par(2), alpaka::math::cos(acc, phiF), alpaka::math::sin(acc, phiF)};
    const double arcNode0 = abscissaeOnFittedCircle(acc, hits, fitted, cDip, slope, data, fitWs);

    lineFit(acc, hits_ge, fast_fit, bField, data, line_results, fitWs, ctx);

    // t, the line's fitted slope in the (s, z') plane, and L = toPerigee
    const double s0 = data.sTotal(0), ds = data.sTotal(1) - s0;
    const double t = (fitWs.lineU(1) - fitWs.lineU(0)) / ds;
    const double offset = fitWs.lineU(0) - t * s0;
    const double dLine = cDip * (1. - t * slope);
    riemannFit::Matrix2d rotate, shift;
    rotate << 1. / (dLine * dLine), 0., offset * slope * cDip / (dLine * dLine), 1. / dLine;
    shift << 1., 0., -s0, 1.;
    const riemannFit::Matrix2d toPerigee = rotate * shift;  // L
    Eigen::Matrix<double, 2, 3> jac;
    propagateCircleIntoLine(hits_ge,
                            q,
                            slope,
                            cDip,
                            t,
                            ds,
                            toPerigee,
                            data,
                            circle_results,
                            estimate,
                            line_results,
                            fitWs,
                            jac,
                            cov5,
                            coarse);

    Eigen::Matrix<double, 5, 1> transverse0, dip0;
    scatteringBeforeNode0(acc, data, line_results, arcNode0, projection, fitted, toPerigee, transverse0, dip0, cov5);
    node0Correlations(data, fitWs, estimate, q, ds, toPerigee, jac, transverse0, dip0, cov5);
  }

  /*!
    \brief Moves the state from the fitted circle's perigee to the trajectory's own closest approach (fitCorrections).

    \param hits hits coordinates.
    \param data PreparedBrokenLineDataMap (the charge).
    \param bField the effective bending field of the fit [GeV/cm].
    \param circle_results the circle fit's output (phi, d, k).
    \param line_results the line fit's output (cot(theta), zip) on the fitted circle's arcs.
    \param ctx the field map and its normalization (FitContext).

    \details Between the perigee and hit 0, the radial offset v of the trajectory from the fitted circle solves
    v'' + v/R^2 = -(B_bend/bField - 1)/R, with v = v' = 0 at hit 0. The function moves phi, d and zip to first order
    in v. The curvature, the dip and the covariances do not change. Reference: CMS, JINST 9 (2014) P10009, Sec. 4.3.
  */
  template <alpaka::concepts::Acc TAcc, typename M3xN, typename TData>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void transportToFittedPca(const TAcc& acc,
                                                           const M3xN& hits,
                                                           const TData& data,
                                                           const double bField,
                                                           karimaki_circle_fit& circle_results,
                                                           riemannFit::LineFit& line_results,
                                                           const FitContext& ctx) {
    const double kFit = circle_results.par(2);
    if (!(alpaka::math::abs(acc, kFit) > 0.) || !(ctx.bFieldOrigin > 0. && bField > 0.)) {
      return;
    }
    const double qCharge = double(data.qCharge);
    const double radFit = 1. / alpaka::math::abs(acc, kFit);
    // perigee = d * mVec and centre = (d + q R) * mVec, so the perigee's radius vector is -q R mVec
    const double mx = alpaka::math::sin(acc, circle_results.par(0));
    const double my = -alpaka::math::cos(acc, circle_results.par(0));
    const double cFit = circle_results.par(1) + qCharge * radFit;  // signed distance of the centre along mVec
    const double cx = cFit * mx, cy = cFit * my;
    const double dx = hits(0, 0) - cx, dy = hits(1, 0) - cy;
    const double ex = -qCharge * radFit * mx, ey = -qCharge * radFit * my;
    // the arc of hit 0 from the fitted perigee, in the convention of sTransverse
    const double sFit = qCharge * radFit * alpaka::math::atan2(acc, dx * ey - dy * ex, dx * ex + dy * ey);
    const double cot = line_results.par(0);
    // the local curvature's departure from the fitted one, dkappa = (B_bend(r, z)/bField - 1)/R, on the fitted circle
    auto dKappaAt = [&](double x, double y, double z) {
      const double r = alpaka::math::sqrt(acc, x * x + y * y);
      if (!(r > 0.)) {
        return 0.;
      }
      const double tanLambdaCosAlphaFit = cot * qCharge * (cx * y - cy * x) / (radFit * r);
      return (ctx.bFieldOrigin * blBFieldMap::bBendAt(*ctx.bMap, r, z, tanLambdaCosAlphaFit) / bField - 1.) / radFit;
    };
    // the particular solution P from the perigee to hit 0, on the sub-steps of latticeSteps
    const double zPca = hits(2, 0) - cot * sFit;
    const int nSub =
        latticeSteps(acc, cx, cy, radFit, ex / radFit, ey / radFit, -qCharge * sFit / radFit, zPca, hits(2, 0));
    const double h = sFit / double(nSub);
    const CurvilinearStep step(acc, h, radFit);
    const double a = -qCharge * h / radFit;  // the radius vector turns by -q h/R per sub-step
    const double ca = alpaka::math::cos(acc, a), sa = alpaka::math::sin(acc, a);
    double rx = ex, ry = ey;
    double pu = 0., pup = 0., pInt = 0.;  // P(L), P'(L), int_0^L P ds
    double fPrev = dKappaAt(cx + rx, cy + ry, zPca);
    for (int k = 1; k <= nSub; ++k) {
      const double rxNew = rx * ca - ry * sa, ryNew = rx * sa + ry * ca;
      rx = rxNew;
      ry = ryNew;
      const double fCur = (k == nSub) ? dKappaAt(hits(0, 0), hits(1, 0), hits(2, 0))
                                      : dKappaAt(cx + rx, cy + ry, zPca + cot * h * double(k));
      step.apply(pu, pup, pInt, fPrev, fCur);
      fPrev = fCur;
    }
    // the homogeneous solution that cancels P at hit 0: (v, R v') at the perigee is (P, R P') at hit 0 rotated back
    // by Theta = L/R, with the opposite sign
    const double theta = sFit / radFit;
    const double cT = alpaka::math::cos(acc, theta), sT = alpaka::math::sin(acc, theta);
    const double omcT = oneMinusCos(cT, sT);
    const double v0 = -(pu * cT - radFit * pup * sT);
    const double vp0 = -(pu * sT / radFit + pup * cT);
    const double vInt = v0 * radFit * sT + vp0 * radFit * radFit * omcT + pInt;  // int_0^L v ds
    const double aCentre = alpaka::math::abs(acc, cFit);
    const double sPca = (aCentre > 0.) ? qCharge * vp0 * circle_results.par(1) * radFit / aCentre : 0.;
    const double dsArc = -sPca + vInt / radFit;  // the arc the trajectory spends over the lever, minus the circle's
    circle_results.par(1) -= qCharge * v0;
    circle_results.par(0) =
        cms::alpakatools::reducePhiRange(acc, circle_results.par(0) + qCharge * (vp0 - sPca / radFit));
    line_results.par(1) -= cot * dsArc;
  }

  // A hit's sensor: the local axes M (precise) and E (coarse) in global coordinates, and the local variances sigma_m^2
  // along M and sigma_s^2 along E. sigma_s^2 = 0 for a hit that the rule does not take, such as a pixel hit.
  struct HitSensor {
    double mx, my, mz;
    double ex, ey, ez;
    double sig2m, sig2s;
  };

  // Whether the coarse-coordinate rule takes a hit: an outer-tracker hit whose coarse axis has a transverse part the
  // float covariance can show, sigma_s^2 |E_T|^2 > u_f sigma_m^2 |M_T|^2 (u_f = 2^-24).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isCoarseCandidate(const HitSensor& s) {
    constexpr double kFloatUnitRoundoff = 0.5 * double(std::numeric_limits<float>::epsilon());  // 2^-24
    const double eT2 = s.ex * s.ex + s.ey * s.ey, mT2 = s.mx * s.mx + s.my * s.my;
    return s.sig2s > 0. && s.sig2s * eT2 > kFloatUnitRoundoff * s.sig2m * mT2;
  }

  // The line on the pre-fit arcs (fitCorrections), before circleFit, on the hits as measured. It writes the weights
  // 1/(g^T C g) into lineWeights, with g = (-c slope q n_y, c slope q n_x, c) the direction that z' measures. It
  // factors the band on the pre-fit abscissae and writes u = M^-1 (w z'), z'_i = zInSZplane_i - dipShift_i, into
  // circleU. lineDipSteps and coarseCoordinateRule read the offsets and the band.
  template <alpaka::concepts::Acc TAcc, typename M6xN, typename V4, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void lineOnPreFitArcs(
      const TAcc& acc, const M6xN& hits_ge, const V4& fast_fit, TData& data, TWs& fitWs) {
    const double slope = -data.qCharge / fast_fit(3);
    const double cDip = 1. / alpaka::math::sqrt(acc, kinkProjection(slope));
    const double q = double(data.qCharge);
    auto& weights = fitWs.lineWeights;
    for (int i = 0; i < n; i++) {
      const double rNorm = data.radii.block(0, i, 2, 1).norm();
      const double gx = -cDip * slope * q * data.radii(1, i) / rNorm, gy = cDip * slope * q * data.radii(0, i) / rNorm;
      const double cxx = hits_ge.col(i)[0], cxy = hits_ge.col(i)[1], cyy = hits_ge.col(i)[2];
      const double cxz = hits_ge.col(i)[3], cyz = hits_ge.col(i)[4], czz = hits_ge.col(i)[5];
      weights(i) = 1. / (gx * (cxx * gx + cxy * gy + cxz * cDip) + gy * (cxy * gx + cyy * gy + cyz * cDip) +
                         cDip * (cxz * gx + cyz * gy + czz * cDip));
    }
    auto& band = fitWs.band;
    auto& u = fitWs.circleU;
    matrixC_uBandLower(weights, data.sTotal, data.varBeta, band);
    band.factor();
    for (int i = 0; i < n; i++) {
      u(i) = weights(i) * (data.zInSZplane(i) - fitWs.dipShift()(i));
    }
    band.solveInPlace(u);
  }

  // The dip-angle step S_g of every gap into dipSteps (fitCorrections): the change of the line's slope from gap 0 to
  // gap g, S_g = D_g/L_g with D_g = (u_{g+1} - u_g) - L_g (u_1 - u_0)/L_0 and L_g = s_{g+1} - s_g, on the line of
  // lineOnPreFitArcs. S_0 = S_{n-1} = 0. Every S_g is 0 for a failed fit.
  template <typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void lineDipSteps(const TData& data, TWs& fitWs) {
    auto& steps = fitWs.dipSteps();
    for (int g = 0; g < n; g++) {
      steps(g) = 0.;
    }
    if (data.failed) {
      return;
    }
    const auto& u = fitWs.circleU;
    const double slopeOfGap0 = (u(1) - u(0)) / (data.sTotal(1) - data.sTotal(0));
    for (int g = 1; g + 1 < n; g++) {
      const double gap = data.sTotal(g + 1) - data.sTotal(g);
      const double offsetChangeBeyondGap0 = (u(g + 1) - u(g)) - gap * slopeOfGap0;  // D_g
      steps(g) = offsetChangeBeyondGap0 / gap;
    }
  }

  // The partner sensor of a vector hit: its rechit's position, moved by projectPartner and coarseCoordinateRule, and
  // the local variance sigma_m^2 of its precise coordinate. Its axes are the published sensor's. sig2m = 0: no partner.
  struct PartnerSensor {
    double x, y, z;
    double sig2m;
  };
  // A partner's circle measurement at its node: its offset y_Q from the pre-fit circle and the weight 1/a_Q, with
  // a_Q = sigma_m^2 (n_Q.M)^2. Weight 0: no partner.
  struct PartnerMeasurement {
    double offset, weight;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hasPartner(const PartnerSensor& q) { return q.sig2m > 0.; }

  // The partner q of node i moved along the published sensor's coarse axis E to the crossing of its plane (normal
  // N = M x E, through q) with the straight line through the published hit p along the track's direction D:
  //     X = p + (N.(q - p)/(N.D)) D,   q' = q + (E.(X - q)) E.
  // The precise coordinate of q stays and its coarse one becomes X's. D is the track direction at the node, from the
  // pre-fit circle's tangent and the slope of the line of lineOnPreFitArcs on the gap at the partner's side.
  // q does not move if N.D has no significant digit, |N.D| <= gamma_3 sum_k |N_k D_k| ([Hi02] Sec. 3.1).
  template <alpaka::concepts::Acc TAcc, typename V4, typename TData, typename TWs, int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void projectPartner(const TAcc& acc,
                                                     const V4& fast_fit,
                                                     const TData& data,
                                                     const TWs& fitWs,
                                                     int i,
                                                     double px,
                                                     double py,
                                                     double pz,
                                                     const HitSensor& s,
                                                     PartnerSensor& q) {
    const double nx = s.my * s.ez - s.mz * s.ey, ny = s.mz * s.ex - s.mx * s.ez, nz = s.mx * s.ey - s.my * s.ex;
    const double separation = nx * (q.x - px) + ny * (q.y - py) + nz * (q.z - pz);  // N.(q - p)
    const double charge = double(data.qCharge);
    const double slope = -charge / fast_fit(3);
    const double rx = data.radii(0, i), ry = data.radii(1, i);
    const double rNorm = alpaka::math::sqrt(acc, rx * rx + ry * ry);
    const double nT = charge * (nx * ry - ny * rx);  // |r| N.T
    const bool downstream = separation * (nT + nz * rNorm * slope) > 0.;
    const int g = downstream ? (i + 1 < n ? i : n - 2) : (i > 0 ? i - 1 : 0);
    const double ds = data.sTotal(g + 1) - data.sTotal(g), du = fitWs.circleU(g + 1) - fitWs.circleU(g);
    const double alongT = ds - slope * du;
    const double dx = alongT * charge * ry, dy = -alongT * charge * rx, dz = rNorm * (slope * ds + du);
    const double nD = nx * dx + ny * dy + nz * dz;
    constexpr double unitRoundoff = 0.5 * std::numeric_limits<double>::epsilon();
    constexpr double gamma3 = 3. * unitRoundoff / (1. - 3. * unitRoundoff);
    const double bound =
        gamma3 * (alpaka::math::abs(acc, nx * dx) + alpaka::math::abs(acc, ny * dy) + alpaka::math::abs(acc, nz * dz));
    if (!(alpaka::math::abs(acc, nD) > bound)) {
      return;
    }
    const double lambda = separation / nD;
    const double e =
        s.ex * (px + lambda * dx - q.x) + s.ey * (py + lambda * dy - q.y) + s.ez * (pz + lambda * dz - q.z);
    q.x += e * s.ex;
    q.y += e * s.ey;
    q.z += e * s.ez;
  }

  // The coarse-coordinate rule (fitCorrections, outer-tracker hits). For each candidate (isCoarseCandidate), it
  // predicts the coarse coordinate a = g zeta of the track's crossing with the module from the line of
  // lineOnPreFitArcs: zeta = k (u_i - z'_i), k = c (1 + cot slope), g = w_z/(E.w), w = n x T, T = (q (n_y, -n_x), cot).
  // c and slope are the pre-fit's cos(lambda) and cot(theta), cot is the line's and n is the hit's radial direction.
  // The prediction has the variance g^2 k^2 Sigma_ii + f^2 sigma_m^2, f = (M.w)/(E.w), with Sigma_ii the line's
  // variance at node i. If it is smaller than sigma_s^2, the hit and its partner move by a along E, clamped to the
  // half-strip, and the hit takes that variance. moved(0, i) = 1 for a moved hit, and the caller clears it. circleRu
  // holds M^-1 e_i. Returns the number of moved hits.
  template <alpaka::concepts::Acc TAcc,
            typename M3xN,
            typename M6xN,
            typename TSensors,
            typename TPartners,
            typename MMoved,
            typename V4,
            typename TData,
            typename TWs,
            int n = TData::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int coarseCoordinateRule(const TAcc& acc,
                                                          M3xN& hits,
                                                          M6xN& hits_ge,
                                                          const TSensors& sensors,
                                                          TPartners& partners,
                                                          MMoved& moved,
                                                          const V4& fast_fit,
                                                          TData& data,
                                                          TWs& fitWs) {
    const double slope = -data.qCharge / fast_fit(3);
    const double cDip = 1. / alpaka::math::sqrt(acc, kinkProjection(slope));
    const double q = double(data.qCharge);
    const auto& band = fitWs.band;
    const auto& u = fitWs.circleU;
    const double t = (u(1) - u(0)) / (data.sTotal(1) - data.sTotal(0));
    const double cot = (slope + t) / (1. - slope * t);
    const double k = cDip * (1. + cot * slope);  // zeta = k (u_i - z'_i)
    auto& xi = fitWs.circleRu;

    int nMoved = 0;
    for (int i = 0; i < n; i++) {
      const HitSensor s = sensors.sensor(i);
      if (!isCoarseCandidate(s)) {
        continue;
      }
      for (int j = 0; j < n; j++) {
        xi(j) = 0.;
      }
      xi(i) = 1.;
      band.solveInPlace(xi);
      // Sigma_ii: M^-1 minus the true-kink terms, as in lineFit
      double covII = xi(i);
      double prevI = 0.;
      for (int g = 0; g + 1 < n; g++) {
        double nextI = 0.;
        const int a = g + 1;  // the arrival node of gap g
        if (a + 1 < n) {
          const auto [ra, rb, rc] = straightKinkRow(data.sTotal, a);
          nextI = (ra * xi(a - 1) + rb * xi(a) + rc * xi(a + 1)) / data.varBeta(a);
        }
        const double vI = prevI - nextI;
        covII -= fitWs.pointsSZ(1, g) * vI * vI;
        prevI = nextI;
      }
      const double zeta = k * (u(i) - (data.zInSZplane(i) - fitWs.dipShift()(i)));
      const double varZeta = k * k * covII;
      if (!(varZeta > 0.) || edm::isNotFinite(zeta) || edm::isNotFinite(varZeta)) {
        continue;
      }
      // the crossing with the module plane: w = n x T, g = w_z/(E.w)
      const double rNorm = data.radii.block(0, i, 2, 1).norm();
      const double nx = data.radii(0, i) / rNorm, ny = data.radii(1, i) / rNorm;
      const double wx = ny * cot, wy = -nx * cot, wz = -q;
      const double eW = s.ex * wx + s.ey * wy + s.ez * wz, mW = s.mx * wx + s.my * wy + s.mz * wz;
      const double num = wz * wz * varZeta + mW * mW * s.sig2m;
      // take the prediction where its variance beats the strip's
      if (!(num < s.sig2s * eW * eW)) {
        continue;
      }
      const double varU = num / (eW * eW);
      const double gain = wz / eW;
      const double half = alpaka::math::sqrt(acc, 3. * s.sig2s);
      const double a = alpaka::math::min(acc, alpaka::math::max(acc, gain * zeta, -half), half);
      hits(0, i) += a * s.ex;
      hits(1, i) += a * s.ey;
      hits_ge(0, i) = float(s.sig2m * s.mx * s.mx + varU * s.ex * s.ex);
      hits_ge(1, i) = float(s.sig2m * s.mx * s.my + varU * s.ex * s.ey);
      hits_ge(2, i) = float(s.sig2m * s.my * s.my + varU * s.ey * s.ey);
      hits_ge(3, i) = float(s.sig2m * s.mx * s.mz + varU * s.ex * s.ez);
      hits_ge(4, i) = float(s.sig2m * s.my * s.mz + varU * s.ey * s.ez);
      hits_ge(5, i) = float(s.sig2m * s.mz * s.mz + varU * s.ez * s.ez);
      moved(0, i) = 1.f;
      ++nMoved;
      // the partner moves with its hit; z stays, as the moved hit's
      PartnerSensor partner = partners.partner(i);
      if (hasPartner(partner)) {
        partner.x += a * s.ex;
        partner.y += a * s.ey;
        partners.setPartner(i, partner);
      }
    }
    return nMoved;
  }

  // The dip-angle row for the moved hits' pre-fit: dipShift = c dz_i scales by c'/c, c' = 1/sqrt(1 + slopeMoved^2).
  template <alpaka::concepts::Acc TAcc, typename TWs, int n = TWs::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void reprojectDipShift(const TAcc& acc,
                                                        const double slope,
                                                        const double slopeMoved,
                                                        TWs& fitWs) {
    const double scale = alpaka::math::sqrt(acc, kinkProjection(slope) / kinkProjection(slopeMoved));
    for (int i = 0; i < n; i++) {
      fitWs.dipShift()(i) *= scale;
    }
  }

  // The partners' circle measurements on the pre-fit circle fast_fit (centre c, radius R): the offset
  // y_Q = |x_Q - c| - R and the weight 1/a_Q, a_Q = sigma_m^2 (n_Q.M)^2. Weight 0 without a partner or when a_Q = 0.
  template <alpaka::concepts::Acc TAcc, typename V4, typename TSensors, typename TPartners, int n = TPartners::kN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void partnerMeasurements(const TAcc& acc,
                                                          const V4& fast_fit,
                                                          const TSensors& sensors,
                                                          TPartners& partners) {
    for (int i = 0; i < n; i++) {
      PartnerMeasurement m{0., 0.};
      const PartnerSensor q = partners.partner(i);
      if (hasPartner(q)) {
        const HitSensor s = sensors.sensor(i);
        const double rx = q.x - fast_fit(0), ry = q.y - fast_fit(1);
        const double r = alpaka::math::sqrt(acc, rx * rx + ry * ry);
        const double nx = rx / r, ny = ry / r;
        const double mN = s.mx * nx + s.my * ny;
        const double aQ = q.sig2m * mN * mN;
        if (aQ > 0.) {
          m = PartnerMeasurement{r - fast_fit(2), 1. / aQ};
        }
      }
      partners.setMeasurement(i, m);
    }
  }

  // The rules' view of one lane for circleFit and propagateCircleIntoLine: the hits' sensors, the moved hits of
  // coarseCoordinateRule and the partners' circle measurements of partnerMeasurements.
  template <typename TSensors, typename MMoved, typename TPartners>
  struct CoarseHitsLane {
    static constexpr bool kActive = true;
    const TSensors& sensors;
    const MMoved& movedHits;
    const TPartners& partners;
    ALPAKA_FN_ACC bool moved(int i) const { return movedHits(0, i) != 0.f; }
    ALPAKA_FN_ACC PartnerMeasurement partnerMeasurement(int i) const { return partners.measurement(i); }
    ALPAKA_FN_ACC HitSensor sensor(int i) const { return sensors.sensor(i); }
    // D_i, the covariance of hit i's circle and line measurement errors. A moved hit has its precise share
    // (n.M)(g.M) sigma_m^2, another hit has shared. A node combined with its partner subtracts a_P/(a_P + a_Q) times
    // the precise share.
    ALPAKA_FN_ACC double crossShare(int i, double nx, double ny, double gx, double gy, double gz, double shared) const {
      const HitSensor s = sensors.sensor(i);
      const double nM = nx * s.mx + ny * s.my;
      const double precise = nM * (gx * s.mx + gy * s.my + gz * s.mz) * s.sig2m;
      const double own = moved(i) ? precise : shared;
      const double wQ = partners.measurement(i).weight;
      if (!(wQ > 0.)) {
        return own;
      }
      const double aP = s.sig2m * nM * nM, aQ = 1. / wQ;
      return own - aP / (aP + aQ) * precise;
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::brokenline

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BrokenLine_h
