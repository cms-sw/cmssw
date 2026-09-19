#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_GeneralBrokenLine_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_GeneralBrokenLine_h

// General Broken Lines device fit (alpaka). This is the implementation the pixel-track fit producers run and it
// is normative for the fit model; the host implementation in interface/GeneralBrokenLine.h is a readable
// reference for the same algebra, compiled only into the unit tests. Relative to it: std:: -> alpaka::math::,
// the dynamic system replaced by fixed-N stack matrices, the Jacobian-block inverses by explicit inv2/inv3 (the
// device fitting code avoids Eigen .inverse()), and the normal matrix solved in bordered-band form out of a
// caller-provided scratch block. The host file carries the long-form derivation and the references to DESY
// GblTrajectory.cpp / CMSSW AnalyticalCurvilinearJacobian.

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"    // normalized (Bz,Br) r-z map
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"  // the ionization column of a path
namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine {

  using namespace cms::alpakatools;

  using ElossColumn = blMaterialMap::ElossColumn;

  // Constants of the two ionization energy-loss laws below (elossMostProbable, elossTypicalColumn): both
  // evaluate the same Landau xi on the same column and differ only in which statistic of the loss
  // distribution they return. The medium is no longer a constant: every material quantity comes from the
  // column the material map's dE/dx lattice produced for the path actually walked.
  namespace elossMedium {
    constexpr double kPionMass = 0.13957;           // pion mass [GeV]
    constexpr double kElectronMass = 0.5109989e-3;  // electron mass [GeV]
    constexpr double kK = 0.307075e-3;              // 0.307 MeV cm^2/mol -> GeV
    // standard Landau: lambda_median - lambda_mode = 1.35578 - (-0.22278)
    constexpr double kLandauMedianMinusMode = 1.35578 + 0.22278;
    // hbar omega_p = 28.816 eV sqrt(rho Z/A) (PDG RPP 34.2.5) and the eV -> GeV shift, both as logarithms:
    // the laws work on the column's log-means and never form I or the plasma energy.
    constexpr double kLnPlasmaEV = 3.360930788433600;  // ln(28.816)
    constexpr double kLnGeVinEV = 20.723265836946410;  // ln(1e9)
  }  // namespace elossMedium

  // Landau scale xi [GeV] and most-probable bracket of a composite column at total momentum p (pion mass). The
  // Landau family is stable, so a column of lumps is one Landau with xi = sum xi_i and
  //   Delta_mp = xi [ln(2 me beta^2 gamma^2 xi) + 0.2 - beta^2] - sum_i xi_i [2 ln I_i + delta_i],
  // which needs the xi-weighted means of ln I and of ln rho_e that ElossColumn carries. Evaluating the
  // density effect at the column's effective (ln I, ln rho_e) is exact asymptotically and second order in the
  // spread of Cbar (< 1 % for the tracker's mix); delta follows Sternheimer's parametrization with the generic
  // Sternheimer-Peierls parameters (PDG RPP 34.2.5).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool elossLandau(
      const TAcc& acc, double p, const ElossColumn& col, double& xi, double& bracket) {
    xi = 0.;
    bracket = 0.;
    if (!(col.e > 0.) || !(p > 0.))
      return false;
    constexpr double kLn10 = 2.302585092994046;
    constexpr double twoLn10 = 2. * kLn10;  // the constant Sternheimer's parametrization rounds to 4.6052
    constexpr double m = elossMedium::kPionMass;
    constexpr double me = elossMedium::kElectronMass;
    constexpr double K = elossMedium::kK;
    const double E = alpaka::math::sqrt(acc, p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;
    xi = 0.5 * K * col.e / beta2;
    // The column's effective medium enters only through ln I and, via the plasma energy, ln rho_e, and the
    // column already carries both as logarithms: I and hbar omega_p are never formed.
    const double lnIeV = col.eLnI / col.e;               // ln(I/eV)
    const double lnI = lnIeV - elossMedium::kLnGeVinEV;  // ln(I/GeV)
    const double cbar = 2. * (lnIeV - 0.5 * col.eLnRho / col.e - elossMedium::kLnPlasmaEV) + 1.;
    // Sternheimer-Peierls generic solid/liquid parameters, by the I < / >= 100 eV rule (m = 3)
    const bool soft = lnIeV < 2. * kLn10;  // I < 100 eV
    const double x1 = soft ? 2. : 3.;
    const double cbarLim = soft ? 3.681 : 5.215;
    const double x0 = (cbar < cbarLim) ? 0.2 : (0.326 * cbar - (soft ? 1.0 : 1.5));
    const double lnBg2 = alpaka::math::log(acc, bg2);
    const double x = 0.5 * lnBg2 / kLn10;  // log10(betagamma)
    double delta = 0.;
    if (x >= x1) {
      delta = twoLn10 * x - cbar;
    } else if (x > x0) {
      const double a = (cbar - twoLn10 * x0) / ((x1 - x0) * (x1 - x0) * (x1 - x0));
      const double d = x1 - x;
      delta = twoLn10 * x - cbar + a * d * d * d;
    }
    // ln(2 me beta^2 gamma^2 / I) + ln(xi / I) = ln(2 me beta^2 gamma^2 xi) - 2 ln I
    bracket = alpaka::math::log(acc, 2. * me * bg2 * xi) - 2. * lnI + 0.2 - beta2 - delta;
    return true;
  }

  // Most-probable (Landau) ionization energy loss [GeV] of the column, at total momentum p [GeV]. The
  // correction has to remove the loss of the typical track, not the unrestricted Bethe-Bloch mean, which
  // includes the Landau delta-ray tail. MP is not additive across sub-columns (the ln xi term).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double elossMostProbable(const TAcc& acc, double p, const ElossColumn& col) {
    double xi, bracket;
    if (!elossLandau(acc, p, col, xi, bracket))
      return 0.;
    const double dmp = xi * bracket;
    return dmp > 0. ? dmp : 0.;
  }

  // Typical (median) ionization loss [GeV] of the CUMULATIVE column, selected over the per-lump most-probable
  // law by the runtime flag elossCumulative. The Landau family is stable under convolution, so the typical
  // loss of a multi-lump column is the single-column law at the SUMMED column, not the sum of per-lump MPVs,
  // and median - mode = (1.35578 + 0.22278) xi for a Landau (PDG RPP 34.2.9). Callers charge per-node
  // increments T(col_cum + col_lump) - T(col_cum).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double elossTypicalColumn(const TAcc& acc, double p, const ElossColumn& col) {
    double xi, bracket;
    if (!elossLandau(acc, p, col, xi, bracket))
      return 0.;
    const double dmp = xi * bracket;
    return (dmp > 0. ? dmp : 0.) + elossMedium::kLandauMedianMinusMode * xi;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_GeneralBrokenLine_h
