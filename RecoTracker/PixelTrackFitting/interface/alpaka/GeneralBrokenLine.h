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
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"  // normalized (Bz,Br) r-z map
namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine {

  using namespace cms::alpakatools;
  // Constants of the two ionization energy-loss laws below (elossMostProbable, elossTypicalColumn).
  // Defined ONCE here: both laws evaluate the same Landau xi in the same medium and differ only in which
  // statistic of the loss distribution they return, so every number below has exactly one home.
  //
  // COMPOSITE EFFECTIVE MEDIUM of the material the fit charges, not silicon. Derived from the D121
  // MaterialBudgetAction step trees (per-step Material X0 and Density) with the Bragg additivity rule
  // (PDG RPP 34.2.5): only the PRODUCT (Z/A)*X0g enters xi, and the X/X0-weighted <(Z/A)*X0g> of the
  // charged band is 14.4 g/cm^2 (per-band 13.9-16.1) against silicon's 0.498*21.82 = 10.866 -- the fit
  // was under-charging the ionization loss by 27-48 %. I_eff and rho_eff are the same weighting (ln I
  // mass-weighted by Z/A); hwp follows from rho_eff by PDG eq. 34.7, 28.816*sqrt(3.15*0.500) = 36.16 eV.
  // No tuned constant. (These are the medium of the loss LAWS; they are unrelated to the eLossPerX0
  // argument the callers pass, which is only an on/off flag -- see kELossPerX0 in BrokenLineFitKernels.h.)
  namespace elossMedium {
    constexpr double kPionMass = 0.13957;           // pion mass [GeV]
    constexpr double kElectronMass = 0.5109989e-3;  // electron mass [GeV]
    constexpr double kK = 0.307075e-3;              // 0.307 MeV cm^2/mol -> GeV
    constexpr double kZ_A = 0.500;                  // composite <Z/A> (A = 2Z closure; +-4 % band)
    constexpr double kI = 122e-9;                   // composite mean excitation energy [GeV]
    constexpr double kX0g = 28.8;                   // composite X0 [g/cm^2]; (Z/A)*X0g = 14.4
    constexpr double kHwp = 36.16e-9;               // composite plasma energy [GeV] (Sternheimer)
    // standard Landau: lambda_median - lambda_mode = 1.35578 - (-0.22278)
    constexpr double kLandauMedianMinusMode = 1.35578 + 0.22278;
  }  // namespace elossMedium

  // Cumulative-column typical (median) energy loss, always compiled; the runtime flag elossCumulative selects
  // it over the per-lump most-probable law at each of the four charge sites.
  // Typical (MEDIAN) ionization loss [GeV] of the CUMULATIVE charged column of thickness x [X/X0],
  // same composite medium as elossMostProbable. Two corrections to the per-lump most-probable value,
  // each checked against the loss the simulation actually deposits:
  //  (1) the Landau family is STABLE under convolution, so the typical loss of a multi-slab column
  //      is the single-column law at the SUMMED thickness, not the sum of per-slab MPVs (the ln xi
  //      term is super-additive; per-lump charging under-states the barrel column by ~15 %);
  //  (2) a deterministic per-track correction can centre ONE statistic of the asymmetric loss
  //      distribution; the fit's own core estimator responds to the distribution CORE, i.e. the
  //      median class, and median - mode = (1.35578 + 0.22278) xi for a Landau (Koelbig-Schorr
  //      standard values; PDG RPP 34.2.9) -- not a tuned constant.
  // Callers charge per-node INCREMENTS T(X_cum + x_lump) - T(X_cum), which preserves the walk's
  // own lump placement (the split builder's two-point-per-gap quadrature) exactly.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double elossTypicalColumn(const TAcc& acc, double p, double xx0) {
    // Same medium and kinematic constants as elossMostProbable: elossMedium above.
    constexpr double m = elossMedium::kPionMass;
    constexpr double me = elossMedium::kElectronMass;
    constexpr double K = elossMedium::kK;
    constexpr double Z_A = elossMedium::kZ_A;
    constexpr double I = elossMedium::kI;
    constexpr double X0g = elossMedium::kX0g;
    constexpr double hwp = elossMedium::kHwp;
    constexpr double kLandauMedianMinusMode = elossMedium::kLandauMedianMinusMode;
    if (xx0 <= 0.)
      return 0.;
    const double E = alpaka::math::sqrt(acc, p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;
    const double xi = 0.5 * K * Z_A * (xx0 * X0g) / beta2;
    const double dhalf = alpaka::math::log(acc, (hwp / I) * alpaka::math::sqrt(acc, bg2)) - 0.5;
    const double delta = dhalf > 0. ? 2. * dhalf : 0.;
    const double bracket =
        alpaka::math::log(acc, 2. * me * bg2 / I) + alpaka::math::log(acc, xi / I) + 0.2 - beta2 - delta;
    const double dmp = xi * bracket;
    return (dmp > 0. ? dmp : 0.) + kLandauMedianMinusMode * xi;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_GeneralBrokenLine_h
