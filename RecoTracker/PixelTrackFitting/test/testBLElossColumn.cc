// Unit test of the ionization energy-loss model of the two fits: the Landau law of a COMPOSITE column
// (blMaterialMap::elossLandau / elossMostProbable / elossTypicalColumn), which takes its medium from the
// material map's dE/dx lattice instead of one composite constant.
//
//  (1) a pure-silicon column reproduces the textbook most-probable loss and density correction;
//  (2) the column struct is additive: the law on the sum of two lumps is the stable-law sum written by hand;
//  (3) fed the constants of the medium the fits used before, it reproduces the old law in the asymptotic
//      region where the old clamped density correction was right.
//
// References: PDG RPP ch. 34 (34.2.2 Landau most-probable loss, 34.2.5 density effect and Bragg additivity,
// 34.2.9 median minus mode); Sternheimer & Peierls, Phys. Rev. B 3 (1971) 3681; Bichsel, Rev. Mod. Phys. 60
// (1988) 663 (78-80 keV for 300 um of silicon at minimum ionization).

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

using blMaterialMap::ElossColumn;
using blMaterialMap::elossLandau;
using blMaterialMap::elossMostProbable;
using blMaterialMap::elossTypicalColumn;

namespace {
  int nFail = 0;

  void check(bool ok, const char* what, double got, double want, double tol) {
    std::printf("  %-52s got %12.6g  want %12.6g  tol %8.2g  %s\n", what, got, want, tol, ok ? "OK" : "FAIL");
    if (!ok)
      ++nFail;
  }

  void close(const char* what, double got, double want, double tol) {
    check(std::abs(got - want) <= tol, what, got, want, tol);
  }

  // an ElossColumn of thickness `dl` [cm] in one material
  ElossColumn lump(double rhoE, double IeV, double dl) {
    const double e = rhoE * dl;
    return ElossColumn{e, e * std::log(IeV), e * std::log(rhoE)};
  }

  // the density correction the law applied, recovered from its bracket
  double deltaOf(double p, const ElossColumn& col) {
    constexpr double m = blMaterialMap::elossMedium::kPionMass;
    constexpr double me = blMaterialMap::elossMedium::kElectronMass;
    double xi, bracket;
    if (!elossLandau(p, col, xi, bracket))
      return 0.;
    const double E = std::sqrt(p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double bg2 = beta2 * (E / m) * (E / m);
    const double I = std::exp(col.eLnI / col.e) * 1e-9;
    return std::log(2. * me * bg2 / I) + std::log(xi / I) + 0.2 - beta2 - bracket;
  }

  // the law the fits used before this test's model: one composite medium, (Z/A) X0g = 14.4 g/cm^2,
  // I = 122 eV, plasma energy 36.16 eV, density correction clamped to its high-betagamma limit.
  double oldTypical(double p, double xx0) {
    constexpr double m = 0.13957, me = 0.5109989e-3, K = 0.307075e-3;
    constexpr double Z_A = 0.5, I = 122e-9, X0g = 28.8, hwp = 36.16e-9;
    const double E = std::sqrt(p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double bg2 = beta2 * (E / m) * (E / m);
    const double xi = 0.5 * K * Z_A * (xx0 * X0g) / beta2;
    const double dhalf = std::log((hwp / I) * std::sqrt(bg2)) - 0.5;
    const double delta = dhalf > 0. ? 2. * dhalf : 0.;
    const double dmp = xi * (std::log(2. * me * bg2 / I) + std::log(xi / I) + 0.2 - beta2 - delta);
    return (dmp > 0. ? dmp : 0.) + blMaterialMap::elossMedium::kLandauMedianMinusMode * xi;
  }
}  // namespace

int main() {
  // (1) 300 um of silicon at p = 1 GeV (pion, betagamma 7.16): rho_e = rho Z/A = 2.329 x 0.4977 mol/cm^3,
  // I = 173 eV. Bichsel's measured most-probable loss for this thickness is 78-80 keV; the tabulated
  // Sternheimer density correction at this betagamma is 0.97, the generic parametrization gives 1.08.
  std::printf("(1) pure silicon, 300 um, 1 GeV pion\n");
  {
    const ElossColumn si = lump(2.329 * 0.4977, 173., 0.030);
    double xi, bracket;
    elossLandau(1.0, si, xi, bracket);
    close("xi [keV]", xi * 1e6, 5.45, 0.10);
    close("most probable loss [keV]", elossMostProbable(1.0, si) * 1e6, 78., 3.);
    close("density correction delta", deltaOf(1.0, si), 1.08, 0.03);
    close("median - mode [keV]",
          (elossTypicalColumn(1.0, si) - elossMostProbable(1.0, si)) * 1e6,
          blMaterialMap::elossMedium::kLandauMedianMinusMode * xi * 1e6,
          1e-9);
  }

  // (2) additivity: the law on the sum of a silicon and a copper lump equals the stable-law sum written out,
  //     Delta_mp = xi [ln(2 me beta^2 gamma^2 xi) + 0.2 - beta^2] - sum_i xi_i [2 ln I_i + delta],
  //     with xi = xi_Si + xi_Cu and delta that of the column's effective medium.
  std::printf("(2) additivity of the column (silicon + copper)\n");
  {
    constexpr double m = blMaterialMap::elossMedium::kPionMass;
    constexpr double me = blMaterialMap::elossMedium::kElectronMass;
    constexpr double K = blMaterialMap::elossMedium::kK;
    const double p = 2.0;
    const ElossColumn si = lump(2.329 * 0.4977, 173., 0.030);
    const ElossColumn cu = lump(8.960 * 0.4564, 322., 0.005);
    ElossColumn sum = si;
    sum += cu;
    const double E = std::sqrt(p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double bg2 = beta2 * (E / m) * (E / m);
    const double xiSi = 0.5 * K * si.e / beta2, xiCu = 0.5 * K * cu.e / beta2;
    const double xi = xiSi + xiCu;
    const double delta = deltaOf(p, sum);
    const double lnISi = si.eLnI / si.e - std::log(1e9);  // ln(I/GeV)
    const double lnICu = cu.eLnI / cu.e - std::log(1e9);
    const double byHand =
        xi * (std::log(2. * me * bg2 * xi) + 0.2 - beta2) - (xiSi * (2. * lnISi + delta) + xiCu * (2. * lnICu + delta));
    close("xi of the sum [keV]", xi * 1e6, 0.5 * K * sum.e / beta2 * 1e6, 1e-12);
    close("most probable loss [keV]", elossMostProbable(p, sum) * 1e6, byHand * 1e6, 1e-9);
  }

  // (3) regression on the medium the fits used before: at betagamma = 10^4 the old clamped density
  //     correction coincides with the Sternheimer parametrization, so the two laws must agree exactly.
  std::printf("(3) the previous composite medium, asymptotic region\n");
  {
    const double xx0 = 0.02;            // X/X0 of a typical pixel gap
    const double e = 0.5 * 28.8 * xx0;  // (Z/A) X0g x X/X0 [mol/cm^2]
    // the electron density whose plasma energy is the 36.16 eV that medium declared
    const double rhoE = (36.16 / 28.816) * (36.16 / 28.816);
    const ElossColumn old{e, e * std::log(122.), e * std::log(rhoE)};
    const double p = blMaterialMap::elossMedium::kPionMass * 1e4;
    close("typical loss vs the old law [keV]", elossTypicalColumn(p, old) * 1e6, oldTypical(p, xx0) * 1e6, 1e-7);
    // and at 1 GeV the new law charges LESS, because the old density correction was clamped to zero there
    check(elossTypicalColumn(1.0, old) < oldTypical(1.0, 0.02),
          "at 1 GeV the density correction lowers the loss",
          elossTypicalColumn(1.0, old) * 1e6,
          oldTypical(1.0, 0.02) * 1e6,
          0.);
  }

  std::printf("%s\n", nFail ? "FAILED" : "all checks passed");
  return nFail ? 1 : 0;
}
