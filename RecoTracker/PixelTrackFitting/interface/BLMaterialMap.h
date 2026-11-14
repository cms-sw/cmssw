// The phi-averaged material of the tracker and the beam pipe on an r-z lattice, for the BrokenLine fit.
// Per cell: the radiation-length density rho [X0/cm] and the dE/dx triple (DeDx).
// One binary file per tracker geometry, format in BLMaterialMapFile.h. test/blMaterialMap regenerates them.
// The EventSetup selects the file by the ideal-geometry fingerprint (BLMaterialMapFingerprint.h).
#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMap_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMap_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

namespace blMaterialMap {
  // The lattice: r in [0, 125) cm, z in [-280, 280] cm, cells of 0.5 x 1 cm. It encloses the tracker.
  constexpr int kNR = 250;
  constexpr int kNZ = 560;
  constexpr float kDR = 0.5000f;   // cm
  constexpr float kDZ = 1.0000f;   // cm
  constexpr float kZMAX = 280.0f;  // cm; z in [-kZMAX, +kZMAX]
  constexpr int kSize = kNR * kNZ;

  // The dE/dx triple of a cell: the electron density rho_e = rho Z/A [mol/cm^3], and the rho_e-weighted means of
  // ln(I/eV) and of ln rho_e (Bragg additivity, PDG RPP 2026, Sec. 34.2.10, Eq. 34.14). All zero without material.
  struct DeDx {
    float rhoE;
    float lnI;
    float lnRhoE;
  };

  // The density lattice, then the dE/dx lattice, both r-major. The host copies it to the device as one object.
  struct Map {
    float rho[kSize];
    DeDx dedx[kSize];
  };
  static_assert(std::is_trivially_copyable_v<Map> && std::is_standard_layout_v<Map>);

  // Index of cell (r,z), or -1 outside the grid.
  constexpr inline int cellAt(float r, float z) {
    const float fr = r / kDR, fz = (z + kZMAX) / kDZ;
    if (!(fr >= 0.f && fr < float(kNR) && fz >= 0.f && fz < float(kNZ))) {
      return -1;
    }
    return int(fr) * kNZ + int(fz);
  }

  // Density [X0/cm] of cell c, 0 outside the grid.
  constexpr inline float rhoOfCell(const Map& map, int c) { return c < 0 ? 0.f : map.rho[c]; }

  // The dE/dx triple of cell c, zeros outside the grid.
  constexpr inline DeDx dedxOfCell(const Map& map, int c) { return c < 0 ? DeDx{0.f, 0.f, 0.f} : map.dedx[c]; }

  // The ionization column of a path: the integrals that fix its Landau loss. The struct is additive.
  struct ElossColumn {
    double e = 0.;       // int rho_e dl [mol/cm^2]
    double eLnI = 0.;    // int rho_e ln(I/eV) dl
    double eLnRho = 0.;  // int rho_e ln(rho_e) dl
    constexpr ElossColumn& operator+=(const ElossColumn& o) {
      e += o.e;
      eLnI += o.eLnI;
      eLnRho += o.eLnRho;
      return *this;
    }
    constexpr ElossColumn operator+(const ElossColumn& o) const {
      ElossColumn s = *this;
      s += o;
      return s;
    }
    constexpr ElossColumn operator*(double k) const { return ElossColumn{e * k, eLnI * k, eLnRho * k}; }
  };

  // Constants of the energy-loss law below. The law takes the accelerator, or alpaka::math::MathStdLib{} on the host.
  namespace elossMedium {
    // PDG RPP 2026: the pion mass (particle listings), the electron mass and K = 4 pi N_A r_e^2 m_e c^2 (Table 34.1).
    constexpr double kPionMass = 0.13957;           // pion mass [GeV]
    constexpr double kElectronMass = 0.5109989e-3;  // electron mass [GeV]
    constexpr double kK = 0.307075e-3;              // 0.307 MeV cm^2/mol -> GeV
    // The median minus the mode of the Landau distribution, in units of xi.
    // Mode from Bichsel, Rev. Mod. Phys. 60 (1988) 663, Eq. E1.
    constexpr double kLandauMedianMinusMode = 1.35578 + 0.22278;
    // hbar omega_p = 28.816 eV sqrt(rho Z/A) (PDG RPP 2026, Table 34.1) and the eV -> GeV shift, as logarithms.
    constexpr double kLnPlasmaEV = 3.360930788433600;  // ln(28.816)
    constexpr double kLnGeVinEV = 20.723265836946410;  // ln(1e9)
  }  // namespace elossMedium

  // Landau scale xi [GeV] and most-probable bracket of a column at total momentum p, for a pion. With xi = sum xi_i:
  //   Delta_mp = xi [ln(2 me beta^2 gamma^2 xi) + 0.2 - beta^2 - 2 <ln I> - delta(<Cbar>)],
  // <.> being the xi-weighted means of the column (PDG RPP 2026, Sec. 34.2.9, Eq. 34.12; Landau's constant 0.200
  // from Bichsel, Rev. Mod. Phys. 60 (1988) 663, Eq. E1). delta is Sternheimer's parametrization (PDG RPP 2026,
  // Sec. 34.2.5, Eq. 34.7), with the generic parameters of Sternheimer and Peierls, Phys. Rev. B3 (1971) 3681,
  // Eqs. 33-36.
  ALPAKA_NO_HOST_ACC_WARNING
  template <typename TMath>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool elossLandau(
      const TMath& math, double p, const ElossColumn& col, double& xi, double& bracket) {
    xi = 0.;
    bracket = 0.;
    if (!(col.e > 0.) || !(p > 0.)) {
      return false;
    }
    constexpr double kLn10 = 2.302585092994046;
    constexpr double twoLn10 = 2. * kLn10;  // the constant Sternheimer's parametrization rounds to 4.6052
    constexpr double m = elossMedium::kPionMass;
    constexpr double me = elossMedium::kElectronMass;
    constexpr double K = elossMedium::kK;
    const double E = alpaka::math::sqrt(math, p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;  // (beta*gamma)^2
    xi = 0.5 * K * col.e / beta2;
    const double lnIeV = col.eLnI / col.e;               // ln(I/eV)
    const double lnI = lnIeV - elossMedium::kLnGeVinEV;  // ln(I/GeV)
    const double cbar = 2. * (lnIeV - 0.5 * col.eLnRho / col.e - elossMedium::kLnPlasmaEV) + 1.;
    const bool soft = lnIeV < 2. * kLn10;  // I < 100 eV
    const double x1 = soft ? 2. : 3.;
    const double cbarLim = soft ? 3.681 : 5.215;
    const double x0 = (cbar < cbarLim) ? 0.2 : (0.326 * cbar - (soft ? 1.0 : 1.5));
    const double lnBg2 = alpaka::math::log(math, bg2);
    const double x = 0.5 * lnBg2 / kLn10;  // log10(betagamma)
    double delta = 0.;
    if (x >= x1) {
      delta = twoLn10 * x - cbar;
    } else if (x > x0) {
      const double a = (cbar - twoLn10 * x0) / ((x1 - x0) * (x1 - x0) * (x1 - x0));
      const double d = x1 - x;
      delta = twoLn10 * x - cbar + a * d * d * d;
    }
    // delta >= 0. The x >= x1 branch is negative for a large Cbar.
    if (delta < 0.) {
      delta = 0.;
    }
    // ln(2 me beta^2 gamma^2 / I) + ln(xi / I) = ln(2 me beta^2 gamma^2 xi) - 2 ln I
    bracket = alpaka::math::log(math, 2. * me * bg2 * xi) - 2. * lnI + 0.2 - beta2 - delta;
    return true;
  }

  // Median ionization loss [GeV] of the column at total momentum p [GeV]: xi * bracket + kLandauMedianMinusMode xi.
  ALPAKA_NO_HOST_ACC_WARNING
  template <typename TMath>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE double elossTypicalColumn(const TMath& math, double p, const ElossColumn& col) {
    double xi, bracket;
    if (!elossLandau(math, p, col, xi, bracket)) {
      return 0.;
    }
    const double dmp = xi * bracket;
    return (dmp > 0. ? dmp : 0.) + elossMedium::kLandauMedianMinusMode * xi;
  }

}  // namespace blMaterialMap
#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMap_h
