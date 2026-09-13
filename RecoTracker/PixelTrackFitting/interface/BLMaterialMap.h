// Phi-averaged material of the Tracker and the beam pipe on a 0.5 cm radial lattice, two lattices in one
// array: the radiation-length density rho(r,z) [X0/cm], whose integral along a track segment is its X/X0,
// and the dE/dx triple, the electron density and the two log-means the Landau loss of a composite column
// needs. One table per geometry is compiled in (src/BLMaterialMap<geometry>.cc), and test/blMaterialMap/
// regenerates it. The readers take the same pointer: the device fits get it from the EventSetup, the unit
// tests read the compiled-in array directly.
#ifndef RecoTracker_PixelTrackFitting_BLMaterialMap_h
#define RecoTracker_PixelTrackFitting_BLMaterialMap_h

#include <cmath>

namespace blMaterialMap {
  constexpr int kNR = 250;
  constexpr int kNZ = 560;
  constexpr float kDR = 0.5000f;   // cm
  constexpr float kDZ = 1.0000f;   // cm
  constexpr float kZMAX = 280.0f;  // cm; z in [-kZMAX, +kZMAX]
  constexpr int kSize = kNR * kNZ;
  // floats per cell of the dE/dx lattice: (rho_e [mol/cm^3], <ln(I/eV)>, <ln rho_e>)
  constexpr int kDedxStride = 3;
  // floats in the compiled-in table and in the buffer the EventSetup uploads: the whole density lattice,
  // then the whole dE/dx lattice. The two are kept in blocks rather than interleaved so that the readers
  // that want only the density -- the extender's march, which walks long z runs -- keep their cache lines
  // dense; a cell-interleaved table costs them more than it saves the fits.
  constexpr int kBufferFloats = (1 + kDedxStride) * kSize;
  // Host pointer to the compiled-in table (src/BLMaterialMap<geometry>.cc).
  const float* blMaterialMapData();
  // The dE/dx lattice belonging to a density lattice: contiguous with it in the compiled-in table and in the
  // uploaded buffer alike, so one pointer carries both and no reader needs a second one.
  constexpr inline const float* dedxOf(const float* rho) { return rho + kSize; }

  // Index of cell (r,z) in the density lattice, or -1 outside the grid.
  constexpr inline int cellAt(float r, float z) {
    int ir = int(r / kDR), iz = int((z + kZMAX) / kDZ);
    if (ir < 0 || ir >= kNR || iz < 0 || iz >= kNZ)
      return -1;
    return ir * kNZ + iz;
  }

  // Local density [X0/cm] at (r,z), 0 outside the grid; rho is a host array or a device buffer.
  // constexpr so it is callable from device code.
  constexpr inline float rhoAt(const float* rho, float r, float z) {
    const int c = cellAt(r, z);
    return c < 0 ? 0.f : rho[c];
  }

  // The cell's dE/dx triple at (r,z): the electron density rho_e = rho_mass Z/A [mol/cm^3] (the Landau xi per
  // cm of path), and the rho_e-weighted means of ln(I/eV) and of ln rho_e over the cell (Bragg additivity,
  // PDG RPP 34.2.5). Zeros outside the grid and where the cell holds no material.
  constexpr inline void dedxAt(const float* dedx, float r, float z, float& rhoE, float& lnI, float& lnRhoE) {
    const int c = cellAt(r, z);
    if (c < 0) {
      rhoE = lnI = lnRhoE = 0.f;
      return;
    }
    rhoE = dedx[kDedxStride * c];
    lnI = dedx[kDedxStride * c + 1];
    lnRhoE = dedx[kDedxStride * c + 2];
  }

  // The ionization column of a path: the three integrals int rho_e dl, int rho_e <ln I> dl and
  // int rho_e <ln rho_e> dl that fix the Landau loss of the whole path; the effective medium of the column
  // is lnI = eLnI/e, lnRho = eLnRho/e. All three are first moments, so the struct is additive: the column of
  // two lumps is the sum, and a share of a lump is the scaled struct.
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

  // Constants of the two ionization energy-loss laws below (elossMostProbable, elossTypicalColumn): both
  // evaluate the same Landau xi on the same column and differ only in which statistic of the loss
  // distribution they return. Every material quantity comes from the column the material map's dE/dx lattice
  // produced for the path actually walked; there is no composite constant medium.
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
  // which needs the xi-weighted means of ln I and of ln rho_e that ElossColumn carries. delta follows
  // Sternheimer's parametrization with the generic Sternheimer-Peierls parameters (PDG RPP 34.2.5).
  inline bool elossLandau(double p, const ElossColumn& col, double& xi, double& bracket) {
    xi = 0.;
    bracket = 0.;
    if (!(col.e > 0.) || !(p > 0.))
      return false;
    constexpr double kLn10 = 2.302585092994046;
    constexpr double twoLn10 = 2. * kLn10;  // the constant Sternheimer's parametrization rounds to 4.6052
    constexpr double m = elossMedium::kPionMass;
    constexpr double me = elossMedium::kElectronMass;
    constexpr double K = elossMedium::kK;
    const double E = std::sqrt(p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;  // (beta*gamma)^2
    xi = 0.5 * K * col.e / beta2;
    // The column's effective medium enters only through ln I and, via the plasma energy, ln rho_e, and the
    // column already carries both as logarithms: I and hbar omega_p are never formed.
    const double lnIeV = col.eLnI / col.e;               // ln(I/eV)
    const double lnI = lnIeV - elossMedium::kLnGeVinEV;  // ln(I/GeV)
    const double cbar = 2. * (lnIeV - 0.5 * col.eLnRho / col.e - elossMedium::kLnPlasmaEV) + 1.;
    const bool soft = lnIeV < 2. * kLn10;  // I < 100 eV
    const double x1 = soft ? 2. : 3.;
    const double cbarLim = soft ? 3.681 : 5.215;
    const double x0 = (cbar < cbarLim) ? 0.2 : (0.326 * cbar - (soft ? 1.0 : 1.5));
    const double x = 0.5 * std::log(bg2) / kLn10;  // log10(betagamma)
    double delta = 0.;
    if (x >= x1) {
      delta = twoLn10 * x - cbar;
    } else if (x > x0) {
      const double a = (cbar - twoLn10 * x0) / ((x1 - x0) * (x1 - x0) * (x1 - x0));
      const double d = x1 - x;
      delta = twoLn10 * x - cbar + a * d * d * d;
    }
    // ln(2 me beta^2 gamma^2 / I) + ln(xi / I) = ln(2 me beta^2 gamma^2 xi) - 2 ln I
    bracket = std::log(2. * me * bg2 * xi) - 2. * lnI + 0.2 - beta2 - delta;
    return true;
  }

  // Most-probable (Landau) ionization energy loss [GeV] of the column, at total momentum p [GeV]. The
  // correction has to remove the loss of the typical track, not the unrestricted Bethe-Bloch mean, which
  // includes the Landau delta-ray tail. MP is not additive across sub-columns (the ln xi term).
  inline double elossMostProbable(double p, const ElossColumn& col) {
    double xi, bracket;
    if (!elossLandau(p, col, xi, bracket))
      return 0.;
    const double dmp = xi * bracket;
    return dmp > 0. ? dmp : 0.;
  }

  // Typical (median) ionization loss [GeV] of the CUMULATIVE column, selected over the per-lump most-probable
  // law by the runtime flag elossCumulative. The Landau family is stable under convolution, so the typical
  // loss of a multi-lump column is the single-column law at the SUMMED column, not the sum of per-lump MPVs,
  // and median - mode = (1.35578 + 0.22278) xi for a Landau (PDG RPP 34.2.9). Callers charge per-node
  // increments T(col_cum + col_lump) - T(col_cum).
  inline double elossTypicalColumn(double p, const ElossColumn& col) {
    double xi, bracket;
    if (!elossLandau(p, col, xi, bracket))
      return 0.;
    const double dmp = xi * bracket;
    return (dmp > 0. ? dmp : 0.) + elossMedium::kLandauMedianMinusMode * xi;
  }
}  // namespace blMaterialMap
#endif
