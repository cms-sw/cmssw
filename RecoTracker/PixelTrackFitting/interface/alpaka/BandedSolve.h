#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h

// Root-free LDL^T factorization of a symmetric band matrix and in-place application of its inverse
// to a right-hand side, for the device fits.
//
// Lower-packed storage: Mb[(i*(kBand+1)+p)*lstride] = M(i, i-p), p = 0..min(i,kBand). bandFactor
// overwrites Mb with its LDL^T (Mb[..+0] = D_i, Mb[..+p] = L(i,i-p)). Band half-width kBand (GBL:
// kGblBand=5; fast-BL: kBand=2); element stride lstride (1 = contiguous, lane count for
// [element][lane] fit scratch).

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine {

  template <int kBand>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void bandFactor(double* Mb, int n, int lstride = 1) {
    const int rstride = (kBand + 1) * lstride;  // physical stride between packed rows
    for (int j = 0; j < n; ++j) {
      double* Mj = Mb + j * rstride;
      double Dj = Mj[0];
      const int pmax = (j < kBand) ? j : kBand;
      for (int p = 1; p <= pmax; ++p)
        Dj -= Mj[p * lstride] * Mj[p * lstride] * Mb[(j - p) * rstride];
      Mj[0] = Dj;
      const int imax = (j + kBand < n - 1) ? (j + kBand) : (n - 1);
      for (int i = j + 1; i <= imax; ++i) {
        double* Mi = Mb + i * rstride;
        double sum = Mi[(i - j) * lstride];  // A(i, j)
        const int kmin = (i - kBand > 0) ? (i - kBand) : 0;
        for (int k = kmin; k < j; ++k)
          sum -= Mi[(i - k) * lstride] * Mj[(j - k) * lstride] * Mb[k * rstride];
        Mi[(i - j) * lstride] = sum / Dj;  // L(i, j)
      }
    }
  }

  // iFirst/iStop are the sparsity window of a leading-zero right-hand side, skipping steps whose
  // result is provably the +0 already in x: the caller guarantees x[i] == +0 for i < iFirst and reads
  // only x[i >= iStop], and back substitution reads only strictly higher indices. Defaults 0/0 run the
  // full solve.
  template <int kBand>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void bandSolveInPlace(
      const double* Mb, double* x, int n, int lstride = 1, int iFirst = 0, int iStop = 0) {
    const int rstride = (kBand + 1) * lstride;  // physical stride between packed rows
    for (int i = iFirst; i < n; ++i) {          // forward: L y = rhs
      const double* Mi = Mb + i * rstride;
      const int pmax = (i < kBand) ? i : kBand;
      double si = x[i * lstride];
      for (int p = 1; p <= pmax; ++p)
        si -= Mi[p * lstride] * x[(i - p) * lstride];
      x[i * lstride] = si;
    }
    const int iDiag = iFirst;
    for (int i = iDiag; i < n; ++i)  // diagonal: D z = y
      x[i * lstride] /= Mb[i * rstride];
    for (int i = n - 1; i >= iStop; --i) {  // back: L^T x = z
      const int kmax = (i + kBand < n - 1) ? (i + kBand) : (n - 1);
      double si = x[i * lstride];
      for (int k = i + 1; k <= kmax; ++k)
        si -= Mb[k * rstride + (k - i) * lstride] * x[k * lstride];
      x[i * lstride] = si;
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h
