#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h

// Root-free LDL^T factorization of a symmetric band matrix, and the in-place solve of M x = rhs
// (Blobel, NIM A566 (2006) 14, Sec. 3).
// Storage: Mb[packedBandIndex<kBand>(i, p, lstride)] holds M(i, i-p), p = 0..min(i, kBand). bandFactor overwrites it
// with D_i at offset 0 and L(i, i-p) at offset p. kBand is the band half-width, lstride the element stride.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::bandedSolve {

  // The position of element (row i, offset p) in the lower-packed storage.
  template <int kBand, typename I>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr I packedBandIndex(I i, I p, I lstride) {
    return (i * I(kBand + 1) + p) * lstride;
  }

  template <int kBand>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void bandFactor(double* Mb, int n, int lstride) {
    // element (row i, offset p), addressed from the start of row i
    auto M = [=](int i, int p) -> double& { return (Mb + packedBandIndex<kBand>(i, 0, lstride))[p * lstride]; };
    for (int j = 0; j < n; ++j) {
      double Dj = M(j, 0);
      const int pmax = (j < kBand) ? j : kBand;
      for (int p = 1; p <= pmax; ++p) {
        Dj -= M(j, p) * M(j, p) * M(j - p, 0);
      }
      M(j, 0) = Dj;
      const int imax = (j + kBand < n - 1) ? (j + kBand) : (n - 1);
      for (int i = j + 1; i <= imax; ++i) {
        double sum = M(i, i - j);  // M(i, j)
        const int kmin = (i - kBand > 0) ? (i - kBand) : 0;
        for (int k = kmin; k < j; ++k) {
          sum -= M(i, i - k) * M(j, j - k) * M(k, 0);
        }
        M(i, i - j) = sum / Dj;  // L(i, j)
      }
    }
  }

  // Solve M x = rhs in place with the factors of bandFactor. Mb and x must not overlap.
  template <int kBand>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void bandSolveInPlace(const double* __restrict__ Mb,
                                                       double* __restrict__ x,
                                                       int n,
                                                       int lstride) {
    auto M = [=](int i, int p) -> const double& { return (Mb + packedBandIndex<kBand>(i, 0, lstride))[p * lstride]; };
    auto X = [=](int i) -> double& { return x[i * lstride]; };
    for (int i = 0; i < n; ++i) {  // forward: L y = rhs
      const int pmax = (i < kBand) ? i : kBand;
      double si = X(i);
      for (int p = 1; p <= pmax; ++p) {
        si -= M(i, p) * X(i - p);
      }
      X(i) = si;
    }
    for (int i = 0; i < n; ++i) {  // diagonal: D z = y
      X(i) /= M(i, 0);
    }
    for (int i = n - 1; i >= 0; --i) {  // back: L^T x = z
      const int kmax = (i + kBand < n - 1) ? (i + kBand) : (n - 1);
      double si = X(i);
      for (int k = i + 1; k <= kmax; ++k) {
        si -= M(k, k - i) * X(k);
      }
      X(i) = si;
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::bandedSolve

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_BandedSolve_h
