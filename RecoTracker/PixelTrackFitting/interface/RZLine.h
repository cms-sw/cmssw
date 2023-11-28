#ifndef RecoTracker_PixelTrackFitting_RZLine_h
#define RecoTracker_PixelTrackFitting_RZLine_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "CommonTools/Utils/interface/DynArray.h"

#include "CommonTools/Statistics/interface/LinearFit.h"

#include <vector>

class RZLine {
public:
  struct ErrZ2_tag {};

  /**
   * Constructor for containers of GlobalPoint, GlobalError, and bool
   *
   * @tparam P  Container of GlobalPoint
   * @tparam E  Container of GlobalError
   * @tparam B  Container of bool
   *
   * Container can be e.g. std::vector, std::array, or DynArray.
   *
   * Although for std::array use this constructor could be specialized
   * to use std::array instead of DynArray for temporary storage.
   */
  template <typename P, typename E, typename B>
  RZLine(const P& points, const E& errors, const B& isBarrel) {
    const size_t n = points.size();
    const size_t nSafe = n > 0 ? n : 1;
    float r[nSafe];
    float z[nSafe];
    float errZ2[nSafe];
    for (size_t i = 0; i < n; ++i) {
      const GlobalPoint& p = points[i];
      r[i] = p.perp();
      z[i] = p.z();
    }

    float simpleCot2 = n > 1 ? sqr((z[n - 1] - z[0]) / (r[n - 1] - r[0])) : 0.f;
    for (size_t i = 0; i < n; ++i) {
      errZ2[i] = (isBarrel[i]) ? errors[i].czz() : errors[i].rerr(points[i]) * simpleCot2;
    }

    calculate(r, z, errZ2, n);
  }

  /**
   * Constructor for std::vector of r, z, and z standard deviation
   */
  RZLine(const std::vector<float>& r, const std::vector<float>& z, const std::vector<float>& errZ) {
    const size_t n = errZ.size();
    float errZ2[n > 0 ? n : n + 1];
    for (size_t i = 0; i < n; ++i)
      errZ2[i] = sqr(errZ[i]);
    calculate(r.data(), z.data(), errZ2, n);
  }

  /**
   * Constructor for std::array of r, z, and z standard deviation
   */
  template <size_t N>
  RZLine(const std::array<float, N>& r, const std::array<float, N>& z, const std::array<float, N>& errZ) {
    std::array<float, N> errZ2;
    for (size_t i = 0; i < N; ++i)
      errZ2[i] = sqr(errZ[i]);
    calculate(r.data(), z.data(), errZ2.data(), N);
  }

  /**
   * Constructor for container of r, z, and z variance
   *
   * @tparam T  Container of float
   *
   * Container can be e.g. std::vector, std::array, or DynArray.
   *
   * The ErrZ2_tag parameter is used to distinguish this constructor
   * from other 3-parameter constructors.
   *
   * Passing variance is useful in cases where it is already available
   * to avoid making a square of a square root.
   */
  template <typename T>
  RZLine(const T& r, const T& z, const T& errZ2, ErrZ2_tag) {
    calculate(r.data(), z.data(), errZ2.data(), r.size());
  }

  float cotTheta() const { return cotTheta_; }
  float intercept() const { return intercept_; }
  float covss() const { return covss_; }
  float covii() const { return covii_; }
  float covsi() const { return covsi_; }

  float chi2() const { return chi2_; }

private:
  template <typename T>
  void calculate(const T* r, const T* z, const T* errZ2, size_t n) {
    //Have 0 and 1 cases return same value
    if (n < 2) [[unlikely]] {
      n = 0;
    }
    linearFit(r, z, n, errZ2, cotTheta_, intercept_, covss_, covii_, covsi_);
    chi2_ = 0.f;
    for (size_t i = 0; i < n; ++i) {
      chi2_ += sqr(((z[i] - intercept_) - cotTheta_ * r[i])) / errZ2[i];
    }
  }

  template <typename T>
  static constexpr T sqr(T t) {
    return t * t;
  }

  float cotTheta_;
  float intercept_;
  float covss_;
  float covii_;
  float covsi_;
  float chi2_;
};
#endif
