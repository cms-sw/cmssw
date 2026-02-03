#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_FitUtils_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_FitUtils_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "DataFormats/Math/interface/choleskyInversion.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/FitResult.h"
#include "RecoTracker/PixelTrackFitting/interface/FitUtils.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::riemannFit {
  using namespace ::riemannFit;

  template <typename C>
  ALPAKA_FN_ACC void printIt(C* m, const char* prefix = "") {
#ifdef RFIT_DEBUG
    for (uint r = 0; r < m->rows(); ++r) {
      for (uint c = 0; c < m->cols(); ++c) {
        printf("%s Matrix(%d,%d) = %g\n", prefix, r, c, (*m)(r, c));
      }
    }
#endif
  }

  /*!
    \brief raise to square.
   */
  template <typename T>
  constexpr T sqr(const T a) {
    return a * a;
  }

  /*!
    \brief Compute cross product of two 2D vector (assuming z component 0),
    returning z component of the result.
    \param a first 2D vector in the product.
    \param b second 2D vector in the product.
    \return z component of the cross product.
   */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double cross2D(const TAcc& acc, const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
  }

  /*!
   *  load error in CMSSW format to our formalism
   *  
   */
  template <alpaka::concepts::Acc TAcc, typename M6xNf, typename M2Nd>
  ALPAKA_FN_ACC void loadCovariance2D(const TAcc& acc, M6xNf const& ge, M2Nd& hits_cov) {
    // Index numerology:
    // i: index of the hits/point (0,..,3)
    // j: index of space component (x,y,z)
    // l: index of space components (x,y,z)
    // ge is always in sync with the index i and is formatted as:
    // ge[] ==> [xx, xy, yy, xz, yz, zz]
    // in (j,l) notation, we have:
    // ge[] ==> [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)]
    // so the index ge_idx corresponds to the matrix elements:
    // | 0  1  3 |
    // | 1  2  4 |
    // | 3  4  5 |
    constexpr uint32_t hits_in_fit = M6xNf::ColsAtCompileTime;
    for (uint32_t i = 0; i < hits_in_fit; ++i) {
      {
        constexpr uint32_t ge_idx = 0, j = 0, l = 0;
        hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      }
      {
        constexpr uint32_t ge_idx = 2, j = 1, l = 1;
        hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      }
      {
        constexpr uint32_t ge_idx = 1, j = 1, l = 0;
        hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
            ge.col(i)[ge_idx];
      }
    }
  }

  template <alpaka::concepts::Acc TAcc, typename M6xNf, typename M3xNd>
  ALPAKA_FN_ACC void loadCovariance(const TAcc& acc, M6xNf const& ge, M3xNd& hits_cov) {
    // Index numerology:
    // i: index of the hits/point (0,..,3)
    // j: index of space component (x,y,z)
    // l: index of space components (x,y,z)
    // ge is always in sync with the index i and is formatted as:
    // ge[] ==> [xx, xy, yy, xz, yz, zz]
    // in (j,l) notation, we have:
    // ge[] ==> [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)]
    // so the index ge_idx corresponds to the matrix elements:
    // | 0  1  3 |
    // | 1  2  4 |
    // | 3  4  5 |
    constexpr uint32_t hits_in_fit = M6xNf::ColsAtCompileTime;
    for (uint32_t i = 0; i < hits_in_fit; ++i) {
      {
        constexpr uint32_t ge_idx = 0, j = 0, l = 0;
        hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      }
      {
        constexpr uint32_t ge_idx = 2, j = 1, l = 1;
        hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      }
      {
        constexpr uint32_t ge_idx = 5, j = 2, l = 2;
        hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      }
      {
        constexpr uint32_t ge_idx = 1, j = 1, l = 0;
        hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
            ge.col(i)[ge_idx];
      }
      {
        constexpr uint32_t ge_idx = 3, j = 2, l = 0;
        hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
            ge.col(i)[ge_idx];
      }
      {
        constexpr uint32_t ge_idx = 4, j = 2, l = 1;
        hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
            ge.col(i)[ge_idx];
      }
    }
  }

  /*!
    \brief Transform circle parameter from (X0,Y0,R) to (phi,Tip,p_t) and
    consequently covariance matrix.
    \param circle_uvr parameter (X0,Y0,R), covariance matrix to
    be transformed and particle charge.
    \param B magnetic field in Gev/cm/c unit.
    \param error flag for errors computation.
  */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void par_uvrtopak(const TAcc& acc,
                                                   CircleFit& circle,
                                                   const double B,
                                                   const bool error) {
    Vector3d par_pak;
    const double temp0 = circle.par.head(2).squaredNorm();
    const double temp1 = alpaka::math::sqrt(acc, temp0);
    par_pak << alpaka::math::atan2(acc, circle.qCharge * circle.par(0), -circle.qCharge * circle.par(1)),
        circle.qCharge * (temp1 - circle.par(2)), circle.par(2) * B;
    if (error) {
      const double temp2 = sqr(circle.par(0)) * 1. / temp0;
      const double temp3 = 1. / temp1 * circle.qCharge;
      Matrix3d j4Mat;
      j4Mat << -circle.par(1) * temp2 * 1. / sqr(circle.par(0)), temp2 * 1. / circle.par(0), 0., circle.par(0) * temp3,
          circle.par(1) * temp3, -circle.qCharge, 0., 0., B;
      circle.cov = j4Mat * circle.cov * j4Mat.transpose();
    }
    circle.par = par_pak;
  }

  /*!
    \brief Transform circle parameter from (X0,Y0,R) to (phi,Tip,q/R) and
    consequently covariance matrix.
    \param circle_uvr parameter (X0,Y0,R), covariance matrix to
    be transformed and particle charge.
  */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fromCircleToPerigee(const TAcc& acc, CircleFit& circle) {
    Vector3d par_pak;
    const double temp0 = circle.par.head(2).squaredNorm();
    const double temp1 = alpaka::math::sqrt(acc, temp0);
    par_pak << alpaka::math::atan2(acc, circle.qCharge * circle.par(0), -circle.qCharge * circle.par(1)),
        circle.qCharge * (temp1 - circle.par(2)), circle.qCharge / circle.par(2);

    const double temp2 = sqr(circle.par(0)) * 1. / temp0;
    const double temp3 = 1. / temp1 * circle.qCharge;
    Matrix3d j4Mat;
    j4Mat << -circle.par(1) * temp2 * 1. / sqr(circle.par(0)), temp2 * 1. / circle.par(0), 0., circle.par(0) * temp3,
        circle.par(1) * temp3, -circle.qCharge, 0., 0., -circle.qCharge / (circle.par(2) * circle.par(2));
    circle.cov = j4Mat * circle.cov * j4Mat.transpose();

    circle.par = par_pak;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::riemannFit

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_FitUtils_h
