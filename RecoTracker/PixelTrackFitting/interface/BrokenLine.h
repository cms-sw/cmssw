#ifndef RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h
#define RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h

#include <Eigen/Eigenvalues>

#include "RecoTracker/PixelTrackFitting/interface/FitUtils.h"

namespace brokenline {

  //!< Karimäki's parameters: (phi, d, k=1/R)
  /*!< covariance matrix: \n
    |cov(phi,phi)|cov( d ,phi)|cov( k ,phi)| \n
    |cov(phi, d )|cov( d , d )|cov( k , d )| \n
    |cov(phi, k )|cov( d , k )|cov( k , k )| \n
    as defined in Karimäki V., 1990, Effective circle fitting for particle trajectories, 
    Nucl. Instr. and Meth. A305 (1991) 187.
  */
  using karimaki_circle_fit = riemannFit::CircleFit;

  /*!
    \brief data needed for the Broken Line fit procedure.
  */
  template <int n>
  struct PreparedBrokenLineData {
    int qCharge;                          //!< particle charge
    riemannFit::Matrix2xNd<n> radii;      //!< xy data in the system in which the pre-fitted center is the origin
    riemannFit::VectorNd<n> sTransverse;  //!< total distance traveled in the transverse plane
                                          //   starting from the pre-fitted closest approach
    riemannFit::VectorNd<n> sTotal;       //!< total distance traveled (three-dimensional)
    riemannFit::VectorNd<n> zInSZplane;   //!< orthogonal coordinate to the pre-fitted line in the sz plane
    riemannFit::VectorNd<n> varBeta;      //!< kink angles in the SZ plane
  };

  /*!
    \brief Computes the Coulomb multiple scattering variance of the planar angle.
    
    \param length length of the track in the material.
    \param bField magnetic field in Gev/cm/c.
    \param radius radius of curvature (needed to evaluate p).
    \param layer denotes which of the four layers of the detector is the endpoint of the 
   *             multiple scattered track. For example, if Layer=3, then the particle has 
   *             just gone through the material between the second and the third layer.
    
    \todo add another Layer variable to identify also the start point of the track, 
   *      so if there are missing hits or multiple hits, the part of the detector that 
   *      the particle has traversed can be exactly identified.
    
    \warning the formula used here assumes beta=1, and so neglects the dependence 
   *         of theta_0 on the mass of the particle at fixed momentum.
    
    \return the variance of the planar angle ((theta_0)^2 /3).
  */
  __host__ __device__ inline double multScatt(
      const double& length, const double bField, const double radius, int layer, double slope) {
    // limit R to 20GeV...
    auto pt2 = std::min(20., bField * radius);
    pt2 *= pt2;
    constexpr double inv_X0 = 0.06 / 16.;  //!< inverse of radiation length of the material in cm
    //if(Layer==1) XXI_0=0.06/16.;
    // else XXI_0=0.06/16.;
    //XX_0*=1;

    //! number between 1/3 (uniform material) and 1 (thin scatterer) to be manually tuned
    constexpr double geometry_factor = 0.7;
    constexpr double fact = geometry_factor * riemannFit::sqr(13.6 / 1000.);
    return fact / (pt2 * (1. + riemannFit::sqr(slope))) * (std::abs(length) * inv_X0) *
           riemannFit::sqr(1. + 0.038 * log(std::abs(length) * inv_X0));
  }

  /*!
    \brief Computes the 2D rotation matrix that transforms the line y=slope*x into the line y=0.
    
    \param slope tangent of the angle of rotation.
    
    \return 2D rotation matrix.
  */
  __host__ __device__ inline riemannFit::Matrix2d rotationMatrix(double slope) {
    riemannFit::Matrix2d rot;
    rot(0, 0) = 1. / sqrt(1. + riemannFit::sqr(slope));
    rot(0, 1) = slope * rot(0, 0);
    rot(1, 0) = -rot(0, 1);
    rot(1, 1) = rot(0, 0);
    return rot;
  }

  /*!
    \brief Changes the Karimäki parameters (and consequently their covariance matrix) under a 
   *       translation of the coordinate system, such that the old origin has coordinates (x0,y0) 
   *       in the new coordinate system. The formulas are taken from Karimäki V., 1990, Effective 
   *       circle fitting for particle trajectories, Nucl. Instr. and Meth. A305 (1991) 187.
    
    \param circle circle fit in the old coordinate system. circle.par(0) is phi, circle.par(1) is d and circle.par(2) is rho. 
    \param x0 x coordinate of the translation vector.
    \param y0 y coordinate of the translation vector.
    \param jacobian passed by reference in order to save stack.
  */
  __host__ __device__ inline void translateKarimaki(karimaki_circle_fit& circle,
                                                    double x0,
                                                    double y0,
                                                    riemannFit::Matrix3d& jacobian) {
    // Avoid multiple access to the circle.par vector.
    using scalar = std::remove_reference<decltype(circle.par(0))>::type;
    scalar phi = circle.par(0);
    scalar dee = circle.par(1);
    scalar rho = circle.par(2);

    // Avoid repeated trig. computations
    scalar sinPhi = sin(phi);
    scalar cosPhi = cos(phi);

    // Intermediate computations for the circle parameters
    scalar deltaPara = x0 * cosPhi + y0 * sinPhi;
    scalar deltaOrth = x0 * sinPhi - y0 * cosPhi + dee;
    scalar tempSmallU = 1 + rho * dee;
    scalar tempC = -rho * y0 + tempSmallU * cosPhi;
    scalar tempB = rho * x0 + tempSmallU * sinPhi;
    scalar tempA = 2. * deltaOrth + rho * (riemannFit::sqr(deltaOrth) + riemannFit::sqr(deltaPara));
    scalar tempU = sqrt(1. + rho * tempA);

    // Intermediate computations for the error matrix transform
    scalar xi = 1. / (riemannFit::sqr(tempB) + riemannFit::sqr(tempC));
    scalar tempV = 1. + rho * deltaOrth;
    scalar lambda = (0.5 * tempA) / (riemannFit::sqr(1. + tempU) * tempU);
    scalar mu = 1. / (tempU * (1. + tempU)) + rho * lambda;
    scalar zeta = riemannFit::sqr(deltaOrth) + riemannFit::sqr(deltaPara);
    jacobian << xi * tempSmallU * tempV, -xi * riemannFit::sqr(rho) * deltaOrth, xi * deltaPara,
        2. * mu * tempSmallU * deltaPara, 2. * mu * tempV, mu * zeta - lambda * tempA, 0, 0, 1.;

    // translated circle parameters
    // phi
    circle.par(0) = atan2(tempB, tempC);
    // d
    circle.par(1) = tempA / (1 + tempU);
    // rho after translation. It is invariant, so noop
    // circle.par(2)= rho;

    // translated error matrix
    circle.cov = jacobian * circle.cov * jacobian.transpose();
  }

  /*!
    \brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.
    
    \param hits hits coordinates.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param results PreparedBrokenLineData to be filled (see description of PreparedBrokenLineData).
  */
  template <typename M3xN, typename V4, int n>
  __host__ __device__ inline void prepareBrokenLineData(const M3xN& hits,
                                                        const V4& fast_fit,
                                                        const double bField,
                                                        PreparedBrokenLineData<n>& results) {
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
    results.qCharge = riemannFit::cross2D(dVec, eVec) > 0 ? -1 : 1;

    const double slope = -results.qCharge / fast_fit(3);

    riemannFit::Matrix2d rotMat = rotationMatrix(slope);

    // calculate radii and s
    results.radii = hits.block(0, 0, 2, n) - fast_fit.head(2) * riemannFit::MatrixXd::Constant(1, n, 1);
    eVec = -fast_fit(2) * fast_fit.head(2) / fast_fit.head(2).norm();
    for (u_int i = 0; i < n; i++) {
      dVec = results.radii.block(0, i, 2, 1);
      results.sTransverse(i) = results.qCharge * fast_fit(2) *
                               atan2(riemannFit::cross2D(dVec, eVec), dVec.dot(eVec));  // calculates the arc length
    }
    riemannFit::VectorNd<n> zVec = hits.block(2, 0, 1, n).transpose();

    //calculate sTotal and zVec
    riemannFit::Matrix2xNd<n> pointsSZ = riemannFit::Matrix2xNd<n>::Zero();
    for (u_int i = 0; i < n; i++) {
      pointsSZ(0, i) = results.sTransverse(i);
      pointsSZ(1, i) = zVec(i);
      pointsSZ.block(0, i, 2, 1) = rotMat * pointsSZ.block(0, i, 2, 1);
    }
    results.sTotal = pointsSZ.block(0, 0, 1, n).transpose();
    results.zInSZplane = pointsSZ.block(1, 0, 1, n).transpose();

    //calculate varBeta
    results.varBeta(0) = results.varBeta(n - 1) = 0;
    for (u_int i = 1; i < n - 1; i++) {
      results.varBeta(i) = multScatt(results.sTotal(i + 1) - results.sTotal(i), bField, fast_fit(2), i + 2, slope) +
                           multScatt(results.sTotal(i) - results.sTotal(i - 1), bField, fast_fit(2), i + 1, slope);
    }
  }

  /*!
    \brief Computes the n-by-n band matrix obtained minimizing the Broken Line's cost function w.r.t u. 
   *       This is the whole matrix in the case of the line fit and the main n-by-n block in the case 
   *       of the circle fit.
    
    \param weights weights of the first part of the cost function, the one with the measurements 
   *         and not the angles (\sum_{i=1}^n w*(y_i-u_i)^2).
    \param sTotal total distance traveled by the particle from the pre-fitted closest approach.
    \param varBeta kink angles' variance.
    
    \return the n-by-n matrix of the linear system
  */
  template <int n>
  __host__ __device__ inline riemannFit::MatrixNd<n> matrixC_u(const riemannFit::VectorNd<n>& weights,
                                                               const riemannFit::VectorNd<n>& sTotal,
                                                               const riemannFit::VectorNd<n>& varBeta) {
    riemannFit::MatrixNd<n> c_uMat = riemannFit::MatrixNd<n>::Zero();
    for (u_int i = 0; i < n; i++) {
      c_uMat(i, i) = weights(i);
      if (i > 1)
        c_uMat(i, i) += 1. / (varBeta(i - 1) * riemannFit::sqr(sTotal(i) - sTotal(i - 1)));
      if (i > 0 && i < n - 1)
        c_uMat(i, i) +=
            (1. / varBeta(i)) * riemannFit::sqr((sTotal(i + 1) - sTotal(i - 1)) /
                                                ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
      if (i < n - 2)
        c_uMat(i, i) += 1. / (varBeta(i + 1) * riemannFit::sqr(sTotal(i + 1) - sTotal(i)));

      if (i > 0 && i < n - 1)
        c_uMat(i, i + 1) =
            1. / (varBeta(i) * (sTotal(i + 1) - sTotal(i))) *
            (-(sTotal(i + 1) - sTotal(i - 1)) / ((sTotal(i + 1) - sTotal(i)) * (sTotal(i) - sTotal(i - 1))));
      if (i < n - 2)
        c_uMat(i, i + 1) +=
            1. / (varBeta(i + 1) * (sTotal(i + 1) - sTotal(i))) *
            (-(sTotal(i + 2) - sTotal(i)) / ((sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i))));

      if (i < n - 2)
        c_uMat(i, i + 2) = 1. / (varBeta(i + 1) * (sTotal(i + 2) - sTotal(i + 1)) * (sTotal(i + 1) - sTotal(i)));

      c_uMat(i, i) *= 0.5;
    }
    return c_uMat + c_uMat.transpose();
  }

  /*!
    \brief A very fast helix fit.
    
    \param hits the measured hits.
    
    \return (X0,Y0,R,tan(theta)).
    
    \warning sign of theta is (intentionally, for now) mistaken for negative charges.
  */

  template <typename M3xN, typename V4>
  __host__ __device__ inline void fastFit(const M3xN& hits, V4& result) {
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

    auto tmp = 0.5 / riemannFit::cross2D(c, a);
    result(0) = hits(0, 0) - (a(1) * c.squaredNorm() + c(1) * a.squaredNorm()) * tmp;
    result(1) = hits(1, 0) + (a(0) * c.squaredNorm() + c(0) * a.squaredNorm()) * tmp;
    // check Wikipedia for these formulas

    result(2) = sqrt(a.squaredNorm() * b.squaredNorm() * c.squaredNorm()) / (2. * std::abs(riemannFit::cross2D(b, a)));
    // Using Math Olympiad's formula R=abc/(4A)

    const riemannFit::Vector2d d = hits.block(0, 0, 2, 1) - result.head(2);
    const riemannFit::Vector2d e = hits.block(0, n - 1, 2, 1) - result.head(2);

    result(3) = result(2) * atan2(riemannFit::cross2D(d, e), d.dot(e)) / (hits(2, n - 1) - hits(2, 0));
    // ds/dz slope between last and first point
  }

  /*!
    \brief Performs the Broken Line fit in the curved track case (that is, the fit 
   *       parameters are the interceptions u and the curvature correction \Delta\kappa).
    
    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineData.
    \param circle_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (phi, d, k); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    
    \details The function implements the steps 2 and 3 of the Broken Line fit 
   *         with the curvature correction.\n
   * The step 2 is the least square fit, done by imposing the minimum constraint on 
   * the cost function and solving the consequent linear system. It determines the 
   * fitted parameters u and \Delta\kappa and their covariance matrix.
   * The step 3 is the correction of the fast pre-fitted parameters for the innermost 
   * part of the track. It is first done in a comfortable coordinate system (the one 
   * in which the first hit is the origin) and then the parameters and their 
   * covariance matrix are transformed to the original coordinate system.
  */
  template <typename M3xN, typename M6xN, typename V4, int n>
  __host__ __device__ inline void circleFit(const M3xN& hits,
                                            const M6xN& hits_ge,
                                            const V4& fast_fit,
                                            const double bField,
                                            PreparedBrokenLineData<n>& data,
                                            karimaki_circle_fit& circle_results) {
    circle_results.qCharge = data.qCharge;
    auto& radii = data.radii;
    const auto& sTransverse = data.sTransverse;
    const auto& sTotal = data.sTotal;
    auto& zInSZplane = data.zInSZplane;
    auto& varBeta = data.varBeta;
    const double slope = -circle_results.qCharge / fast_fit(3);
    varBeta *= 1. + riemannFit::sqr(slope);  // the kink angles are projected!

    for (u_int i = 0; i < n; i++) {
      zInSZplane(i) = radii.block(0, i, 2, 1).norm() - fast_fit(2);
    }

    riemannFit::Matrix2d vMat;           // covariance matrix
    riemannFit::VectorNd<n> weightsVec;  // weights
    riemannFit::Matrix2d rotMat;         // rotation matrix point by point
    for (u_int i = 0; i < n; i++) {
      vMat(0, 0) = hits_ge.col(i)[0];               // x errors
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];  // cov_xy
      vMat(1, 1) = hits_ge.col(i)[2];               // y errors
      rotMat = rotationMatrix(-radii(0, i) / radii(1, i));
      weightsVec(i) =
          1. / ((rotMat * vMat * rotMat.transpose())(1, 1));  // compute the orthogonal weight point by point
    }

    riemannFit::VectorNplusONEd<n> r_uVec;
    r_uVec(n) = 0;
    for (u_int i = 0; i < n; i++) {
      r_uVec(i) = weightsVec(i) * zInSZplane(i);
    }

    riemannFit::MatrixNplusONEd<n> c_uMat;
    c_uMat.block(0, 0, n, n) = matrixC_u(weightsVec, sTransverse, varBeta);
    c_uMat(n, n) = 0;
    //add the border to the c_uMat matrix
    for (u_int i = 0; i < n; i++) {
      c_uMat(i, n) = 0;
      if (i > 0 && i < n - 1) {
        c_uMat(i, n) +=
            -(sTransverse(i + 1) - sTransverse(i - 1)) * (sTransverse(i + 1) - sTransverse(i - 1)) /
            (2. * varBeta(i) * (sTransverse(i + 1) - sTransverse(i)) * (sTransverse(i) - sTransverse(i - 1)));
      }
      if (i > 1) {
        c_uMat(i, n) +=
            (sTransverse(i) - sTransverse(i - 2)) / (2. * varBeta(i - 1) * (sTransverse(i) - sTransverse(i - 1)));
      }
      if (i < n - 2) {
        c_uMat(i, n) +=
            (sTransverse(i + 2) - sTransverse(i)) / (2. * varBeta(i + 1) * (sTransverse(i + 1) - sTransverse(i)));
      }
      c_uMat(n, i) = c_uMat(i, n);
      if (i > 0 && i < n - 1)
        c_uMat(n, n) += riemannFit::sqr(sTransverse(i + 1) - sTransverse(i - 1)) / (4. * varBeta(i));
    }

#ifdef CPP_DUMP
    std::cout << "CU5\n" << c_uMat << std::endl;
#endif
    riemannFit::MatrixNplusONEd<n> iMat;
    math::cholesky::invert(c_uMat, iMat);
#ifdef CPP_DUMP
    std::cout << "I5\n" << iMat << std::endl;
#endif

    riemannFit::VectorNplusONEd<n> uVec = iMat * r_uVec;  // obtain the fitted parameters by solving the linear system

    // compute (phi, d_ca, k) in the system in which the midpoint of the first two corrected hits is the origin...

    radii.block(0, 0, 2, 1) /= radii.block(0, 0, 2, 1).norm();
    radii.block(0, 1, 2, 1) /= radii.block(0, 1, 2, 1).norm();

    riemannFit::Vector2d dVec = hits.block(0, 0, 2, 1) + (-zInSZplane(0) + uVec(0)) * radii.block(0, 0, 2, 1);
    riemannFit::Vector2d eVec = hits.block(0, 1, 2, 1) + (-zInSZplane(1) + uVec(1)) * radii.block(0, 1, 2, 1);
    auto eMinusd = eVec - dVec;
    auto eMinusd2 = eMinusd.squaredNorm();
    auto tmp1 = 1. / eMinusd2;
    auto tmp2 = sqrt(riemannFit::sqr(fast_fit(2)) - 0.25 * eMinusd2);

    circle_results.par << atan2(eMinusd(1), eMinusd(0)), circle_results.qCharge * (tmp2 - fast_fit(2)),
        circle_results.qCharge * (1. / fast_fit(2) + uVec(n));

    tmp2 = 1. / tmp2;

    riemannFit::Matrix3d jacobian;
    jacobian << (radii(1, 0) * eMinusd(0) - eMinusd(1) * radii(0, 0)) * tmp1,
        (radii(1, 1) * eMinusd(0) - eMinusd(1) * radii(0, 1)) * tmp1, 0,
        circle_results.qCharge * (eMinusd(0) * radii(0, 0) + eMinusd(1) * radii(1, 0)) * tmp2,
        circle_results.qCharge * (eMinusd(0) * radii(0, 1) + eMinusd(1) * radii(1, 1)) * tmp2, 0, 0, 0,
        circle_results.qCharge;

    circle_results.cov << iMat(0, 0), iMat(0, 1), iMat(0, n), iMat(1, 0), iMat(1, 1), iMat(1, n), iMat(n, 0),
        iMat(n, 1), iMat(n, n);

    circle_results.cov = jacobian * circle_results.cov * jacobian.transpose();

    //...Translate in the system in which the first corrected hit is the origin, adding the m.s. correction...

    translateKarimaki(circle_results, 0.5 * eMinusd(0), 0.5 * eMinusd(1), jacobian);
    circle_results.cov(0, 0) +=
        (1 + riemannFit::sqr(slope)) * multScatt(sTotal(1) - sTotal(0), bField, fast_fit(2), 2, slope);

    //...And translate back to the original system

    translateKarimaki(circle_results, dVec(0), dVec(1), jacobian);

    // compute chi2
    circle_results.chi2 = 0;
    for (u_int i = 0; i < n; i++) {
      circle_results.chi2 += weightsVec(i) * riemannFit::sqr(zInSZplane(i) - uVec(i));
      if (i > 0 && i < n - 1)
        circle_results.chi2 +=
            riemannFit::sqr(uVec(i - 1) / (sTransverse(i) - sTransverse(i - 1)) -
                            uVec(i) * (sTransverse(i + 1) - sTransverse(i - 1)) /
                                ((sTransverse(i + 1) - sTransverse(i)) * (sTransverse(i) - sTransverse(i - 1))) +
                            uVec(i + 1) / (sTransverse(i + 1) - sTransverse(i)) +
                            (sTransverse(i + 1) - sTransverse(i - 1)) * uVec(n) / 2) /
            varBeta(i);
    }
  }

  /*!
    \brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).
    
    \param hits hits coordinates.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param bField magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineData.
    \param line_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (cot(theta), Zip); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    
    \details The function implements the steps 2 and 3 of the Broken Line fit without 
   *        the curvature correction.\n
   * The step 2 is the least square fit, done by imposing the minimum constraint 
   * on the cost function and solving the consequent linear system. It determines 
   * the fitted parameters u and their covariance matrix.
   * The step 3 is the correction of the fast pre-fitted parameters for the innermost 
   * part of the track. It is first done in a comfortable coordinate system (the one 
   * in which the first hit is the origin) and then the parameters and their covariance 
   * matrix are transformed to the original coordinate system.
   */
  template <typename V4, typename M6xN, int n>
  __host__ __device__ inline void lineFit(const M6xN& hits_ge,
                                          const V4& fast_fit,
                                          const double bField,
                                          const PreparedBrokenLineData<n>& data,
                                          riemannFit::LineFit& line_results) {
    const auto& radii = data.radii;
    const auto& sTotal = data.sTotal;
    const auto& zInSZplane = data.zInSZplane;
    const auto& varBeta = data.varBeta;

    const double slope = -data.qCharge / fast_fit(3);
    riemannFit::Matrix2d rotMat = rotationMatrix(slope);

    riemannFit::Matrix3d vMat = riemannFit::Matrix3d::Zero();  // covariance matrix XYZ
    riemannFit::Matrix2x3d jacobXYZtosZ =
        riemannFit::Matrix2x3d::Zero();  // jacobian for computation of the error on s (xyz -> sz)
    riemannFit::VectorNd<n> weights = riemannFit::VectorNd<n>::Zero();
    for (u_int i = 0; i < n; i++) {
      vMat(0, 0) = hits_ge.col(i)[0];               // x errors
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];  // cov_xy
      vMat(0, 2) = vMat(2, 0) = hits_ge.col(i)[3];  // cov_xz
      vMat(1, 1) = hits_ge.col(i)[2];               // y errors
      vMat(2, 1) = vMat(1, 2) = hits_ge.col(i)[4];  // cov_yz
      vMat(2, 2) = hits_ge.col(i)[5];               // z errors
      auto tmp = 1. / radii.block(0, i, 2, 1).norm();
      jacobXYZtosZ(0, 0) = radii(1, i) * tmp;
      jacobXYZtosZ(0, 1) = -radii(0, i) * tmp;
      jacobXYZtosZ(1, 2) = 1.;
      weights(i) = 1. / ((rotMat * jacobXYZtosZ * vMat * jacobXYZtosZ.transpose() * rotMat.transpose())(
                            1, 1));  // compute the orthogonal weight point by point
    }

    riemannFit::VectorNd<n> r_u;
    for (u_int i = 0; i < n; i++) {
      r_u(i) = weights(i) * zInSZplane(i);
    }
#ifdef CPP_DUMP
    std::cout << "CU4\n" << matrixC_u(w, sTotal, varBeta) << std::endl;
#endif
    riemannFit::MatrixNd<n> iMat;
    math::cholesky::invert(matrixC_u(weights, sTotal, varBeta), iMat);
#ifdef CPP_DUMP
    std::cout << "I4\n" << iMat << std::endl;
#endif

    riemannFit::VectorNd<n> uVec = iMat * r_u;  // obtain the fitted parameters by solving the linear system

    // line parameters in the system in which the first hit is the origin and with axis along SZ
    line_results.par << (uVec(1) - uVec(0)) / (sTotal(1) - sTotal(0)), uVec(0);
    auto idiff = 1. / (sTotal(1) - sTotal(0));
    line_results.cov << (iMat(0, 0) - 2 * iMat(0, 1) + iMat(1, 1)) * riemannFit::sqr(idiff) +
                            multScatt(sTotal(1) - sTotal(0), bField, fast_fit(2), 2, slope),
        (iMat(0, 1) - iMat(0, 0)) * idiff, (iMat(0, 1) - iMat(0, 0)) * idiff, iMat(0, 0);

    // translate to the original SZ system
    riemannFit::Matrix2d jacobian;
    jacobian(0, 0) = 1.;
    jacobian(0, 1) = 0;
    jacobian(1, 0) = -sTotal(0);
    jacobian(1, 1) = 1.;
    line_results.par(1) += -line_results.par(0) * sTotal(0);
    line_results.cov = jacobian * line_results.cov * jacobian.transpose();

    // rotate to the original sz system
    auto tmp = rotMat(0, 0) - line_results.par(0) * rotMat(0, 1);
    jacobian(1, 1) = 1. / tmp;
    jacobian(0, 0) = jacobian(1, 1) * jacobian(1, 1);
    jacobian(0, 1) = 0;
    jacobian(1, 0) = line_results.par(1) * rotMat(0, 1) * jacobian(0, 0);
    line_results.par(1) = line_results.par(1) * jacobian(1, 1);
    line_results.par(0) = (rotMat(0, 1) + line_results.par(0) * rotMat(0, 0)) * jacobian(1, 1);
    line_results.cov = jacobian * line_results.cov * jacobian.transpose();

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

  /*!
    \brief Helix fit by three step:
    -fast pre-fit (see Fast_fit() for further info); \n
    -circle fit of the hits projected in the transverse plane by Broken Line algorithm (see BL_Circle_fit() for further info); \n
    -line fit of the hits projected on the (pre-fitted) cilinder surface by Broken Line algorithm (see BL_Line_fit() for further info); \n
    Points must be passed ordered (from inner to outer layer).
    
    \param hits Matrix3xNd hits coordinates in this form: \n
    |x1|x2|x3|...|xn| \n
    |y1|y2|y3|...|yn| \n
    |z1|z2|z3|...|zn|
    \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
    |(x1,x1)|(x2,x1)|(x3,x1)|(x4,x1)|.|(y1,x1)|(y2,x1)|(y3,x1)|(y4,x1)|.|(z1,x1)|(z2,x1)|(z3,x1)|(z4,x1)| \n
    |(x1,x2)|(x2,x2)|(x3,x2)|(x4,x2)|.|(y1,x2)|(y2,x2)|(y3,x2)|(y4,x2)|.|(z1,x2)|(z2,x2)|(z3,x2)|(z4,x2)| \n
    |(x1,x3)|(x2,x3)|(x3,x3)|(x4,x3)|.|(y1,x3)|(y2,x3)|(y3,x3)|(y4,x3)|.|(z1,x3)|(z2,x3)|(z3,x3)|(z4,x3)| \n
    |(x1,x4)|(x2,x4)|(x3,x4)|(x4,x4)|.|(y1,x4)|(y2,x4)|(y3,x4)|(y4,x4)|.|(z1,x4)|(z2,x4)|(z3,x4)|(z4,x4)| \n
    .       .       .       .       . .       .       .       .       . .       .       .       .       . \n
    |(x1,y1)|(x2,y1)|(x3,y1)|(x4,y1)|.|(y1,y1)|(y2,y1)|(y3,x1)|(y4,y1)|.|(z1,y1)|(z2,y1)|(z3,y1)|(z4,y1)| \n
    |(x1,y2)|(x2,y2)|(x3,y2)|(x4,y2)|.|(y1,y2)|(y2,y2)|(y3,x2)|(y4,y2)|.|(z1,y2)|(z2,y2)|(z3,y2)|(z4,y2)| \n
    |(x1,y3)|(x2,y3)|(x3,y3)|(x4,y3)|.|(y1,y3)|(y2,y3)|(y3,x3)|(y4,y3)|.|(z1,y3)|(z2,y3)|(z3,y3)|(z4,y3)| \n
    |(x1,y4)|(x2,y4)|(x3,y4)|(x4,y4)|.|(y1,y4)|(y2,y4)|(y3,x4)|(y4,y4)|.|(z1,y4)|(z2,y4)|(z3,y4)|(z4,y4)| \n
    .       .       .    .          . .       .       .       .       . .       .       .       .       . \n
    |(x1,z1)|(x2,z1)|(x3,z1)|(x4,z1)|.|(y1,z1)|(y2,z1)|(y3,z1)|(y4,z1)|.|(z1,z1)|(z2,z1)|(z3,z1)|(z4,z1)| \n
    |(x1,z2)|(x2,z2)|(x3,z2)|(x4,z2)|.|(y1,z2)|(y2,z2)|(y3,z2)|(y4,z2)|.|(z1,z2)|(z2,z2)|(z3,z2)|(z4,z2)| \n
    |(x1,z3)|(x2,z3)|(x3,z3)|(x4,z3)|.|(y1,z3)|(y2,z3)|(y3,z3)|(y4,z3)|.|(z1,z3)|(z2,z3)|(z3,z3)|(z4,z3)| \n
    |(x1,z4)|(x2,z4)|(x3,z4)|(x4,z4)|.|(y1,z4)|(y2,z4)|(y3,z4)|(y4,z4)|.|(z1,z4)|(z2,z4)|(z3,z4)|(z4,z4)|
    \param bField magnetic field in the center of the detector in Gev/cm/c, in order to perform the p_t calculation.
    
    \warning see BL_Circle_fit(), BL_Line_fit() and Fast_fit() warnings.
    
    \bug see BL_Circle_fit(), BL_Line_fit() and Fast_fit() bugs.
    
    \return (phi,Tip,p_t,cot(theta)),Zip), their covariance matrix and the chi2's of the circle and line fits.
  */
  template <int n>
  inline riemannFit::HelixFit helixFit(const riemannFit::Matrix3xNd<n>& hits,
                                       const Eigen::Matrix<float, 6, 4>& hits_ge,
                                       const double bField) {
    riemannFit::HelixFit helix;
    riemannFit::Vector4d fast_fit;
    fastFit(hits, fast_fit);

    PreparedBrokenLineData<n> data;
    karimaki_circle_fit circle;
    riemannFit::LineFit line;
    riemannFit::Matrix3d jacobian;

    prepareBrokenLineData(hits, fast_fit, bField, data);
    lineFit(hits_ge, fast_fit, bField, data, line);
    circleFit(hits, hits_ge, fast_fit, bField, data, circle);

    // the circle fit gives k, but here we want p_t, so let's change the parameter and the covariance matrix
    jacobian << 1., 0, 0, 0, 1., 0, 0, 0,
        -std::abs(circle.par(2)) * bField / (riemannFit::sqr(circle.par(2)) * circle.par(2));
    circle.par(2) = bField / std::abs(circle.par(2));
    circle.cov = jacobian * circle.cov * jacobian.transpose();

    helix.par << circle.par, line.par;
    helix.cov = riemannFit::MatrixXd::Zero(5, 5);
    helix.cov.block(0, 0, 3, 3) = circle.cov;
    helix.cov.block(3, 3, 2, 2) = line.cov;
    helix.qCharge = circle.qCharge;
    helix.chi2_circle = circle.chi2;
    helix.chi2_line = line.chi2;

    return helix;
  }

}  // namespace brokenline

#endif  // RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h
