#ifndef GaussianSumUtilities_h_
#define GaussianSumUtilities_h_

#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"
#include <vector>

/** Utility class for the analysis of multi-dimensional Gaussian
 *  mixtures. The input state is assumed to exist for 
 *  the lifetime of this object.
 */

template <unsigned int N>
class GaussianSumUtilities {
public:
  typedef SingleGaussianState<N> SingleState;
  typedef MultiGaussianState<N> MultiState; 
//   typedef ROOT::Math::SVector<double, N> Vector;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepStd<double,N> > GenMatrix;

  typedef typename SingleState::Vector Vector;
  typedef typename SingleState::Matrix Matrix;
  typedef typename MultiState::SingleStatePtr SingleStatePtr;
  typedef typename MultiState::SingleStateContainer SingleStateContainer;

private:
  enum ModeStatus { Valid, NotValid, NotComputed };

public:
  GaussianSumUtilities (const MultiState& state) :
    theState(state), 
    theModeStatus(NotComputed) {
  } 
  ~GaussianSumUtilities () {
  }

  /// number of components
  inline unsigned int size () const {
    return components().size();
  }
  /// components
  const SingleStateContainer& components () const {
    return theState.components();
  }
  /// multi-state
  const MultiState& state () const {
    return theState;
  }
  /// weight of a component
  inline double weight (unsigned int i) const {
    return components()[i]->weight();
  }
  /// mean value of a component
  inline const Vector& mean (unsigned int i) const {
    return components()[i]->mean();
  }
  /// covariance matrix of a component
  inline const Matrix& covariance (unsigned int i) const {
    return components()[i]->covariance();
  }
  /// mode status
  bool modeIsValid () const;
  /** Mode "state": mean = mode, covariance = local covariance at mode,
   *  weight chosen to have pdf(mode) equal to the one of the mixture */
  const SingleGaussianState<N>& mode () const;
  /// value of the p.d.f.
  double pdf (const Vector&) const;
  /// gradient
  Vector d1Pdf (const Vector&) const;
  /// Hessian
  Matrix d2Pdf (const Vector&) const;
  /// value of ln(pdf)
  double lnPdf (const Vector&) const;
  /// gradient of ln(pdf)
  Vector d1LnPdf (const Vector&) const;
  /// Hessian of ln(pdf)
  Matrix d2LnPdf (const Vector&) const;

  /// combined weight
  double weight () const {
    return theState.weight();
  }
  /// combined mean
  const Vector& mean () const {
    return theState.mean();
  }
  /// combined covariance
  const Matrix& covariance () const {
    return theState.covariance();
  }


protected:
  /// calculation of mode
  Vector computeModeWithoutTransform () const;

private:
  /// Symmetric Tensor Product (not recognized by standard ROOT Math)
  Matrix tensorProduct (const Vector&) const;
  /// value of gaussian distribution
  double gauss (const double&, const double&, const double&) const;
  /// value of multidimensional gaussian distribution
  double gauss (const Vector&, 
		const Vector&,
		const Matrix&) const;
  /// mode from starting value in ln(pdf); returns true on success
  bool findMode (Vector& mode, double& pdfAtMode,
		 const Vector& xStart) const;
  /// calculation of mode with transformation of pdf
  void computeMode () const;
  /// state constrained to a line x = s*d+x0
  MultiGaussianState1D constrainedState (const Vector& d,
					 const Vector& x0) const;
//   /// replacement of CLHEP determinant (which rounds off small values)
//   double determinant (const Matrix& matrix) const;
  /** Local variance from Hessian matrix.
   *  Only valid if x corresponds to a (local) maximum! */
  Matrix localCovariance (const Vector& x) const;
  /// set mode "state" from solution of mode finding
  void setMode (const Vector& mode) const;
  /// set mode "state" in case of failure
  void setInvalidMode () const;

  /// pdf components
  std::vector<double> pdfComponents (const Vector&) const;
  /// value of the p.d.f. using the pdf components at the evaluation point
  double pdf (const Vector&, const std::vector<double>&) const;
  /// gradient using the pdf components at the evaluation point
  Vector d1Pdf (const Vector&, const std::vector<double>&) const;
  /// Hessian using the pdf components at the evaluation point
  Matrix d2Pdf (const Vector&, const std::vector<double>&) const;
  /// value of ln(pdf) using the pdf components at the evaluation point
  double lnPdf (const Vector&, const std::vector<double>&) const;
  /// gradient of ln(pdf) using the pdf components at the evaluation point
  Vector d1LnPdf (const Vector&, const std::vector<double>&) const;
  /// Hessian of ln(pdf) using the pdf components at the evaluation point
  Matrix d2LnPdf (const Vector&, const std::vector<double>&) const;



private:
  const MultiState& theState;
//   int theDimension;

  mutable ModeStatus theModeStatus;
//   mutable Vector theMode;
  mutable SingleGaussianState<N> theMode;

};

#include "TrackingTools/GsfTools/interface/GaussianSumUtilities.icc"

#endif
