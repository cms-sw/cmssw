#ifndef SingleGaussianState_H
#define SingleGaussianState_H

#define SMATRIX_USE_CONSTEXPR

#include "Math/SVector.h"
#include "Math/SMatrix.h"

/** Multi-dimensional (single) Gaussian state. Used for the description 
 * of Gaussian mixtures.
 */

template <unsigned int N> class SingleGaussianState {
public:
  using Vector = ROOT::Math::SVector<double, N>;
  using Matrix = ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N>>;
  
public:
  SingleGaussianState() {}

  SingleGaussianState(const Vector& aMean,
		      const Matrix& aCovariance, 
		      double aWeight = 1.) : 
    theCovariance(aCovariance), theMean(aMean), theWeight(aWeight) {}  

  /// weight
  inline double weight() const {return theWeight;}
  /// parameter vector
  inline const Vector & mean() const {return theMean;}
  /// covariance matrix
  inline const Matrix & covariance() const {return theCovariance;}
  /// weight matrix
  const Matrix & weightMatrix() const;
  /// size of parameter vector
  inline unsigned int dimension () const {return N;}
  /// change weight
  void rescaleWeight (double scale) {theWeight *= scale;}

private:
  Matrix theCovariance;
  Vector theMean;
  double theWeight;

  mutable Matrix theWeightMatrix = ROOT::Math::SMatrixNoInit();
  mutable bool theHasWeightMatrix = false;
};

#include "TrackingTools/GsfTools/interface/SingleGaussianState.icc"

#endif
