#ifndef SingleGaussianState_H
#define SingleGaussianState_H

#include "Math/SVector.h"
#include "Math/SMatrix.h"

/** Multi-dimensional (single) Gaussian state. Used for the description 
 * of Gaussian mixtures.
 */

template <unsigned int N> class SingleGaussianState {
public:
  typedef ROOT::Math::SVector<double, N> Vector;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > Matrix;
  
public:
  SingleGaussianState() :
    theHasWeightMatrix(false) {
//     ++instances_;++maxInstances_;
  }

//   SingleGaussianState(const SingleGaussianState<N>& rhs) :
//     theWeight(rhs.theWeight), theMean(rhs.theMean), theCovariance(rhs.theCovariance),
//     theHasWeightMatrix(rhs.theHasWeightMatrix), theWeightMatrix(rhs.theWeightMatrix) {
//     ++instances_;++maxInstances_;
//   }
  
  SingleGaussianState(const Vector& aMean,
			 const Matrix& aCovariance, 
			 double aWeight = 1.) : 
    theWeight(aWeight), theMean(aMean), theCovariance(aCovariance),
    theHasWeightMatrix(false) {
//     ++instances_;++maxInstances_;
  }
  
  ~SingleGaussianState() {
//     --instances_;
  }
  
  //   /**
//    * Creates a new single-state with the given information.
//    * For this base class, no information is passed from the initial 
//    * instance.
//    */
//   SingleGaussianState 
//   	createState(const AlgebraicVector & aMean, 
// 		       const AlgebraicSymMatrix & aCovariance, double aWeight = 1) const;

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

// protected:
private:
  Matrix theCovariance;
  Vector theMean;
  double theWeight;

  mutable Matrix theWeightMatrix;
  mutable bool theHasWeightMatrix;

// public:
//   static int instances_;
//   static int maxInstances_;
//   static int constructsWeightMatrix_;
};

#include "TrackingTools/GsfTools/interface/SingleGaussianState.icc"

// template <unsigned int N> int SingleGaussianState<N>::instances_ = 0;
// template <unsigned int N> int SingleGaussianState<N>::maxInstances_ = 0;
// template <unsigned int N> int SingleGaussianState<N>::constructsWeightMatrix_ = 0;
#endif
