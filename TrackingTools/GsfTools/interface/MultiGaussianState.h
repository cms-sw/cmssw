#ifndef MultiGaussianState_H
#define MultiGaussianState_H

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "boost/shared_ptr.hpp"

#include <vector>

// #include <iostream>
// #include <signal.h>
/// Mixture of multi-variate gaussian states

/** Multi-dimensional multi-Gaussian mixture: weighted sum of single
 *  Gaussian components.
 */

template <unsigned int N>
class MultiGaussianState {
public:
  typedef typename SingleGaussianState<N>::Vector Vector;
  typedef typename SingleGaussianState<N>::Matrix Matrix;
  typedef SingleGaussianState<N> SingleState;
  typedef boost::shared_ptr<SingleState> SingleStatePtr;
//   typedef std::vector< boost::shared_ptr<const SingleState> > SingleStateContainer;
  typedef std::vector< SingleStatePtr > SingleStateContainer;

public:

  MultiGaussianState() : theCombinedStateUp2Date(false) {
//     ++instances_;++maxInstances_;
//     std::cout << "MultiGaussianState() " << N << " " << instances_ << std::endl;
  }

  MultiGaussianState(const SingleStateContainer& stateV)
    : theComponents(stateV), theCombinedStateUp2Date(false) {
//     theComponents[0]->rescaleWeight(1.);
//     ++instances_;++maxInstances_;
//     std::cout << "MultiGaussianState(const SingleStateContainer&) " << N << " " 
// 	      << instances_ << std::endl;
  }

//   MultiGaussianState(const MultiGaussianState<N>& rhs) :
//     theComponents(rhs.theComponents), theCombinedState(rhs.theCombinedState),
//     theCombinedStateUp2Date(rhs.theCombinedStateUp2Date) {
//     ++instances_;++maxInstances_;
//     std::cout << "MultiGaussianState(const MultiGaussianState<N>&) " << N << " " 
// 	      << instances_ << std::endl;
//   }

  ~MultiGaussianState() {
//     --instances_;
//     std::cout << "~MultiGaussianState " << N << " " << instances_ << std::endl;
  }

//   /**
//    * Creates a new multi-state with the given components.
//    * For this base class, no information is passed from the initial
//    * instance.
//    */
//   virtual MultiGaussianState createState(
// 	const std::vector<SingleGaussianState> & stateV) const {
//     return MultiGaussianState(stateV);
//   }

//   /**
//    * Creates a new single-state with the given information.
//    * For this base class, no information is passed from the initial
//    * instance.
//    */
//   virtual SingleGaussianState createSingleState (
// 	const AlgebraicVector & aMean, const AlgebraicSymMatrix & aCovariance,
// 	double aWeight = 1.) const {
//     return SingleGaussianState(aMean, aCovariance, aWeight);
//   }

  /// combined weight
  double weight() const;
  /// combined mean
  const Vector & mean() const;
  /// combined covariance matrix
  const Matrix & covariance() const;
  /// combined weight matrix
  const Matrix & weightMatrix() const;
  /// access to components (single Gaussian states)
  inline const SingleStateContainer& components() const {return theComponents;}
  /// dimension of parameter vector
  int dimension () const {
    return N;
  }
  /// renormalize weight
  void setWeight (double newWeight);
  /// rescale weight
  void rescaleWeight (double scale);

// protected:
private:
  /// calculation of the combined state (on demand)
  void checkCombinedState() const;

//   std::vector<SingleState> theComponents;
// should become a vector of pointers to const SingleState ...
  const SingleStateContainer theComponents;
  mutable SingleStatePtr theCombinedState;
  mutable bool theCombinedStateUp2Date;

// public:
//   static int instances_;
//   static int maxInstances_;
//   static int constructsCombinedState_;
};

#include "TrackingTools/GsfTools/interface/MultiGaussianState.icc"

//   template <unsigned int N> int MultiGaussianState<N>::instances_ = 0;
//   template <unsigned int N> int MultiGaussianState<N>::maxInstances_ = 0;
//   template <unsigned int N> int MultiGaussianState<N>::constructsCombinedState_ = 0;

#endif
