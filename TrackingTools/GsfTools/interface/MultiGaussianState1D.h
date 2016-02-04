#ifndef MultiGaussianState1D_H_
#define MultiGaussianState1D_H_

#include "TrackingTools/GsfTools/interface/SingleGaussianState1D.h"

#include <vector>

/** One-dimensional multi-Gaussian mixture: weighted sum of single
 *  Gaussian components.
 */

class MultiGaussianState1D {
public:
  typedef std::vector<SingleGaussianState1D> SingleState1dContainer;

public:

  MultiGaussianState1D() : theCombinedStateUp2Date(false) {}

  MultiGaussianState1D(const SingleState1dContainer& stateV)
    : theComponents(stateV), theCombinedStateUp2Date(false) {}

  ~MultiGaussianState1D() {}

  /// combined weight
  double weight() const;
  /// combined mean
  double mean() const;
  /// combined variance
  double variance() const;
  /// access to components
  const SingleState1dContainer& components() const {return theComponents;}

// protected:
private:
  /// calculation of the combined state (on demand)
  void checkCombinedState() const;

  // should become a vector of pointers to const SingleState
  const SingleState1dContainer theComponents;
  mutable SingleGaussianState1D theCombinedState;
  mutable bool theCombinedStateUp2Date;
};

#endif
