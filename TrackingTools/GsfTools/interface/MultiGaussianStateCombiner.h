#ifndef MultiGaussianStateCombiner_H
#define MultiGaussianStateCombiner_H

#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"
#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"

  /**
   * Class to collapse (combine) a Gaussian mixture of states
   * into one.
   * (c.f. R. Fruewirth et.al., Comp.Phys.Comm 100 (1997) 1
   */

template <unsigned int N>
class MultiGaussianStateCombiner {

private:
  typedef SingleGaussianState<N> SingleState;
  typedef MultiGaussianState<N> MultiState;
  typedef typename MultiGaussianState<N>::SingleStatePtr SingleStatePtr;
  typedef typename MultiGaussianState<N>::SingleStateContainer VSC;
  
public:

//   typedef std::vector<SingleState> VSC;

  SingleStatePtr combine(const MultiState & theState) const;
  SingleStatePtr combine(const VSC& theComponents) const;

};

#include "TrackingTools/GsfTools/interface/MultiGaussianStateCombiner.icc"

#endif
