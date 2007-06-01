#ifndef MultiGaussianStateMerger_H
#define MultiGaussianStateMerger_H

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"

/** Abstract base class for trimming or merging a MultiGaussianState into 
 *  one with a smaller number of components.
 */

template <unsigned int N> class MultiGaussianStateMerger {
public:
  typedef SingleGaussianState<N> SingleState;
  typedef MultiGaussianState<N> MultiState;

public:
  virtual MultiState merge(const MultiState& mgs) const = 0;
  virtual ~MultiGaussianStateMerger() {}
  virtual MultiGaussianStateMerger* clone() const = 0;

protected:

  MultiGaussianStateMerger() {}
  typedef std::vector<SingleState> SGSVector;

};  

#endif
