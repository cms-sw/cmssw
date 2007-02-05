#ifndef MultiGaussianStateMerger_H
#define MultiGaussianStateMerger_H

#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/RCMultiGaussianState.h"

/** Abstract base class for trimming or merging a MultiGaussianState into 
 *  one with a smaller number of components.
 */

class MultiGaussianStateMerger {

public:
  virtual RCMultiGaussianState merge(const RCMultiGaussianState& mgs) const = 0;
  virtual ~MultiGaussianStateMerger() {}
  virtual MultiGaussianStateMerger* clone() const = 0;

protected:

  MultiGaussianStateMerger() {}
  typedef std::vector<RCSingleGaussianState> SGSVector;

};  

#endif
