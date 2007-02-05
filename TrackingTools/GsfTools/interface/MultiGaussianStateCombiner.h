#ifndef MultiGaussianStateCombiner_H
#define MultiGaussianStateCombiner_H

#include "TrackingTools/GsfTools/interface/RCMultiGaussianState.h"

#include <vector>

  /**
   * Class to collapse (combine) a Gaussian mixture of states
   * into one.
   * (c.f. R. Fruewirth et.al., Comp.Phys.Comm 100 (1997) 1
   */

class MultiGaussianStateCombiner {

public:

  typedef std::vector<RCSingleGaussianState> VSC;

  RCSingleGaussianState combine(const RCMultiGaussianState & theState) const;
  RCSingleGaussianState combine(const std::vector<RCSingleGaussianState> & 
				theComponents) const;

};

#endif
