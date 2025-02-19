#ifndef GaussianStateLessWeight_h_
#define GaussianStateLessWeight_h_

#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"

/** \class GaussianStateLessWeight
 * Compare two SingleGaussianState acc. to their weight.
 */

class GaussianStateLessWeight {
  
public:
  GaussianStateLessWeight() {}
  bool operator()(const RCSingleGaussianState a, 
		  const RCSingleGaussianState b) const
  {
// ThS: No validity for RCSingleGaussianState
//     if ( !a.isValid() || !b.isValid() )  return false;
    return a->weight()>b->weight();
  }
};

#endif
