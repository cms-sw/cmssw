#ifndef GaussianStateLessWeight_h_
#define GaussianStateLessWeight_h_

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "boost/shared_ptr.hpp"

/** \class GaussianStateLessWeight
 * Compare two SingleGaussianState acc. to their weight.
 */

template <unsigned int N>
class GaussianStateLessWeight {
  
private:
  typedef boost::shared_ptr< SingleGaussianState<N> > SingleStatePtr;

public:
  GaussianStateLessWeight() {}
  bool operator()(const SingleStatePtr& a, 
		  const SingleStatePtr& b) const
  {
// ThS: No validity for SingleGaussianState
//     if ( !a.isValid() || !b.isValid() )  return false;
    return a->weight()>b->weight();
  }
};

#endif
