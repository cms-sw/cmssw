#ifndef _CR_PropagationDirectionFromPath
#define _CR_PropagationDirectionFromPath

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
/** \class PropagationDirectionFromPath
 *  Converts sign of path to propagation direction.
 */
class PropagationDirectionFromPath {
public:
  /// Direction from sign of path length
  inline PropagationDirection operator()(const double& s) const
  {
    return s>=0 ? alongMomentum : oppositeToMomentum;
  }
  /// Direction from second argument, from sign of path length, 
  inline PropagationDirection 
  operator()(const double& s,
	     const PropagationDirection propDir) const
  {
    if ( propDir!=anyDirection )  return propDir;
    return (*this)(s);
  }
};
#endif
