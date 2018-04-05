#ifndef GsfToolsGetComponents_H
#define GsfToolsGetComponents_H
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
 struct GetComponents {
  explicit GetComponents(TrajectoryStateOnSurface const &tsos) : 
    comps(&single) {
    if (tsos.singleState()) single.push_back(tsos);
    else comps = &tsos.components();
  }
  TrajectoryStateOnSurface::Components const & operator()() const { return *comps;}

 TrajectoryStateOnSurface::Components single;
 TrajectoryStateOnSurface::Components const * comps;
};
#endif
