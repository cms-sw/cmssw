#ifndef TkDetLayers_LayerCrossingSide_h
#define TkDetLayers_LayerCrossingSide_h

// temporary solution
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

/** Helper class to determine if a TrajectoryStateOnSurface would cross a layer from 
 *  the inside, or from the outside, if propagated with a propagator with a defined 
 *  direction. No propagations performed, the result is very fast but not correct in 
 *  case of a looping track that first goes outwards, then inwards, before crossing
 *  a layer.
 */

#pragma GCC visibility push(hidden)
class LayerCrossingSide {
public:

  /// returns 0 if barrel layer crossed from inside, 1 if from outside
  int barrelSide(const TrajectoryStateOnSurface& startingState, const Propagator& prop) const {
    GlobalPoint pos = startingState.globalPosition();
    GlobalVector radial(pos.x(), pos.y(), 0);
    if (startingState.globalMomentum().dot( radial) > 0) {  // momentum points outwards
      return (prop.propagationDirection() == alongMomentum ? 0 : 1);
    }
    else {  // momentum points inwards
      return (prop.propagationDirection() == oppositeToMomentum ? 0 : 1);
    }
  }

  /** returns 0 if endcap layer crossed from inside, ie from the side of the 
   *  interation region, 1 if from outside
   */
  int endcapSide(const TrajectoryStateOnSurface& startingState, const Propagator& prop) const {
    float zpos = startingState.globalPosition().z();
    if (startingState.globalMomentum().z() * zpos > 0) {  // momentum points outwards
      return (prop.propagationDirection() == alongMomentum ? 0 : 1);
    }
    else {  // momentum points inwards
      return (prop.propagationDirection() == oppositeToMomentum ? 0 : 1);
    }
  }

};

#pragma GCC visibility pop
#endif
