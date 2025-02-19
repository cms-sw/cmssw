#ifndef MultiStatePropagation_h_
#define MultiStatePropagation_h_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"

/** \class MultiStatePropagation
 *  Helper class to propagate all components of a state,
 *  using a single state propagator and keeping the specialisation 
 *  into planes / cylinders. Designed for short lifetime: will 
 *  directly use the propagator passed by invoking object.
 */
template <class T>
class MultiStatePropagation {  
  
public:
  /** Constructor with explicit propagator
   */
  MultiStatePropagation(const Propagator& aPropagator) :
    thePropagator(aPropagator) {}

  ~MultiStatePropagation() {};

  /** Propagation to surface with path length calculation:
   */
  std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath (const TrajectoryStateOnSurface& tsos, 
		     const T& surface) const;

private:
  /// creation of new state with different weight
  TrajectoryStateOnSurface setWeight (const TrajectoryStateOnSurface,
				      const double) const;

private:
  // Single state propagator
  const Propagator& thePropagator;

  typedef std::pair<TrajectoryStateOnSurface,double> TsosWP;
  typedef std::vector<TrajectoryStateOnSurface> MultiTSOS;
};

#include "TrackingTools/GsfTools/src/MultiStatePropagation.icc"
#endif
