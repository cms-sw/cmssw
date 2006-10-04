#ifndef GeomPropagators_StateOnTrackerBound_H
#define GeomPropagators_StateOnTrackerBound_H

class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class Propagator;

/** Propagates to the Tracker bounds, i.e. either to the
 *  barrel cylinder or to one of the forward disks that 
 *  constitute the envelope of the sensitive Tracker volumes.
 *  Ported from ORCA
 *  $Date: 2006/04/24 20:36:03 $
 *  $Revision: 1.2 $
 */
class StateOnTrackerBound {
public:

  StateOnTrackerBound(Propagator* prop);

  ~StateOnTrackerBound();

  TrajectoryStateOnSurface 
  operator()( const TrajectoryStateOnSurface& tsos) const;

  TrajectoryStateOnSurface 
  operator()( const FreeTrajectoryState& fts) const;

private:

  Propagator* thePropagator;

};
#endif


