#ifndef GsfPropagatorAdapter_h_
#define GsfPropagatorAdapter_h_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

class MagneticField;

/** \class GsfPropagatorAdapter
 * Propagation of multiple trajectory state by propagation of
 * components, using an specified single-state propagator.
 */
class GsfPropagatorAdapter : public Propagator {

public:
  /// Constructor with explicit propagator
  GsfPropagatorAdapter (const Propagator& Propagator);

  ~GsfPropagatorAdapter() {}

  /** Propagation to generic surface: specialisation done in base class.
   */
  virtual TrajectoryStateOnSurface propagate (const TrajectoryStateOnSurface& tsos, 
					      const Surface& surface) override
  {
    return Propagator::propagate(tsos,surface);
  }
  /** Propagation to plane: use propagationWithPath (adequate for use with
   *  AnalyticalPropagator, should be implemented to be more general).
   */
  virtual TrajectoryStateOnSurface propagate (const TrajectoryStateOnSurface& tsos, 
					      const Plane& plane) override
  {
    return propagateWithPath(tsos,plane).first;
  }
  /** Propagation to cylinder: use propagationWithPath (adequate for use with
   *  AnalyticalPropagator, should be implemented to be more general).
   */
  virtual TrajectoryStateOnSurface propagate (const TrajectoryStateOnSurface& tsos, 
					      const Cylinder& cylinder) override
  {
    return propagateWithPath(tsos,cylinder).first;
  }

  /** Propagation to generic surface with path length calculation: 
   *  specialisation done in base class.
   */
  virtual std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath (const TrajectoryStateOnSurface& tsos, 
		     const Surface& surface) override
  {
    return Propagator::propagateWithPath(tsos,surface);
  }
  /** Propagation to plane with path length calculation.
   */
  virtual std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath (const TrajectoryStateOnSurface&, 
		     const Plane&) override;
  /** Propagation to cylinder with path length calculation.
   */
  virtual std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath (const TrajectoryStateOnSurface&, 
		     const Cylinder&) override;

  /** Propagation to generic surface: specialisation done in base class.
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual TrajectoryStateOnSurface propagate (const FreeTrajectoryState& fts, 
					      const Surface& surface) override
  {
    return Propagator::propagate(fts,surface);
  }
  /** Propagation to plane: use propagationWithPath (adequate for use with
   *  AnalyticalPropagator, should be implemented to be more general).
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual TrajectoryStateOnSurface propagate (const FreeTrajectoryState& fts, 
					      const Plane& plane) override
  {
    return propagateWithPath(fts,plane).first;
  }
  /** Propagation to cylinder: use propagationWithPath (adequate for use with
   *  AnalyticalPropagator, should be implemented to be more general).
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual TrajectoryStateOnSurface propagate (const FreeTrajectoryState& fts, 
					      const Cylinder& cylinder) override
  {
    return propagateWithPath(fts,cylinder).first;
  }

  /** Propagation to generic surface with path length calculation: 
   *  specialisation done in base class.
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual std::pair<TrajectoryStateOnSurface,double> propagateWithPath (const FreeTrajectoryState& fts, 
									const Surface& surface) override
  {
    return Propagator::propagateWithPath(fts,surface);
  }
  /** Propagation to plane with path length calculation.
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual std::pair<TrajectoryStateOnSurface,double> propagateWithPath (const FreeTrajectoryState&, 
									const Plane&) override;
  /** Propagation to cylinder with path length calculation. 
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual std::pair<TrajectoryStateOnSurface,double> propagateWithPath (const FreeTrajectoryState&, 
									const Cylinder&) override;

  virtual bool setMaxDirectionChange( float phiMax) { 
    return thePropagator->setMaxDirectionChange(phiMax);
  }

 virtual void setPropagationDirection (PropagationDirection dir) override;

  /// access to single state propagator
  inline const Propagator& propagator () const
  {
    return *thePropagator;
  }

  inline Propagator& propagator ()
  {
    return *thePropagator;
  }

  virtual GsfPropagatorAdapter* clone() const override
  {
    return new GsfPropagatorAdapter(*thePropagator);
  }

  virtual const MagneticField* magneticField() const override {
    return thePropagator->magneticField();
  }

private:
  // Single state propagator
  DeepCopyPointerByClone<Propagator> thePropagator;
};

#endif
