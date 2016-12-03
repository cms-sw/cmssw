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
class GsfPropagatorAdapter final : public Propagator {

public:
  /// Constructor with explicit propagator
  GsfPropagatorAdapter (const Propagator& Propagator);

  ~GsfPropagatorAdapter() {}

  using Propagator::propagate;
  using Propagator::propagateWithPath;


  virtual std::pair<TrajectoryStateOnSurface,double>
  propagateWithPath (const TrajectoryStateOnSurface&,
		     const Plane&) const  override;

  /** Propagation to cylinder with path length calculation.
   */
  virtual std::pair<TrajectoryStateOnSurface,double>
  propagateWithPath (const TrajectoryStateOnSurface&,
		     const Cylinder&) const  override;

  /** Propagation to plane with path length calculation.
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual std::pair<TrajectoryStateOnSurface,double> propagateWithPath (const FreeTrajectoryState&,
									const Plane&) const  override;


  /** Propagation to cylinder with path length calculation.
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual std::pair<TrajectoryStateOnSurface,double> propagateWithPath (const FreeTrajectoryState&,
									const Cylinder&) const  override;

public:

  virtual bool setMaxDirectionChange( float phiMax) override {
    return thePropagator->setMaxDirectionChange(phiMax);
  }

 virtual void setPropagationDirection (PropagationDirection dir) override;

  /// access to single state propagator
  inline const Propagator& propagator () const
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
