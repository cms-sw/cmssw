#ifndef RKPropagatorInZ_H
#define RKPropagatorInZ_H

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"

class RKPropagatorInZ GCC11_FINAL : public Propagator {
public:

  RKPropagatorInZ( const MagVolume& vol, PropagationDirection dir = alongMomentum) : 
    Propagator(dir), theVolume( &vol) {}

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Plane&) override;

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Cylinder&) override;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Plane&) override;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) override;

  virtual Propagator * clone() const override;

  virtual const MagneticField* magneticField() const override {return theVolume;}

private:

  const MagVolume* theVolume;

};

#endif
