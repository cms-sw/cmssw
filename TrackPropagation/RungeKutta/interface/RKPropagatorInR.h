#ifndef RKPropagatorInR_H
#define RKPropagatorInR_H

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"

class RKPropagatorInR final : public Propagator {
public:

  RKPropagatorInR( const MagVolume& vol, PropagationDirection dir = alongMomentum) : 
    Propagator(dir), theVolume( &vol) {}

  TrajectoryStateOnSurface 
  myPropagate (const FreeTrajectoryState&, const Plane&) const;

  TrajectoryStateOnSurface 
  myPropagate (const FreeTrajectoryState&, const Cylinder&) const;

  std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Plane&) const override;

  std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const override;

  Propagator * clone() const override;

  const MagneticField* magneticField() const override {return theVolume;}

private:

  const MagVolume* theVolume;

};

#endif
