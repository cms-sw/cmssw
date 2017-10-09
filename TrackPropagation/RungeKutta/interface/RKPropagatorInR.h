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

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Plane&) const;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const;

  virtual Propagator * clone() const;

  virtual const MagneticField* magneticField() const {return theVolume;}

private:

  const MagVolume* theVolume;

};

#endif
