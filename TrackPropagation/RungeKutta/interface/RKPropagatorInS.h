#ifndef RKPropagatorInS_H
#define RKPropagatorInS_H

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/Vector/interface/Basic3DVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"


class GlobalTrajectoryParameters;
class MagVolume;
class RKLocalFieldProvider;

class RKPropagatorInS : public Propagator {
public:

  // RKPropagatorInS( PropagationDirection dir = alongMomentum) : Propagator(dir), theVolume(0) {}

  RKPropagatorInS( const MagVolume& vol, PropagationDirection dir = alongMomentum) : 
    Propagator(dir), theVolume( &vol) {}

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Plane&) const;

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Cylinder&) const;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Plane&) const;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const;

  virtual Propagator * clone() const;

  virtual const MagneticField* magneticField() const {return theVolume;}

private:

  const MagVolume* theVolume;

  GlobalTrajectoryParameters gtpFromLocal( const Basic3DVector<double>& lpos,
					   const Basic3DVector<double>& lmom,
					   TrackCharge ch, const Surface& surf) const;

  RKLocalFieldProvider fieldProvider() const;
  RKLocalFieldProvider fieldProvider( const Cylinder& cyl) const;

  PropagationDirection invertDirection( PropagationDirection dir) const;

  Basic3DVector<double> rkPosition( const GlobalPoint& pos) const;
  Basic3DVector<double> rkMomentum( const GlobalVector& mom) const;
  GlobalPoint           globalPosition( const Basic3DVector<double>& pos) const;
  GlobalVector          globalMomentum( const Basic3DVector<double>& mom) const;

};

#endif
