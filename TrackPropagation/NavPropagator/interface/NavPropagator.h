#ifndef NavPropagator_H_
#define NavPropagator_H_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TrackingTools/MaterialEffects/interface/VolumeMediumProperties.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsUpdator.h"
#include "TrackingTools/MaterialEffects/interface/VolumeMultipleScatteringEstimator.h"
#include "TrackingTools/MaterialEffects/interface/VolumeEnergyLossEstimator.h"

#include <map>

class MagneticField; 
class VolumeBasedMagneticField;
class RKPropagatorInS;
class NavVolume;
class MagVolume;


class NavPropagator : public Propagator {

public:

  NavPropagator( const MagneticField* field,
		 PropagationDirection dir = alongMomentum);

  ~NavPropagator();
  //
  // use base class methods where necessary:
  // - propagation from TrajectoryStateOnSurface 
  //     (will use propagation from FreeTrajectoryState)
  // - propagation to general Surface
  //     (will use specialised methods for planes or cylinders)
  //
  using Propagator::propagate;
  using Propagator::propagateWithPath;

  /// propagation of TSOS to plane
  TrajectoryStateOnSurface propagate(const TrajectoryStateOnSurface& ts, 
                                     const Plane& plane) const {
    return propagateWithPath(ts,plane).first;
  }
  /// propagation of TSOS to plane with path length  
  std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath(const TrajectoryStateOnSurface& , 
		    const Plane& plane) const; 
  
  /// propagation of FTS to plane
  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& ts, 
                                     const Plane& plane) const {
    return propagateWithPath(ts,plane).first;
  }
  /// propagation of FTS to plane with path length  
  std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath(const FreeTrajectoryState& , 
		    const Plane& plane) const; 
  
  /// propagation to cylinder
  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts, 
                                     const Cylinder& cylinder) const {
    return propagateWithPath(fts,cylinder).first;
  }
  /// propagation to cylinder with path length
  std::pair<TrajectoryStateOnSurface,double> 
  propagateWithPath(const FreeTrajectoryState& fts, 
		    const Cylinder& cylinder) const;

  virtual NavPropagator * clone() const {return new NavPropagator(*this);}

  virtual const MagneticField* magneticField() const;

private:

  typedef std::pair<TrajectoryStateOnSurface,double> TsosWP;
  typedef RKPropagatorInS                            PropagatorType;
  typedef std::map<const MagVolume*, NavVolume*>     MagVolumeMap;
  typedef TrajectoryStateOnSurface                   TSOS;


  const VolumeBasedMagneticField*   theField;
  mutable MagVolumeMap              theNavVolumeMap;
  bool  isIronVolume[272];


  const NavVolume* findVolume( const TrajectoryStateOnSurface& inputState) const;

  const NavVolume* navVolume( const MagVolume* magVolume) const;

  bool destinationCrossed( const TSOS& startState,
			   const TSOS& endState, const Plane& plane) const;

  TrajectoryStateOnSurface noNextVolume( TrajectoryStateOnSurface startingState) const;

  std::pair<TrajectoryStateOnSurface,double> 
  propagateInVolume( const NavVolume* currentVolume, 
		     const TrajectoryStateOnSurface& startingState, 
		     const Plane& targetPlane) const;

  const VolumeMediumProperties  theAirMedium;
  const VolumeMediumProperties  theIronMedium;
  const VolumeMultipleScatteringEstimator theMSEstimator;
  const VolumeEnergyLossEstimator theELEstimator;
  const VolumeMaterialEffectsUpdator theMaterialUpdator;
};

#endif
