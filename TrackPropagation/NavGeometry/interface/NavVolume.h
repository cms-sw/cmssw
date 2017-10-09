#ifndef NavVolume_H
#define NavVolume_H

#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

#include  "TrackPropagation/NavGeometry/interface/SurfaceAndBounds.h"
#include  "TrackPropagation/NavGeometry/interface/VolumeCrossReturnType.h"

#include <vector>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


class NavSurface;
class Bounds;
class TrajectoryStateOnSurface;

class NavVolume : public MagVolume {
public:

  NavVolume( const PositionType& pos, const RotationType& rot, 
	     DDSolidShape shape, const MagneticFieldProvider<float> * mfp) :
    MagVolume(pos,rot,shape,mfp) {}

  typedef std::vector<SurfaceAndBounds>                         Container;
  ////  typedef std::pair<const NavVolume*, TrajectoryStateOnSurface>  VolumeCrossReturnType;


  virtual ~NavVolume() {} 

  virtual Container nextSurface( const LocalPoint& pos, const LocalVector& mom, double charge, 
				 PropagationDirection propDir = alongMomentum) const = 0;

  /// Same, giving lowest priority to the surface we are on now (=NotThisSurface)
  virtual Container nextSurface( const LocalPoint& pos, const LocalVector& mom, double charge, 
				 PropagationDirection propDir,
				 ConstReferenceCountingPointer<Surface> NotThisSurfaceP) const = 0;
  
  virtual VolumeCrossReturnType crossToNextVolume( const TrajectoryStateOnSurface& currentState, 
						   const Propagator& prop) const = 0;

  virtual bool isIron() const = 0;

};

#endif

