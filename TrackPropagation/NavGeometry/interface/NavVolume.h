#ifndef NavVolume_H
#define NavVolume_H

#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

#include  "TrackPropagation/NavGeometry/interface/SurfaceAndBounds.h"

#include <vector>

class NavSurface;
class Bounds;
//class SurfaceAndBounds

class NavVolume : public MagVolume {
public:

  NavVolume( const PositionType& pos, const RotationType& rot, 
	     DDSolidShape shape, const MagneticFieldProvider<float> * mfp) :
    MagVolume(pos,rot,shape,mfp) {}

  //    typedef std::pair< const NavSurface*, const Bounds*>       SurfaceAndBounds;
    typedef std::vector<SurfaceAndBounds>                      Container;

    virtual ~NavVolume() {} 

    virtual Container nextSurface( const LocalPoint& pos, const LocalVector& mom, double charge, 
				   PropagationDirection propDir = alongMomentum) const = 0;

};

#endif

