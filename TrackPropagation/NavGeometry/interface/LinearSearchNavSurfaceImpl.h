#ifndef LinearSearchNavSurfaceImpl_H
#define LinearSearchNavSurfaceImpl_H

#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include <vector>

class NavVolume;
class Bounds;

class LinearSearchNavSurfaceImpl {
public:

    const NavVolume* nextVolume( const NavSurface::LocalPoint& point, 
					 SurfaceOrientation::Side side) const;

    const Bounds* bounds( const NavVolume* vol);

    void addVolume( const NavVolume* vol, const Bounds* bounds, 
		    SurfaceOrientation::Side side);

private:

    typedef std::pair< const NavVolume*, const Bounds*>        VolumeAndBounds;
    typedef std::vector< VolumeAndBounds>                      VolumeContainer;

    VolumeContainer thePosVolumes;
    VolumeContainer theNegVolumes;

    const NavVolume* findVolume( const NavSurface::LocalPoint& point,
				 const VolumeContainer& volumes) const;

};

#endif
