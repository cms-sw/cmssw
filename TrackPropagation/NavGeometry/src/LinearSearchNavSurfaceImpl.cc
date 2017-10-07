#include "TrackPropagation/NavGeometry/interface/LinearSearchNavSurfaceImpl.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

const NavVolume* LinearSearchNavSurfaceImpl::nextVolume( const NavSurface::LocalPoint& point, 
							 SurfaceOrientation::Side side) const
{
    if (side == SurfaceOrientation::positiveSide) {
	return findVolume( point, thePosVolumes);
    }
    else if (side == SurfaceOrientation::negativeSide) {
	return findVolume( point, theNegVolumes);
    }
    else return nullptr;  // should not be called with SurfaceOrientation::onSurface
}

const NavVolume* LinearSearchNavSurfaceImpl::findVolume( const NavSurface::LocalPoint& point,
							 const VolumeContainer& vols) const
{
// simple linear search for volume who's bounds contain the point

//MM:return 0 if no volume was defined on this side! 
  if (vols.empty()) return nullptr;

    for (VolumeContainer::const_iterator i=vols.begin(); i!=vols.end(); i++) {
	if (i->second->inside(point)) return i->first;
    }
    return nullptr; // if point outside of all bounds on this side
}

const Bounds* LinearSearchNavSurfaceImpl::bounds( const NavVolume* vol)
{
    for (VolumeContainer::const_iterator i=thePosVolumes.begin(); i!=thePosVolumes.end(); i++) {
	if (i->first == vol) return i->second;
    }
    for (VolumeContainer::const_iterator i=theNegVolumes.begin(); i!=theNegVolumes.end(); i++) {
	if (i->first == vol) return i->second;
    }
    return nullptr; // if volume not found
}

void LinearSearchNavSurfaceImpl::addVolume( const NavVolume* vol, const Bounds* bounds, 
					    SurfaceOrientation::Side side)
{
    if (side == SurfaceOrientation::positiveSide) {
	thePosVolumes.push_back( VolumeAndBounds( vol, bounds->clone()));
    }
    else {
	theNegVolumes.push_back( VolumeAndBounds( vol, bounds->clone()));
    }
}
