#ifndef NavVolume6Faces_H
#define NavVolume6Faces_H

#include "TrackPropagation/NavGeometry/interface/NavVolume.h"
#include "TrackPropagation/NavGeometry/interface/NavVolumeSide.h"

#include <vector>

class Bounds;
class Plane;

class NavVolume6Faces : public NavVolume {
public:

    NavVolume6Faces( const PositionType& pos, const RotationType& rot, 
		     DDSolidShape shape, const std::vector<NavVolumeSide>& faces,
		     const MagneticFieldProvider<float> * mfp);

    /// A NavVolume6Faces that corresponds exactly to a MagVolume
    explicit NavVolume6Faces( const MagVolume& magvol);

    virtual Container nextSurface( const NavVolume::LocalPoint& pos, 
				   const NavVolume::LocalVector& mom,
				   double charge, PropagationDirection propDir=alongMomentum) const;

private:

    Container theNavSurfaces;

    void computeBounds(const std::vector<NavVolumeSide>& faces);
    Bounds* computeBounds( int index, const std::vector<const Plane*>& bpc);
    Bounds* computeBounds( int index, const std::vector<NavVolumeSide>& faces);

};

#endif
