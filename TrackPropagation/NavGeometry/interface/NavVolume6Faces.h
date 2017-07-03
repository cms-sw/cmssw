#ifndef NavVolume6Faces_H
#define NavVolume6Faces_H

#include "TrackPropagation/NavGeometry/interface/NavVolume.h"
#include "TrackPropagation/NavGeometry/interface/NavVolumeSide.h"

#include <vector>

class Bounds;
class Plane;
class TrajectoryStateOnSurface;

class NavVolume6Faces final : public NavVolume {
public:

    NavVolume6Faces( const PositionType& pos, const RotationType& rot, 
		     DDSolidShape shape, const std::vector<NavVolumeSide>& faces,
		     const MagneticFieldProvider<float> * mfp);

    /// A NavVolume6Faces that corresponds exactly to a MagVolume
    explicit NavVolume6Faces( const MagVolume& magvol, const bool isIron=false);

    /// Give a sorted list of possible surfaces to propagate to
    Container nextSurface( const NavVolume::LocalPoint& pos, 
				   const NavVolume::LocalVector& mom,
				   double charge, PropagationDirection propDir=alongMomentum) const override;

    /// Same, giving lowest priority to the surface we are on now (=NotThisSurface)
    Container nextSurface( const NavVolume::LocalPoint& pos, 
				   const NavVolume::LocalVector& mom,
				   double charge, PropagationDirection propDir,
				   ConstReferenceCountingPointer<Surface> NotThisSurfaceP) const override;

    /// Cross this volume and point at the next
    VolumeCrossReturnType crossToNextVolume( const TrajectoryStateOnSurface& currentState,
						     const Propagator& prop) const override;

    /// Access to volume faces
    const std::vector<VolumeSide>& faces() const override { return theFaces; }

    /// Access to Iron/Air information:
    bool isIron() const override { return isThisIron; }

    using MagVolume::inside;
    bool inside( const GlobalPoint& gp, double tolerance) const override; 

private:

    std::vector<VolumeSide> theFaces;
    Container theNavSurfaces;
    bool isThisIron; 

    void computeBounds(const std::vector<NavVolumeSide>& faces);
    Bounds* computeBounds( int index, const std::vector<const Plane*>& bpc);
    Bounds* computeBounds( int index, const std::vector<NavVolumeSide>& faces);

};

/* bool NavVolume6Faces::inside( const GlobalPoint& gp, double tolerance) const 
{
   
  // check if the point is on the correct side of all delimiting surfaces
  for (std::vector<VolumeSide>::const_iterator i=theFaces.begin(); i!=theFaces.end(); i++) {
    Surface::Side side = i->surface().side( gp, tolerance);
    if ( side != i->surfaceSide() && side != SurfaceOrientation::onSurface) return false;
  }
  return true;
}
*/

#endif
