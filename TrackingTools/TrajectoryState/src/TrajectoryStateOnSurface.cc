#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"

typedef BasicSingleTrajectoryState              BTSOS;

TrajectoryStateOnSurface::
TrajectoryStateOnSurface(const Surface& aSurface) :
  Base( new BTSOS(aSurface)) {}

TrajectoryStateOnSurface::
TrajectoryStateOnSurface(const FreeTrajectoryState& fts,
			 const Surface& aSurface, const SurfaceSide side) :
  Base( new BTSOS( fts, aSurface, side)) {}


TrajectoryStateOnSurface::
TrajectoryStateOnSurface(const GlobalTrajectoryParameters& gp,
			 const Surface& aSurface, const SurfaceSide side) :
  Base( new BTSOS( gp, aSurface, side)) {}

TrajectoryStateOnSurface::
TrajectoryStateOnSurface( const GlobalTrajectoryParameters& gp,
			  const CartesianTrajectoryError& err,
			  const Surface& aSurface, const SurfaceSide side) :
  Base( new BTSOS( gp, err, aSurface, side)) {}

TrajectoryStateOnSurface::
TrajectoryStateOnSurface( const GlobalTrajectoryParameters& gp,
			  const CurvilinearTrajectoryError& err,
			  const Surface& aSurface, const SurfaceSide side, double weight) :
  Base( new BTSOS( gp, err, aSurface, side, weight)) {}

TrajectoryStateOnSurface::
TrajectoryStateOnSurface( const GlobalTrajectoryParameters& gp,
			  const CurvilinearTrajectoryError& err,
			  const Surface& aSurface, double weight) :
  Base( new BTSOS( gp, err, aSurface, SurfaceSideDefinition::atCenterOfSurface, weight)) {}

TrajectoryStateOnSurface::
TrajectoryStateOnSurface( const LocalTrajectoryParameters& p,
			  const Surface& aSurface, 
			  const MagneticField* field, 
			  const SurfaceSide side) :
  Base( new BTSOS( p, aSurface, field, side)) {}

TrajectoryStateOnSurface::
TrajectoryStateOnSurface( const LocalTrajectoryParameters& p,
			  const LocalTrajectoryError& err,
			  const Surface& aSurface, 
			  const MagneticField* field, 
			  const SurfaceSide side, double weight) :
  Base( new BTSOS( p, err, aSurface, field, side, weight)) {}

TrajectoryStateOnSurface::
TrajectoryStateOnSurface( const LocalTrajectoryParameters& p,
			  const LocalTrajectoryError& err,
			  const Surface& aSurface, 
			  const MagneticField* field, 
			  double weight) :
  Base( new BTSOS( p, err, aSurface, field, SurfaceSideDefinition::atCenterOfSurface, weight)) {}


void
TrajectoryStateOnSurface::
update( const LocalTrajectoryParameters& p,
        const Surface& aSurface,
        const MagneticField* field,
        const SurfaceSide side) 
{
    if (data().canUpdateLocalParameters()) {
        unsharedData().update(p, aSurface, field, side);
    } else {
        *this = TrajectoryStateOnSurface(p, aSurface, field, side);
    }
}

void
TrajectoryStateOnSurface::
update( const LocalTrajectoryParameters& p,
        const LocalTrajectoryError& err,
        const Surface& aSurface,
        const MagneticField* field,
        const SurfaceSide side, 
        double weight) 
{
   if (data().canUpdateLocalParameters()) {
        unsharedData().update(p, err, aSurface, field, side, weight);
    } else {
        *this = TrajectoryStateOnSurface(p, err, aSurface, field, side, weight);
    }
}
