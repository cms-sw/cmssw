#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/BoundCylinder.h"

//Ported from ORCA
//  $Date: 2006/04/21 14:30:11 $
//  $Revision: 1.1 $

StateOnTrackerBound::StateOnTrackerBound( Propagator* prop) :
  thePropagator( prop->clone())
{}
 
StateOnTrackerBound::~StateOnTrackerBound()
{
  delete thePropagator;
};

TrajectoryStateOnSurface 
StateOnTrackerBound::operator()(const FreeTrajectoryState& fts) const
{
  // try to guess if propagation should be first to cylinder or first to disk
  float tanTheta = fts.position().perp()/fts.position().z();
  float corner = TrackerBounds::radius() / TrackerBounds::halfLength();

  TrajectoryStateOnSurface firstTry;
  if (tanTheta < 0 && fabs(tanTheta) < corner) {
    firstTry = 
      thePropagator->propagate( fts, TrackerBounds::negativeEndcapDisk());
    return checkDisk( firstTry);
  }
  else if (tanTheta > 0 && fabs(tanTheta) < corner) {
    firstTry = 
      thePropagator->propagate( fts, TrackerBounds::positiveEndcapDisk());
    return checkDisk( firstTry);
  }
  else {
    // barrel
    firstTry = 
      thePropagator->propagate( fts, TrackerBounds::barrelBound());
    return checkCylinder( firstTry);
  }    
}

TrajectoryStateOnSurface StateOnTrackerBound::
checkDisk(const TrajectoryStateOnSurface& firstTry) const
{
  if (!firstTry.isValid()) return firstTry;
  if (firstTry.globalPosition().perp() > TrackerBounds::radius()) {
    // the propagation should have gone to the cylinder
    return thePropagator->propagate( firstTry, TrackerBounds::barrelBound());
  }
  else return firstTry;
}

TrajectoryStateOnSurface StateOnTrackerBound::
checkCylinder(const TrajectoryStateOnSurface& firstTry) const
{
  if (!firstTry.isValid()) return firstTry;
  if (firstTry.globalPosition().z() < -TrackerBounds::halfLength()) {
    // the propagation should have gone to the negative disk
    return thePropagator->propagate( firstTry, 
				     TrackerBounds::negativeEndcapDisk());
  }
  else if (firstTry.globalPosition().z() > TrackerBounds::halfLength()) {
    // the propagation should have gone to the positive disk
    return thePropagator->propagate( firstTry, 
				     TrackerBounds::positiveEndcapDisk());
  }
  else return firstTry;
}

