#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

//Ported from ORCA
//  $Date: 2011/02/08 14:59:45 $
//  $Revision: 1.9 $

StateOnTrackerBound::StateOnTrackerBound( const Propagator* prop) :
  thePropagator( prop->clone())
{}
 
StateOnTrackerBound::~StateOnTrackerBound()
{
  delete thePropagator;
}

TrajectoryStateOnSurface
StateOnTrackerBound::operator()(const TrajectoryStateOnSurface& tsos) const
{
  return operator()(*tsos.freeState());
}

TrajectoryStateOnSurface 
StateOnTrackerBound::operator()(const FreeTrajectoryState& fts) const
{
  // try to guess if propagation should be first to cylinder or first to disk
  float tanTheta = (fts.position().perp() > 70 ) ? fts.position().perp()/fts.position().z() : fts.momentum().perp()/fts.momentum().z();
  float corner = TrackerBounds::radius() / TrackerBounds::halfLength();

  TrajectoryStateOnSurface firstTry;
  if (tanTheta < 0 && fabs(tanTheta) < corner) {
     firstTry = 
     thePropagator->propagate( fts, TrackerBounds::negativeEndcapDisk());

     if (!firstTry.isValid()) {
       return thePropagator->propagate( fts, TrackerBounds::barrelBound());
      }
     if (firstTry.globalPosition().perp() > TrackerBounds::radius()) {
        // the propagation should have gone to the cylinder
        return thePropagator->propagate( fts, TrackerBounds::barrelBound());
      }
      else return firstTry;
  }
  else if (tanTheta > 0 && fabs(tanTheta) < corner) {
      firstTry = 
      thePropagator->propagate( fts, TrackerBounds::positiveEndcapDisk());
      if (!firstTry.isValid()) {
        return thePropagator->propagate( fts, TrackerBounds::barrelBound());
      }
      if (firstTry.globalPosition().perp() > TrackerBounds::radius()) {
        return thePropagator->propagate( fts, TrackerBounds::barrelBound());
      }
      else return firstTry;
  }
  else {
    // barrel
    firstTry = 
    thePropagator->propagate( fts, TrackerBounds::barrelBound());
    if (!firstTry.isValid()) {
       if (tanTheta < 0 ) return thePropagator->propagate( fts,TrackerBounds::negativeEndcapDisk());
       if (tanTheta >= 0 ) return thePropagator->propagate( fts,TrackerBounds::positiveEndcapDisk());
       return firstTry;
     }
     if (firstTry.globalPosition().z() < -TrackerBounds::halfLength()) {
        // the propagation should have gone to the negative disk
        return thePropagator->propagate( fts,
                                         TrackerBounds::negativeEndcapDisk());
     }
     else if (firstTry.globalPosition().z() > TrackerBounds::halfLength()) {
        // the propagation should have gone to the positive disk
        return thePropagator->propagate( fts,
                                         TrackerBounds::positiveEndcapDisk());
     }
     else return firstTry;
  }    
  return firstTry;
}
