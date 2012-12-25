/******* \class SmartPropagator *******
 *
 * Description: A propagator which use different algorithm to propagate inside
 * or outside tracker
 *  
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 * \porting author: Chang Liu - Purdue University 
 * $Date: 2009/07/05 19:56:54 $
 * $Revision: 1.7 $
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"

/* Collaborating Class Header */
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"

#include <DataFormats/GeometrySurface/interface/BoundCylinder.h>

#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/* Base Class Headers */

/* C++ Headers */

/* ====================================================================== */

/* static member */
ReferenceCountingPointer<BoundCylinder> & SmartPropagator::theTkVolume() {
  static ReferenceCountingPointer<BoundCylinder> local=0;
  return local;
}



/* Constructor */ 
SmartPropagator::SmartPropagator(Propagator* aTkProp, Propagator* aGenProp, const MagneticField* field,
                                 PropagationDirection dir, float epsilon) :
  Propagator(dir), theTkProp(aTkProp->clone()), theGenProp(aGenProp->clone()), theField(field) { 

  if (theTkVolume()==0) initTkVolume(epsilon);

}


SmartPropagator::SmartPropagator(const Propagator& aTkProp, const Propagator& aGenProp,const MagneticField* field,
                                 PropagationDirection dir, float epsilon) :
  Propagator(dir), theTkProp(aTkProp.clone()), theGenProp(aGenProp.clone()), theField(field) {

  if (theTkVolume()==0) initTkVolume(epsilon);

}


SmartPropagator::SmartPropagator(const SmartPropagator& aProp) :
  Propagator(aProp.propagationDirection()), theTkProp(0), theGenProp(0) { 
    if (aProp.theTkProp)
      theTkProp=aProp.getTkPropagator()->clone();
    if (aProp.theGenProp)
      theTkProp=aProp.getGenPropagator()->clone();

    //SL since it's a copy constructor, then the TkVolume has been already
    //initialized
    //if (theTkVolume==0) initTkVolume(epsilon);

  }

/* Destructor */ 
SmartPropagator::~SmartPropagator() {

  delete theTkProp;
  delete theGenProp;

}


/* Operations */ 
void SmartPropagator::initTkVolume(float epsilon) {

  //
  // fill tracker dimensions
  //
  float radius = TrackerBounds::radius();
  float r_out  = radius + epsilon/2;
  float r_in   = r_out - epsilon;
  float z_max  = TrackerBounds::halfLength();
  float z_min  = - z_max;

  Surface::PositionType pos(0,0,0); // centered at the global origin
  Surface::RotationType rot; // unit matrix - barrel cylinder orientation

  theTkVolume() = Cylinder::build(radius, pos, rot, new SimpleCylinderBounds(r_in, r_out, z_min, z_max));

}


TrajectoryStateOnSurface SmartPropagator::propagate(const FreeTrajectoryState& fts, 
                                                    const Surface& surface) const {
  return Propagator::propagate( fts, surface);
}


TrajectoryStateOnSurface SmartPropagator::propagate(const FreeTrajectoryState& fts, 
                                                    const Plane& plane) const {

  if (insideTkVol(fts) && insideTkVol(plane)) {
    return getTkPropagator()->propagate(fts, plane);
  } else {
    return getGenPropagator()->propagate(fts, plane);
  }

}


TrajectoryStateOnSurface SmartPropagator::propagate(const FreeTrajectoryState& fts, 
                                                    const Cylinder& cylinder) const {
  if (insideTkVol(fts) && insideTkVol(cylinder)) {
    return getTkPropagator()->propagate(fts, cylinder);
  } else {
    return getGenPropagator()->propagate(fts, cylinder);
  }

}

std::pair<TrajectoryStateOnSurface,double> 
SmartPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
                                   const Plane& plane) const 
{
  if (insideTkVol(fts) && insideTkVol(plane)) {
    return getTkPropagator()->propagateWithPath(fts, plane);
  } else {
    return getGenPropagator()->propagateWithPath(fts, plane);
  }
}

std::pair<TrajectoryStateOnSurface,double> 
SmartPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
                                   const Cylinder& cylinder) const
{
  if (insideTkVol(fts) && insideTkVol(cylinder)) {
    return getTkPropagator()->propagateWithPath(fts, cylinder);
  } else {
    return getGenPropagator()->propagateWithPath(fts, cylinder);
  }
}

bool SmartPropagator::insideTkVol(const FreeTrajectoryState& fts) const {

  GlobalPoint gp = fts.position();
//  LocalPoint lp = theTkVolume()->toLocal(gp);
//  return theTkVolume()->bounds().inside(lp);

  return ( (gp.perp()<= TrackerBounds::radius()+10.) && (fabs(gp.z())<= TrackerBounds::halfLength()+10.) );
}


bool SmartPropagator::insideTkVol(const Surface& surface) const {

  GlobalPoint gp = surface.position();
 // LocalPoint lp = theTkVolume()->toLocal(gp);

 // return theTkVolume()->bounds().inside(lp);
  return ( (gp.perp()<= TrackerBounds::radius()+10.) && (fabs(gp.z())<= TrackerBounds::halfLength()+10.) );


}


bool SmartPropagator::insideTkVol( const BoundCylinder& cylin)  const {

  GlobalPoint gp(cylin.radius(),0.,(cylin.bounds().length())/2.);
//  LocalPoint lp = theTkVolume()->toLocal(gp);
//  return theTkVolume()->bounds().inside(lp);
  return ( (gp.perp()<= TrackerBounds::radius()+10.) && (fabs(gp.z())<= TrackerBounds::halfLength()+10.) );


}


bool SmartPropagator::insideTkVol( const Plane& plane)  const {

  GlobalPoint gp = plane.position();
//  LocalPoint lp = theTkVolume()->toLocal(gp);
//  return theTkVolume()->bounds().inside(lp);
  return ( (gp.perp()<= TrackerBounds::radius()+10.) && (fabs(gp.z())<= TrackerBounds::halfLength()+10.) );


}


Propagator* SmartPropagator::getTkPropagator() const {

  return theTkProp;

}


Propagator* SmartPropagator::getGenPropagator() const {

  return theGenProp;

}


