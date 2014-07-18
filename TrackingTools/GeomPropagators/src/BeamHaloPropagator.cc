/******* \class BeamHaloPropagator *******
 *
 * Description: A propagator which use different algorithm to propagate
 * within an endcap or to cross over to the other endcap
 *
 *
 * \author : Jean-Roch VLIMANT UCSB
 *
 *********************************/

/* This Class Header */
#include "TrackingTools/GeomPropagators/interface/BeamHaloPropagator.h"

/* Collaborating Class Header */
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"

#include <DataFormats/GeometrySurface/interface/Cylinder.h>

#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/* Base Class Headers */

/* C++ Headers */

/* ====================================================================== */




/* Constructor */
void BeamHaloPropagator::directionCheck(PropagationDirection dir) {

  //consistency check for direction
  if (getEndCapTkPropagator()->propagationDirection() != dir &&
      getEndCapTkPropagator()->propagationDirection() != anyDirection) {
    edm::LogError("BeamHaloPropagator")<<"composite propagator set with inconsistent direction components\n"
				  <<"EndCap propagator is: "<<getEndCapTkPropagator()->propagationDirection()
				  <<"\n to be set on: "<<dir;
    theEndCapTkProp->setPropagationDirection(dir);
  }

  if (getCrossTkPropagator()->propagationDirection() != dir &&
      getCrossTkPropagator()->propagationDirection() != anyDirection) {
    edm::LogError("BeamHaloPropagator")<<"composite propagator set with inconsistent direction components\n"
				  <<"Cross propagator is: "<<getCrossTkPropagator()->propagationDirection()
				  <<"\n to be set on: "<<dir;
    theCrossTkProp->setPropagationDirection(dir);
  }
}

BeamHaloPropagator::BeamHaloPropagator(const Propagator* aEndCapTkProp, const Propagator* aCrossTkProp, const MagneticField* field,
                                 PropagationDirection dir) :
  Propagator(dir), theEndCapTkProp(aEndCapTkProp->clone()), theCrossTkProp(aCrossTkProp->clone()), theField(field) {
  directionCheck(dir);
}


BeamHaloPropagator::BeamHaloPropagator(const Propagator& aEndCapTkProp, const Propagator& aCrossTkProp,const MagneticField* field,
                                 PropagationDirection dir) :
  Propagator(dir), theEndCapTkProp(aEndCapTkProp.clone()), theCrossTkProp(aCrossTkProp.clone()), theField(field) {
  directionCheck(dir);
}


BeamHaloPropagator::BeamHaloPropagator(const BeamHaloPropagator& aProp) :
  Propagator(aProp.propagationDirection()), theEndCapTkProp(0), theCrossTkProp(0) {
  if (aProp.theEndCapTkProp)
    theEndCapTkProp=aProp.getEndCapTkPropagator()->clone();
  if (aProp.theCrossTkProp)
    theCrossTkProp=aProp.getCrossTkPropagator()->clone();
}

/* Destructor */
BeamHaloPropagator::~BeamHaloPropagator() {

  delete theEndCapTkProp;
  delete theCrossTkProp;

}

bool BeamHaloPropagator::crossingTk(const FreeTrajectoryState& fts, const Plane& plane)  const{
  LogDebug("BeamHaloPropagator")<<"going from: "<<fts.position()<<" to: "<<plane.position()<<"\n"
				<<"and hence "<<((fts.position().z()*plane.position().z()<0)?"crossing":"not crossing");
  return (fts.position().z()*plane.position().z()<0);}

TrajectoryStateOnSurface BeamHaloPropagator::propagate(const FreeTrajectoryState& fts,
                                                    const Surface& surface) const {
  return Propagator::propagate( fts, surface);
}


TrajectoryStateOnSurface BeamHaloPropagator::propagate(const FreeTrajectoryState& fts,
                                                    const Plane& plane) const {

  if (crossingTk(fts,plane)){
    return getCrossTkPropagator()->propagate(fts, plane);}
  else{
    return getEndCapTkPropagator()->propagate(fts, plane);}
}


TrajectoryStateOnSurface BeamHaloPropagator::propagate(const FreeTrajectoryState& fts,
                                                    const Cylinder& cylinder) const {
  return getCrossTkPropagator()->propagate(fts, cylinder);
}

std::pair<TrajectoryStateOnSurface,double>
BeamHaloPropagator::propagateWithPath(const FreeTrajectoryState& fts,
                                   const Plane& plane) const
{
  if (crossingTk(fts,plane)){
    return getCrossTkPropagator()->propagateWithPath(fts, plane);}
  else{
    return getEndCapTkPropagator()->propagateWithPath(fts, plane);}
}

std::pair<TrajectoryStateOnSurface,double>
BeamHaloPropagator::propagateWithPath(const FreeTrajectoryState& fts,
                                   const Cylinder& cylinder) const
{  return getCrossTkPropagator()->propagateWithPath(fts, cylinder);}


const Propagator* BeamHaloPropagator::getEndCapTkPropagator() const {
  LogDebug("BeamHaloPropagator")<<"using the EndCap propagator";
  return theEndCapTkProp;}


const Propagator* BeamHaloPropagator::getCrossTkPropagator() const {
  LogDebug("BeamHaloPropagator")<<"using the Crossing propagator";
  return theCrossTkProp;}


