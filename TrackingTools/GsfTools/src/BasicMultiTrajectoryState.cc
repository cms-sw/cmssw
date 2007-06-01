#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

BasicMultiTrajectoryState::BasicMultiTrajectoryState( const std::vector<TSOS>& tsvec) :
  theCombinedStateUp2Date( false)
{
  for (std::vector<TSOS>::const_iterator i=tsvec.begin(); i!=tsvec.end(); i++) {
    if (!i->isValid()) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState constructed with invalid state";
    }
    if (i->hasError() != tsvec.front().hasError()) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState mixes states with and without errors";
    }
    if ( &i->surface() != &tsvec.front().surface()) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState mixes states with different surfaces";
    }
    if ( i->surfaceSide() != tsvec.front().surfaceSide()) {
      throw cms::Exception("LogicError") 
	<< "MultiTrajectoryState mixes states defined before and after material";
    }
    if ( i->localParameters().pzSign()*tsvec.front().localParameters().pzSign()<0. ) {
      throw cms::Exception("LogicError") 
	<< "MultiTrajectoryState mixes states with different signs of local p_z";
    }
    if ( i==tsvec.begin() ) {
      // only accept planes!!
      const BoundPlane* bp = dynamic_cast<const BoundPlane*>(&i->surface());
      if ( bp==0 )
	throw cms::Exception("LogicError") << "MultiTrajectoryState constructed on cylinder";
    }
    theStates.push_back( *i);
  }
}

void BasicMultiTrajectoryState::checkCombinedState() const
{
  if (theCombinedStateUp2Date) return;

  theCombinedState = theCombiner.combine( theStates);
  theCombinedStateUp2Date = true;
  
}

double BasicMultiTrajectoryState::weight() const {

  if (theStates.empty()) {
    edm::LogError("BasicMultiTrajectoryState") 
      << "Asking for weight of empty MultiTrajectoryState, returning zero!";
    return 0.;
  }

  double sumw = 0.;
  for (std::vector<TSOS>::const_iterator it = theStates.begin(); it != theStates.end(); it++) {
    sumw += it->weight();
  }
  return sumw;
}


void BasicMultiTrajectoryState::rescaleError(double factor) {

  if (theStates.empty()) {
    edm::LogError("BasicMultiTrajectoryState") << "Trying to rescale errors of empty MultiTrajectoryState!";
    return;
  }
  
  for (std::vector<TSOS>::iterator it = theStates.begin(); it != theStates.end(); it++) {
    it->rescaleError(factor);
  }
  theCombinedStateUp2Date = false;
}

const MagneticField*
BasicMultiTrajectoryState::magneticField () const
{
  //
  // Magnetic field should be identical in all components:
  // avoid forcing the combination of states and take value from 1st component!
  //
  if (theStates.empty()) {
    edm::LogError("BasicMultiTrajectoryState") 
      << "Asking for magneticField of empty MultiTrajectoryState, returning null pointer!";
    return 0;
  }
  return theStates.front().magneticField();
}

SurfaceSide
BasicMultiTrajectoryState::surfaceSide () const
{
  //
  // SurfaceSide should be identical in all components:
  // avoid forcing the combination of states and take value from 1st component!
  //
  if (theStates.empty()) {
    edm::LogError("BasicMultiTrajectoryState") 
      << "Asking for magneticField of empty MultiTrajectoryState, returning atCenterOfSurface!";
    return atCenterOfSurface;
  }
  return theStates.front().surfaceSide();
}

