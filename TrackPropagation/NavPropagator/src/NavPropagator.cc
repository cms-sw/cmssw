#include "TrackPropagation/NavPropagator/interface/NavPropagator.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include  "TrackPropagation/NavGeometry/interface/NavVolume6Faces.h"
#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

NavPropagator::NavPropagator( const MagneticField* field,
			      PropagationDirection dir) :
  Propagator(dir)
{
  theField = dynamic_cast<const VolumeBasedMagneticField*>(field);
}

NavPropagator::~NavPropagator() {
    for (MagVolumeMap::iterator i = theNavVolumeMap.begin(); i != theNavVolumeMap.end(); ++i) {
      delete i->second;
    }
}

const MagneticField*  NavPropagator::magneticField() const {return theField;}

std::pair<TrajectoryStateOnSurface,double> 
NavPropagator::propagateWithPath(const TrajectoryStateOnSurface& inputState, 
				 const Plane& targetPlane) const
{

   LogDebug("NavPropagator") <<  "NavPropagator::propagateWithPath(TrajectoryStateOnSurface, Plane) called with "
       << inputState;

  VolumeCrossReturnType exitState( 0, inputState, 0.0);
  TSOS startingState = inputState;
  TSOS TempState = inputState;
  const NavVolume* currentVolume;

  int count = 0;
  int maxCount = 100;
  while (!destinationCrossed( startingState, exitState.tsos(), targetPlane)) {

     LogDebug("NavPropagator") <<  "NavPropagator:: at beginning of while loop at iteration " << count ;

    startingState = TempState;
    if (exitState.volume() != 0) { // next volume connected
      currentVolume = exitState.volume();
    }
    else {
      currentVolume = findVolume( startingState);
      if (currentVolume == 0) {
	 LogDebug("NavPropagator") <<  "NavPropagator: findVolume failed to find volume containing point "
	     << startingState.globalPosition() ;
	return std::pair<TrajectoryStateOnSurface,double>(noNextVolume( startingState), 0);
      }
    }

    PropagatorType currentPropagator( *currentVolume);

     LogDebug("NavPropagator") <<  "NavPropagator: calling crossToNextVolume" ;
    exitState = currentVolume->crossToNextVolume( startingState, currentPropagator);
     LogDebug("NavPropagator") <<  "NavPropagator: crossToNextVolume returned" ;
     LogDebug("NavPropagator") <<  "Volume pointer: " << exitState.volume() << " and new ";
     LogDebug("NavPropagator") <<  exitState.tsos() ;
     LogDebug("NavPropagator") <<  "So that was a path length " << exitState.path() ;


    if ( !exitState.tsos().isValid()) {
      // return propagateInVolume( currentVolume, startingState, targetPlane);
       LogDebug("NavPropagator") <<  "NavPropagator: failed to crossToNextVolume in volume at pos "
	   << currentVolume->position();
      break;
    }
    else {
       LogDebug("NavPropagator") <<  "NavPropagator: crossToNextVolume reached volume ";
      if (exitState.volume() != 0) {	   
	 LogDebug("NavPropagator") <<  " at pos "
	     << exitState.volume()->position() 
	     << "with state " << exitState.tsos() ;
      }
      else  LogDebug("NavPropagator") <<  " unknown";
    }

    TempState = exitState.tsos();

    if (fabs(exitState.path())<0.01) {
      LogDebug("NavPropagator") <<  "failed to move at all!! at position: " << exitState.tsos().globalPosition();

      GlobalTrajectoryParameters gtp( exitState.tsos().globalPosition()+0.01*exitState.tsos().globalMomentum().unit(),
				      exitState.tsos().globalMomentum(),
				      exitState.tsos().globalParameters().charge(), theField );

      FreeTrajectoryState fts(gtp);
      TSOS ShiftedState( fts, exitState.tsos().surface());
      //      exitState.tsos() = ShiftedState;
      TempState = ShiftedState;

      LogDebug("NavPropagator") <<  "Shifted to new position " << TempState.globalPosition();

    }

    ++count;
    if (count > maxCount) {
       LogDebug("NavPropagator") <<  "Ohoh, NavPropagator in infinite loop, count = " << count;
      return TsosWP();
    }
  }

   LogDebug("NavPropagator") <<  "NavPropagator: calling propagateInVolume to reach final destination" ;

  // Arriving here only if destinationCrossed or if crossToNextVolume returned invalid state,
  // in which case we should check if the targetPlane is not crossed in the current volume

  return propagateInVolume( currentVolume, startingState, targetPlane);  

}


const NavVolume* NavPropagator::findVolume( const TrajectoryStateOnSurface& inputState) const
{
   LogDebug("NavPropagator") <<  "NavPropagator: calling theField->findVolume ";

  GlobalPoint gp = inputState.globalPosition() + 0.1*inputState.globalMomentum().unit();

  const MagVolume* magVolume = theField->findVolume( gp);

   LogDebug("NavPropagator") <<  "NavPropagator: got MagVolume* " << magVolume 
       << " when searching with pos " << gp ;

  return navVolume(magVolume);
}


const NavVolume* NavPropagator::navVolume( const MagVolume* magVolume)  const
{
  NavVolume* result;
  MagVolumeMap::iterator i = theNavVolumeMap.find( magVolume);
  if (i == theNavVolumeMap.end()) {
    result= new NavVolume6Faces( *magVolume);
    theNavVolumeMap[magVolume] = result;
  }
  else result = i->second;
  return result;
}

std::pair<TrajectoryStateOnSurface,double> 
NavPropagator::propagateInVolume( const NavVolume* currentVolume, 
				  const TrajectoryStateOnSurface& startingState, 
				  const Plane& targetPlane) const
{
  PropagatorType prop( *currentVolume);
  TsosWP res = prop.propagateWithPath( startingState, targetPlane);
  if (res.first.isValid()) { 
    if (currentVolume->inside( res.first.globalPosition())) {
      return res;
    }
  } 
  return TsosWP( TrajectoryStateOnSurface(), 0);
}

bool  NavPropagator::destinationCrossed( const TSOS& startState,
					 const TSOS& endState, const Plane& plane) const
{
  bool res = ( plane.side( startState.globalPosition(), 1.e-6) != 
	       plane.side( endState.globalPosition(), 1.e-6));
  /*
   LogDebug("NavPropagator") <<  "NavPropagator::destinationCrossed called with startState "
       << startState << std::endl
       << " endState " << endState 
       << std::endl
       << " plane at " << plane.position()
       << std::endl;

   LogDebug("NavPropagator") <<  "plane.side( startState) " << plane.side( startState.globalPosition(), 1.e-6);
   LogDebug("NavPropagator") <<  "plane.side( endState)   " << plane.side( endState.globalPosition(), 1.e-6);
  */

  return res;
}

TrajectoryStateOnSurface 
NavPropagator::noNextVolume( TrajectoryStateOnSurface startingState) const
{
   LogDebug("NavPropagator") <<  std::endl
       << "Propagation reached end of volume geometry without crossing target surface" 
       << std::endl
       << "starting state of last propagation " << startingState;

  return TrajectoryStateOnSurface();
}



std::pair<TrajectoryStateOnSurface,double> 
NavPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
				 const Cylinder& cylinder) const
{
   LogDebug("NavPropagator") <<  "propagateWithPath(const FreeTrajectoryState&) not implemented yet in NavPropagator" ;
  return TsosWP();
}
std::pair<TrajectoryStateOnSurface,double> 
NavPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
				 const Plane& cylinder) const
{
   LogDebug("NavPropagator") <<  "propagateWithPath(const FreeTrajectoryState&) not implemented yet in NavPropagator" ;
  return TsosWP();
}
