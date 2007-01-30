#include "TrackPropagation/NavPropagator/interface/NavPropagator.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include  "TrackPropagation/NavGeometry/interface/NavVolume6Faces.h"
#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"

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

  cout << "NavPropagator::propagateWithPath(TrajectoryStateOnSurface, Plane) called with "
       << inputState << endl;

  VolumeCrossReturnType exitState( 0, inputState, 0.0);
  TSOS startingState = inputState;
  TSOS TempState = inputState;
  const NavVolume* currentVolume;

  int count = 0;
  int maxCount = 100;
  while (!destinationCrossed( startingState, exitState.tsos(), targetPlane)) {

    cout << "NavPropagator:: at beginning of while loop at iteration " << count << endl;

    startingState = TempState;
    if (exitState.volume() != 0) { // next volume connected
      currentVolume = exitState.volume();
    }
    else {
      currentVolume = findVolume( startingState);
      if (currentVolume == 0) {
	cout << "NavPropagator: findVolume failed to find volume containing point "
	     << startingState.globalPosition() << endl;
	return std::pair<TrajectoryStateOnSurface,double>(noNextVolume( startingState), 0);
      }
    }

    PropagatorType currentPropagator( *currentVolume);

    cout << "NavPropagator: calling crossToNextVolume" << endl;
    exitState = currentVolume->crossToNextVolume( startingState, currentPropagator);
    cout << "NavPropagator: crossToNextVolume returned" << endl;
    cout << "Volume pointer: " << exitState.volume() << " and new " << endl;
    cout << exitState.tsos() << endl;
    cout << "So that was a path length " << exitState.path() << endl;


    if ( !exitState.tsos().isValid()) {
      // return propagateInVolume( currentVolume, startingState, targetPlane);
      cout << "NavPropagator: failed to crossToNextVolume in volume at pos "
	   << currentVolume->position() << endl;
      break;
    }
    else {
      cout << "NavPropagator: crossToNextVolume reached volume ";
      if (exitState.volume() != 0) {	   
	cout << " at pos "
	     << exitState.volume()->position() 
	     << "with state " << exitState.tsos() 
	     << endl;
      }
      else cout << " unknown" << endl;
    }

    TempState = exitState.tsos();

    if (fabs(exitState.path())<0.01) {
      std::cout << "failed to move at all!! at position: " << exitState.tsos().globalPosition() << std::endl;

      GlobalTrajectoryParameters gtp( exitState.tsos().globalPosition()+0.01*exitState.tsos().globalMomentum().unit(),
				      exitState.tsos().globalMomentum(),
				      exitState.tsos().globalParameters().charge(), theField );

      FreeTrajectoryState fts(gtp);
      TSOS ShiftedState( fts, exitState.tsos().surface());
      //      exitState.tsos() = ShiftedState;
      TempState = ShiftedState;

      std::cout << "Shifted to new position " << TempState.globalPosition() <<std::endl;

    }

    ++count;
    if (count > maxCount) {
      cout << "Ohoh, NavPropagator in infinite loop, count = " << count << endl;
      return TsosWP();
    }
  }

  cout << "NavPropagator: calling propagateInVolume to reach final destination" << endl;

  // Arriving here only if destinationCrossed or if crossToNextVolume returned invalid state,
  // in which case we should check if the targetPlane is not crossed in the current volume

  return propagateInVolume( currentVolume, startingState, targetPlane);  

}


const NavVolume* NavPropagator::findVolume( const TrajectoryStateOnSurface& inputState) const
{
  cout << "NavPropagator: calling theField->findVolume " << endl;

  GlobalPoint gp = inputState.globalPosition() + 0.1*inputState.globalMomentum().unit();

  const MagVolume* magVolume = theField->findVolume( gp);

  cout << "NavPropagator: got MagVolume* " << magVolume 
       << " when searching with pos " << gp << endl;

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
  cout << "NavPropagator::destinationCrossed called with startState "
       << startState << endl
       << " endState " << endState 
       << endl
       << " plane at " << plane.position()
       << endl;

  cout << "plane.side( startState) " << plane.side( startState.globalPosition(), 1.e-6) << endl;
  cout << "plane.side( endState)   " << plane.side( endState.globalPosition(), 1.e-6) << endl;
  */

  return res;
}

TrajectoryStateOnSurface 
NavPropagator::noNextVolume( TrajectoryStateOnSurface startingState) const
{
  cout << endl
       << "Propagation reached end of volume geometry without crossing target surface" 
       << endl
       << "starting state of last propagation " << startingState << endl;

  return TrajectoryStateOnSurface();
}



std::pair<TrajectoryStateOnSurface,double> 
NavPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
				 const Cylinder& cylinder) const
{
  cout << "propagateWithPath(const FreeTrajectoryState&) not implemented yet in NavPropagator" << endl;
  return TsosWP();
}
std::pair<TrajectoryStateOnSurface,double> 
NavPropagator::propagateWithPath(const FreeTrajectoryState& fts, 
				 const Plane& cylinder) const
{
  cout << "propagateWithPath(const FreeTrajectoryState&) not implemented yet in NavPropagator" << endl;
  return TsosWP();
}
