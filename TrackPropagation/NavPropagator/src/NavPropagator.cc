#include "TrackPropagation/NavPropagator/interface/NavPropagator.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include  "TrackPropagation/NavGeometry/interface/NavVolume6Faces.h"
// magVolume6Faces needed to get access to volume name and material:
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
//
#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

#include "TrackingTools/MaterialEffects/interface/VolumeMaterialEffectsEstimate.h"

using namespace std;

NavPropagator::NavPropagator( const MagneticField* field,
			      PropagationDirection dir) :
  Propagator(dir),
  theAirMedium(3.e4, 1.3e-3*0.000307075*0.5/2),
  theIronMedium(1.76,7.87*0.000307075*0.46556/2),
  theMSEstimator(0.105), theELEstimator(0.105)
{
  theField = dynamic_cast<const VolumeBasedMagneticField*>(field);
  //FIXME: iron volumes are hard-coded below... will change with new .xml geometry MM 28/6/07
  int myIronVolumes[126] = {6,9,11,14,15,18,22,23,26,29,30,33,36,37,43,46,49,53,56,57,60,62,
			    63,65,71,77,105,106,107,111,112,113,114,115,116,117,118,122,123,
			    125,126,128,129,130,131,132,133,135,137,141,145,147,148,149,153,
			    154,155,156,157,158,159,160,164,165,167,168,170,171,172,173,174,
			    175,177,178,179,180,184,185,186,187,188,189,190,191,195,196,198,
			    200,204,208,210,211,212,216,217,218,219,220,221,222,223,227,228,
			    230,231,233,234,235,236,237,238,240,241,242,243,247,248,249,250,
			    251,252,253,254,258,259,261}; 
  for(int i=0; i<272; i++) isIronVolume[i]=false;
  for(int i=0; i<126 ; i++) isIronVolume[myIronVolumes[i]]=true; 
  //  for(int i=1; i<272 ; i++) {
  //  if(isIronVolume[i]) std::cout << "Volume no." << i << " is made of iron " << std::endl;
  //  if(!isIronVolume[i]) std::cout << "Volume no." << i << " is made of air " << std::endl;
  //}
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
  const NavVolume* currentVolume = 0; // explicit initialization to get rid of compiler warning

  int count = 0;
  int maxCount = 100;
  while (!destinationCrossed( startingState, TempState, targetPlane)) {

     LogDebug("NavPropagator") <<  "NavPropagator:: at beginning of while loop at iteration " << count ;

    startingState = TempState;
    bool isReflected = false;

    if (exitState.volume() != 0) { // next volume connected
      currentVolume = exitState.volume();
    }
    else {
      // 1 mm shifted peek is necessary only because geometry is not yet glued and exists only for z<0
      GlobalPoint gp = startingState.globalPosition() + 0.1*startingState.globalMomentum().unit();

      if (gp.z()>0) { // Reflect in Z (as long as MagVolumes exist only for z<0
	GlobalTrajectoryParameters gtp( GlobalPoint(startingState.globalPosition().x(),
						       startingState.globalPosition().y(),
						    -startingState.globalPosition().z()),
					GlobalVector(startingState.globalMomentum().x(),
						       startingState.globalMomentum().y(),
						       -startingState.globalMomentum().z()),
					startingState.globalParameters().charge(), theField );
	FreeTrajectoryState fts(gtp);
	TempState = TSOS( fts, startingState.surface());
	isReflected = true;
      }
      currentVolume = findVolume( TempState );
      if (currentVolume == 0) {
	 std::cout <<  "NavPropagator: findVolume failed to find volume containing point "
	     << startingState.globalPosition() ;
	return std::pair<TrajectoryStateOnSurface,double>(noNextVolume( startingState), 0);
      }
    }

    PropagatorType currentPropagator( *currentVolume);

    LogDebug("NavPropagator") <<  "NavPropagator: calling crossToNextVolume" ;
    VolumeCrossReturnType exitStateNM = currentVolume->crossToNextVolume( TempState, currentPropagator);
    LogDebug("NavPropagator") <<  "NavPropagator: crossToNextVolume returned" ;
    LogDebug("NavPropagator") <<  "Volume pointer: " << exitState.volume() << " and new ";
    LogDebug("NavPropagator") <<  exitState.tsos() ;
    LogDebug("NavPropagator") <<  "So that was a path length " << exitState.path() ;
    
    
    if (exitStateNM.tsos().isValid()) { 
      // try to add material effects !!
      //FIXME: smarter way of treating material effects is needed! Now only Iron/Air... 
      VolumeMediumProperties thisMedium = currentVolume->isIron()? theIronMedium:theAirMedium;
      VolumeMaterialEffectsEstimate msEstimate(theMSEstimator.estimate(exitStateNM.tsos(),
								       exitStateNM.path(),
								       thisMedium));
      VolumeMaterialEffectsEstimate elEstimate(theELEstimator.estimate(exitStateNM.tsos(),
								       exitStateNM.path(),
								       thisMedium));
      std::vector<const VolumeMaterialEffectsEstimate*> matEstimates;
      matEstimates.push_back(&msEstimate);
      matEstimates.push_back(&elEstimate);
      exitState = VolumeCrossReturnType(exitStateNM.volume(),
					theMaterialUpdator.updateState(exitStateNM.tsos(),alongMomentum,matEstimates),
					exitStateNM.path());
    } else { 
      exitState = exitStateNM; 
    }
    
    
    if ( !exitState.tsos().isValid()) {
      // return propagateInVolume( currentVolume, startingState, targetPlane);
       std::cout <<  "NavPropagator: failed to crossToNextVolume in volume at pos "
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


    //      std::cout << "Just moved " << exitState.path() << " cm through " << ( currentVolume->isIron()? "IRON":"AIR" ); 
    // std::cout << " at radius: " << TempState.globalPosition().perp();
    //  std::cout << " and lost " << exitStateNM.tsos().globalMomentum().mag()-exitState.tsos().globalMomentum().mag() << " GeV of Energy !!!! New energy: " << exitState.tsos().globalMomentum().mag() << std::endl;
    // reflect back to normal universe if necessary:
    if (isReflected) { // reflect back... nobody should know we secretely z-reflected the tsos 
      

      GlobalTrajectoryParameters gtp( GlobalPoint(TempState.globalPosition().x(),
						     TempState.globalPosition().y(),
						  -TempState.globalPosition().z()),
				      GlobalVector(TempState.globalMomentum().x(),
						   TempState.globalMomentum().y(),
						   -TempState.globalMomentum().z()),
				      TempState.globalParameters().charge(), theField );

      FreeTrajectoryState fts(gtp);
      TSOS ReflectedState( fts, TempState.surface());
      TempState = ReflectedState;
      isReflected = false;
    }

    ++count;
    if (count > maxCount) {
       LogDebug("NavPropagator") <<  "Ohoh, NavPropagator in infinite loop, count = " << count;
      return TsosWP();
    }

  }

  

  // Arriving here only if destinationCrossed or if crossToNextVolume returned invalid state,
  // in which case we should check if the targetPlane is not crossed in the current volume

  return propagateInVolume( currentVolume, startingState, targetPlane);  

}


const NavVolume* NavPropagator::findVolume( const TrajectoryStateOnSurface& inputState) const
{
   LogDebug("NavPropagator") <<  "NavPropagator: calling theField->findVolume ";

  GlobalPoint gp = inputState.globalPosition() + 0.1*inputState.globalMomentum().unit();
  // Next protection only needed when 1 mm linear move crosses the z==0 plane:
  GlobalPoint gpSym(gp.x(), gp.y(), (gp.z()<0? gp.z() : -gp.z()));

  const MagVolume* magVolume = theField->findVolume( gpSym);

   LogDebug("NavPropagator") <<  "NavPropagator: got MagVolume* " << magVolume 
       << " when searching with pos " << gpSym ;


   if (!magVolume) {
     cout << "Got invalid volume pointer " << magVolume << " at position " << gpSym;
     return 0;
   } 
  return navVolume(magVolume);
}


const NavVolume* NavPropagator::navVolume( const MagVolume* magVolume)  const
{
  NavVolume* result;
  MagVolumeMap::iterator i = theNavVolumeMap.find( magVolume);
  if (i == theNavVolumeMap.end()) {
    // Create a NavVolume from a MagVolume if the NavVolume doesn't exist yet
    // FIXME: hardcoded iron/air classification should be made into something more general
    // will break with new magnetic field volume .xml file MM 28/6/07
    const MagVolume6Faces* pVol6 = dynamic_cast<const MagVolume6Faces*> ( magVolume );
    int n=0;
    bool isIron=false;
    if(pVol6) { 
      std::stringstream ss(pVol6->name.substr(5));
      ss >> n;
      //      std::cout << " Found volume with number " << n << std::endl;
    } else {
      std::cout << "Error (NavVolume6Faces) failed to get MagVolume6Faces pointer" << std::endl;
    }
    if(n<1 || n>271) {
      std::cout << "Error (NavVolume6Faces) unexpected Volume number!" << std::endl;
    } else {
      isIron = isIronVolume[n];
    }

    result= new NavVolume6Faces( *magVolume, isIron);
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

  // OK, fix is rather ugly for now: need to reflect z>0 states to z<0 MM 25/6/07
  bool isReflected = false;
  TSOS okState = startingState;
  PlaneBuilder::ReturnType ReflectedPlane;
  Plane const * okPlane = &targetPlane;
  
  if (startingState.globalPosition().z()>0 || 
      (fabs(startingState.globalPosition().z())<1.e-4 && startingState.globalMomentum().z()>1.e-4)) {
    GlobalTrajectoryParameters gtp( GlobalPoint(startingState.globalPosition().x(),
						startingState.globalPosition().y(),
						-startingState.globalPosition().z()),
				    GlobalVector(startingState.globalMomentum().x(),
						 startingState.globalMomentum().y(),
						 -startingState.globalMomentum().z()),
				    startingState.globalParameters().charge(), prop.magneticField() );
    
    FreeTrajectoryState fts(gtp);
    TSOS ReflectedState( fts, startingState.surface());
    okState = ReflectedState;
      
    GlobalVector TempVec = targetPlane.normalVector();
    GlobalVector ReflectedVector = GlobalVector ( TempVec.x(), TempVec.y(), -TempVec.z());
    GlobalVector zAxis = ReflectedVector.unit();
    GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0);
    GlobalVector xAxis = yAxis.cross( zAxis);
    Surface::RotationType rot = Surface::RotationType( xAxis, yAxis, zAxis);

    GlobalPoint gp = targetPlane.position();
    GlobalPoint gpSym(gp.x(), gp.y(), -gp.z());
    PlaneBuilder pb;
    ReflectedPlane = pb.plane( gpSym, rot);
    
    okPlane =  ReflectedPlane.get();
    isReflected = true;    
  }
  
  // Done Reflecting


  TsosWP res = prop.propagateWithPath( okState, *okPlane);

  // Now reflect back the result if necessary...
  if (isReflected && res.first.isValid()) { // reflect back... nobody should know we secretely z-reflected the tsos 
   

    TSOS TempState = res.first; 
    GlobalTrajectoryParameters gtp( GlobalPoint(TempState.globalPosition().x(),
						TempState.globalPosition().y(),
						-TempState.globalPosition().z()),
				    GlobalVector(TempState.globalMomentum().x(),
						 TempState.globalMomentum().y(),
						 -TempState.globalMomentum().z()),
				    TempState.globalParameters().charge(), prop.magneticField() );
    
    FreeTrajectoryState fts(gtp);
    TSOS ReflectedState( fts, TempState.surface());
    res.first = ReflectedState;
    isReflected = false;
  }
  // Done Reflecting back !

  if (res.first.isValid()) {

    GlobalPoint gp = res.first.globalPosition();
    GlobalPoint gpSym(gp.x(), gp.y(), (gp.z()<0? gp.z() : -gp.z()));
 
    if (currentVolume->inside( gpSym )) {

      //FIXME: smarter way of treating material effects is needed! Now only Iron/Air... 
      VolumeMediumProperties thisMedium = currentVolume->isIron()? theIronMedium:theAirMedium;

      //
      // try to add material effects
      //
      VolumeMaterialEffectsEstimate msEstimate(theMSEstimator.estimate(res.first,
								       res.second,
								       thisMedium));
      VolumeMaterialEffectsEstimate elEstimate(theELEstimator.estimate(res.first,
								       res.second,
								       thisMedium));
      std::vector<const VolumeMaterialEffectsEstimate*> matEstimates;
      matEstimates.push_back(&msEstimate);
      matEstimates.push_back(&elEstimate);
      TSOS newState = theMaterialUpdator.updateState(res.first,alongMomentum,matEstimates);
      return TsosWP(newState,res.second);
    }
    // sometimes fails when propagation on plane and surface are less than 0.1 mm apart... 
  } 

  //  return TsosWP( TrajectoryStateOnSurface(), 0); // Gives 'double free' errors... return res even when Outside volume... 
  return res;

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
