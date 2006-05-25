//Geant4e
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"
#include "TrackPropagation/Geant4e/interface/ConvertFromToCLHEP.h"
#include "TrackPropagation/Geant4e/interface/Geant4eSteppingAction.h"

//CMSSW
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Geant4
#include "G4eTrajStateFree.hh"
#include "G4eTargetPlaneSurface.hh"
#include "G4eTargetCylindricalSurface.hh"


/** Constructor. 
 */
Geant4ePropagator::Geant4ePropagator(const MagneticField* field,
				     const char* particleName,
				     PropagationDirection dir):
  Propagator(dir),
  theField(field),
  theParticleName(particleName),
  theG4eManager(G4eManager::GetG4eManager()),
  theSteppingAction(new Geant4eSteppingAction) {

  theG4eManager->SetUserAction(theSteppingAction);
}

/** Destructor. 
 */
Geant4ePropagator::~Geant4ePropagator() {
  delete theSteppingAction;
}

//
////////////////////////////////////////////////////////////////////////////
//

/** Propagate from a free state (e.g. position and momentum in 
 *  in global cartesian coordinates) to a plane.
 */

TrajectoryStateOnSurface 
Geant4ePropagator::propagate (const FreeTrajectoryState& ftsStart, 
			      const Plane& pDest) const {

  //Get origin point and direction of the destination plane
  GlobalPoint posPlane = pDest.toGlobal(LocalPoint(0,0,0));
  GlobalVector normalPlane = pDest.toGlobal(LocalVector(0,0,1.)); 
  normalPlane = normalPlane.unit();

  //Transform this into HepPoint3D and HepNormal3D that define a plane for
  //Geant4e
  HepPoint3D  surfPos  = 
    TrackPropagation::globalPointToHepPoint3D(posPlane);
  HepNormal3D surfNorm = 
    TrackPropagation::globalVectorToHepNormal3D(normalPlane);

  //Set the target surface
  G4eTarget* g4eTarget = new G4eTargetPlaneSurface(surfNorm, surfPos);
  theG4eManager->SetTarget(g4eTarget);

  //Get the starting point and direction and convert them to Hep3Vector for G4
  GlobalVector p3GV = ftsStart.momentum();
  GlobalPoint  r3GP = ftsStart.position();

  Hep3Vector pv3 = TrackPropagation::globalVectorToHep3Vector(p3GV);
  Hep3Vector xv3 = TrackPropagation::globalPointToHep3Vector(r3GP);

  //Set particle name
  int charge = ftsStart.charge();
  std::string particleName  = theParticleName;
  if (charge > 0)
    particleName += "+";
  else
    particleName += "-";

  //Set the error and trajectories, and finally propagate
  G4eTrajError error( 5, 0 ); //The error matrix
  G4eTrajStateFree* g4eTrajState = 
    new G4eTrajStateFree(particleName, xv3, pv3, error);

  //Set the mode of propagation according to the propagation direction
  G4eMode mode = G4eMode_PropForwards;
  if (propagationDirection() == oppositeToMomentum)
    mode = G4eMode_PropBackwards;
    

  //int ierr =
  theG4eManager->Propagate( g4eTrajState, g4eTarget, mode);


  // Retrieve the state in the end from Geant4e, converte them to CMS vectors
  // and points, and build global trajectory parameters
  HepPoint3D posEnd = g4eTrajState->GetPosition();
  HepVector3D momEnd = g4eTrajState->GetMomentum();

  GlobalPoint  posEndGV = TrackPropagation::hepPoint3DToGlobalPoint(posEnd);
  GlobalVector momEndGV = TrackPropagation::hep3VectorToGlobalVector(momEnd);

  GlobalTrajectoryParameters tParsDest(posEndGV, momEndGV, charge, theField);


  // Get the error covariance matrix from Geant4e. It comes in curvilinear
  // coordinates so use the appropiate CMS class  
  G4eTrajError errorEnd = g4eTrajState->GetError();
  CurvilinearTrajectoryError curvError(errorEnd);


  ////////////////////////////////////////////////////////////////////////
  // WARNING: Since this propagator is not supposed to be used in the 
  // tracker where special treatment need to be used when arriving to
  // a surface, we set the SurfaceSide to atCenterOfSurface.
  ////////////////////////////////////////////////////////////////////////
  LogDebug("Geant4e") << "SurfaceSide is always atCenterOfSurface after propagation";
  SurfaceSide side = atCenterOfSurface;

  return TrajectoryStateOnSurface(tParsDest, curvError, pDest, side);
}


/** Propagate from a free state (e.g. position and momentum in 
 *  in global cartesian coordinates) to a cylinder.
 */
TrajectoryStateOnSurface 
Geant4ePropagator::propagate (const FreeTrajectoryState& ftsStart, 
			      const Cylinder& cDest) const {
  //Get Cylinder parameters
  // - Radius
  G4float radCyl = cDest.radius();
  // - Position: PositionType & GlobalPoint are Basic3DPoint<float,GlobalTag>
  G4ThreeVector posCyl = 
    TrackPropagation::globalPointToHep3Vector(cDest.position());
  // - Rotation: Type in CMSSW is RotationType == TkRotation<T>, T=float
  G4RotationMatrix rotCyl = 
    TrackPropagation::tkRotationFToHepRotation(cDest.rotation());

  //DEBUG --- Remove at some point
  TkRotation<float>  rotation = cDest.rotation();
  LogDebug("Geant4e") << "TkRotation" << rotation;
  LogDebug("Geant4e") << "G4Rotation" << rotCyl;


  //Set the target surface
  G4eTarget* g4eTarget = new G4eTargetCylindricalSurface(radCyl, 
							 posCyl, rotCyl);
  theG4eManager->SetTarget(g4eTarget);

  //Get the starting point and direction and convert them to Hep3Vector for G4
  Hep3Vector pv3 = 
    TrackPropagation::globalVectorToHep3Vector(ftsStart.momentum());
  Hep3Vector xv3 = 
    TrackPropagation::globalPointToHep3Vector(ftsStart.position());

  //Set particle name
  int charge = ftsStart.charge();
  std::string particleName  = theParticleName;
  if (charge > 0)
    particleName += "+";
  else
    particleName += "-";

  //Set the error and trajectories, and finally propagate
  G4eTrajError error( 5, 0 ); //The error matrix
  G4eTrajStateFree* g4eTrajState = 
    new G4eTrajStateFree(particleName, xv3, pv3, error);

  //Set the mode of propagation according to the propagation direction
  G4eMode mode = G4eMode_PropForwards;
  if (propagationDirection() == oppositeToMomentum)
    mode = G4eMode_PropBackwards;
    

  //int ierr =
  theG4eManager->Propagate( g4eTrajState, g4eTarget, mode);


  // Retrieve the state in the end from Geant4e, converte them to CMS vectors
  // and points, and build global trajectory parameters
  HepPoint3D posEnd = g4eTrajState->GetPosition();
  HepVector3D momEnd = g4eTrajState->GetMomentum();

  GlobalPoint  posEndGV = TrackPropagation::hepPoint3DToGlobalPoint(posEnd);
  GlobalVector momEndGV = TrackPropagation::hep3VectorToGlobalVector(momEnd);

  GlobalTrajectoryParameters tParsDest(posEndGV, momEndGV, charge, theField);


  // Get the error covariance matrix from Geant4e. It comes in curvilinear
  // coordinates so use the appropiate CMS class  
  G4eTrajError errorEnd = g4eTrajState->GetError();
  CurvilinearTrajectoryError curvError(errorEnd);


  ////////////////////////////////////////////////////////////////////////
  // WARNING: Since this propagator is not supposed to be used in the 
  // tracker where special treatment need to be used when arriving to
  // a surface, we set the SurfaceSide to atCenterOfSurface.
  ////////////////////////////////////////////////////////////////////////
  SurfaceSide side = atCenterOfSurface;

  return TrajectoryStateOnSurface(tParsDest, curvError, cDest, side);
}

//
////////////////////////////////////////////////////////////////////////////
//

/** The methods propagateWithPath() are identical to the corresponding
 *  methods propagate() in what concerns the resulting 
 *  TrajectoryStateOnSurface, but they provide in addition the
 *  exact path length along the trajectory.
 */

std::pair< TrajectoryStateOnSurface, double> 
Geant4ePropagator::propagateWithPath (const FreeTrajectoryState& ftsStart, 
				      const Plane& pDest) const {

  theSteppingAction->reset();

  //Finally build the pair<...> that needs to be returned where the second
  //parameter is the exact path length. Currently calculated with a stepping
  //action that adds up the length of every step
  return TsosPP(propagate(ftsStart,pDest), theSteppingAction->trackLength());
}

std::pair< TrajectoryStateOnSurface, double> 
Geant4ePropagator::propagateWithPath (const FreeTrajectoryState& ftsStart,
				      const Cylinder& cDest) const {
  theSteppingAction->reset();

  //Finally build the pair<...> that needs to be returned where the second
  //parameter is the exact path length. Currently calculated with a stepping
  //action that adds up the length of every step
  return TsosPP(propagate(ftsStart,cDest), theSteppingAction->trackLength());
}
