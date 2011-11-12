
//Geant4e
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"
#include "TrackPropagation/Geant4e/interface/ConvertFromToCLHEP.h"
#include "TrackPropagation/Geant4e/interface/Geant4eSteppingAction.h"

//CMSSW
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Geant4
#include "G4ErrorFreeTrajState.hh"
#include "G4ErrorPlaneSurfaceTarget.hh"
#include "G4ErrorCylSurfaceTarget.hh"
#include "G4ErrorPropagatorData.hh"
#include "G4EventManager.hh"
#include "G4SteppingControl.hh"

//CLHEP
#include "CLHEP/Units/GlobalSystemOfUnits.h"


/** Constructor. 
 */
Geant4ePropagator::Geant4ePropagator(const MagneticField* field,
				     const char* particleName,
				     PropagationDirection dir):
  Propagator(dir),
  theField(field),
  theParticleName(particleName),
  theG4eManager(G4ErrorPropagatorManager::GetErrorPropagatorManager()),
  theSteppingAction(0) {

  G4ErrorPropagatorData::SetVerbose(0);
}

/** Destructor. 
 */
Geant4ePropagator::~Geant4ePropagator() {
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

  if(theG4eManager->PrintG4ErrorState() == "G4ErrorState_PreInit")
    theG4eManager->InitGeant4e();

  if (!theSteppingAction) {
    theSteppingAction = new Geant4eSteppingAction;
    theG4eManager->SetUserAction(theSteppingAction);
  }

  ///////////////////////////////
  // Construct the target surface
  //

  //* Get position and normal (orientation) of the destination plane
  GlobalPoint posPlane = pDest.toGlobal(LocalPoint(0,0,0));
  GlobalVector normalPlane = pDest.toGlobal(LocalVector(0,0,1.)); 
  normalPlane = normalPlane.unit();

  //* Transform this into HepGeom::Point3D<double>  and HepGeom::Normal3D<double>  that define a plane for
  //  Geant4e.
  //  CMS uses cm and GeV while Geant4 uses mm and MeV
  HepGeom::Point3D<double>   surfPos  = 
    TrackPropagation::globalPointToHepPoint3D(posPlane);
  HepGeom::Normal3D<double>  surfNorm = 
    TrackPropagation::globalVectorToHepNormal3D(normalPlane);

  //DEBUG
  LogDebug("Geant4e") << "G4e -  Destination CMS plane position:" << posPlane << "cm\n"
		      << "G4e -                  (Ro, eta, phi): (" 
		      << posPlane.perp() << " cm, " 
		      << posPlane.eta() << ", " 
		      << posPlane.phi().degrees() << " deg)\n"
		      << "G4e -  Destination G4  plane position: " << surfPos
		      << " mm, Ro = " << surfPos.perp() << " mm";
  LogDebug("Geant4e") << "G4e -  Destination CMS plane normal  : " 
		      << normalPlane << "\n"
		      << "G4e -  Destination G4  plane normal  : " 
		      << normalPlane;
  LogDebug("Geant4e") << "G4e -  Distance from plane position to plane: " 
		      << pDest.localZ(posPlane) << " cm";
  //DEBUG

  //* Set the target surface
  G4ErrorSurfaceTarget* g4eTarget = new G4ErrorPlaneSurfaceTarget(surfNorm,
								  surfPos);

  //g4eTarget->Dump("G4e - ");
  //
  ///////////////////////////////

  ///////////////////////////////
  // Find initial point
  //

  // * Get the starting point and direction and convert them to CLHEP::Hep3Vector 
  //   for G4. CMS uses cm and GeV while Geant4 uses mm and MeV
  GlobalPoint  cmsInitPos = ftsStart.position();
  GlobalVector cmsInitMom = ftsStart.momentum();

  CLHEP::Hep3Vector g4InitPos = 
    TrackPropagation::globalPointToHep3Vector(cmsInitPos);
  CLHEP::Hep3Vector g4InitMom = 
    TrackPropagation::globalVectorToHep3Vector(cmsInitMom*GeV);

  //DEBUG
  LogDebug("Geant4e") << "G4e -  Initial CMS point position:" << cmsInitPos 
		      << "cm\n"
		      << "G4e -              (Ro, eta, phi): (" 
		      << cmsInitPos.perp() << " cm, " 
		      << cmsInitPos.eta() << ", " 
		      << cmsInitPos.phi().degrees() << " deg)\n"
		      << "G4e -  Initial G4  point position: " << g4InitPos 
		      << " mm, Ro = " << g4InitPos.perp() << " mm";
  LogDebug("Geant4e") << "G4e -  Initial CMS momentum      :" << cmsInitMom 
		      << "GeV\n"
		      << "G4e -  Initial G4  momentum      : " << g4InitMom 
		      << " MeV";
  LogDebug("Geant4e") << "G4e -  Distance from initial point to plane: " 
		      << pDest.localZ(cmsInitPos) << " cm";
  //DEBUG

  //
  //////////////////////////////

  //////////////////////////////
  // Set particle name
  //
  int charge = ftsStart.charge();
  std::string particleName  = theParticleName;

  if (charge > 0) {
      particleName += "+";
  } else {
      particleName += "-";
  }

  LogDebug("Geant4e") << "G4e -  Particle name: " << particleName;

  //
  ///////////////////////////////

  ///////////////////////////////
  //Set the error and trajectories, and finally propagate
  //
  G4ErrorTrajErr g4error( 5, 1 );
  if(ftsStart.hasError()) {
    const CurvilinearTrajectoryError initErr = ftsStart.curvilinearError();
    g4error = TrackPropagation::algebraicSymMatrix55ToG4ErrorTrajErr( initErr , charge); //The error matrix
  }
  LogDebug("Geant4e") << "G4e -  Error matrix: " << g4error;

  G4ErrorFreeTrajState* g4eTrajState = 
    new G4ErrorFreeTrajState(particleName, g4InitPos, g4InitMom, g4error);
  LogDebug("Geant4e") << "G4e -  Traj. State: " << (*g4eTrajState);

  //Set the mode of propagation according to the propagation direction
  G4ErrorMode mode = G4ErrorMode_PropForwards;

  if (propagationDirection() == oppositeToMomentum) {
    mode = G4ErrorMode_PropBackwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'backwards\'";
  } else if(propagationDirection() == alongMomentum) {
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'forwards\'";
  } else {   //Mode must be anyDirection then - need to figure out for Geant which it is
    std::cout << "Determining actual direction";
    if(pDest.localZ(cmsInitPos)*pDest.localZ(cmsInitMom) < 0) {
      LogDebug("Geant4e") << "G4e -  Propagator mode is \'forwards\'";
      std::cout << ", got forwards" << std::endl;
    } else {
      mode = G4ErrorMode_PropBackwards;
      LogDebug("Geant4e") << "G4e -  Propagator mode is \'backwards\'";
      std::cout << ", got backwards" << std::endl;
    }
  }

  //
  //////////////////////////////

  //////////////////////////////
  // Propagate

  int ierr;
  if(mode == G4ErrorMode_PropBackwards) {
    //To make geant transport the particle correctly need to give it the opposite momentum
    //because geant flips the B field bending and adds energy instead of subtracting it
    //but still wants the momentum "backwards"
    g4eTrajState->SetMomentum( -g4eTrajState->GetMomentum());
    ierr = theG4eManager->Propagate( g4eTrajState, g4eTarget, mode);
    g4eTrajState->SetMomentum( -g4eTrajState->GetMomentum());
  } else {
    ierr = theG4eManager->Propagate( g4eTrajState, g4eTarget, mode);
  }
  LogDebug("Geant4e") << "G4e -  Return error from propagation: " << ierr;

  if(ierr!=0) {
    LogDebug("Geant4e") << "G4e - Error is not 0, returning invalid trajectory";
    return TrajectoryStateOnSurface();
  }

  //
  //////////////////////////////

  //////////////////////////////
  // Retrieve the state in the end from Geant4e, convert them to CMS vectors
  // and points, and build global trajectory parameters.
  // CMS uses cm and GeV while Geant4 uses mm and MeV
  //
  HepGeom::Point3D<double>  posEnd = g4eTrajState->GetPosition();
  HepGeom::Vector3D<double>  momEnd = g4eTrajState->GetMomentum();

  GlobalPoint  posEndGV = TrackPropagation::hepPoint3DToGlobalPoint(posEnd);
  GlobalVector momEndGV = TrackPropagation::hep3VectorToGlobalVector(momEnd)/GeV;

  //DEBUG
  LogDebug("Geant4e") << "G4e -  Final CMS point position:" << posEndGV 
		      << "cm\n"
		      << "G4e -            (Ro, eta, phi): (" 
		      << posEndGV.perp() << " cm, " 
		      << posEndGV.eta() << ", " 
		      << posEndGV.phi().degrees() << " deg)\n"
		      << "G4e -  Final G4  point position: " << posEnd 
		      << " mm,\tRo =" << posEnd.perp()  << " mm";
  LogDebug("Geant4e") << "G4e -  Final CMS momentum      :" << momEndGV
		      << "GeV\n"
		      << "G4e -  Final G4  momentum      : " << momEnd 
		      << " MeV";
  LogDebug("Geant4e") << "G4e -  Distance from final point to plane: " 
		      << pDest.localZ(posEndGV) << " cm";
  //DEBUG

  GlobalTrajectoryParameters tParsDest(posEndGV, momEndGV, charge, theField);


  // Get the error covariance matrix from Geant4e. It comes in curvilinear
  // coordinates so use the appropiate CMS class  
  G4ErrorTrajErr g4errorEnd = g4eTrajState->GetError();
  CurvilinearTrajectoryError 
    curvError(TrackPropagation::g4ErrorTrajErrToAlgebraicSymMatrix55(g4errorEnd, charge));
  LogDebug("Geant4e") << "G4e -  Error matrix after propagation: " << g4errorEnd;

  ////////////////////////////////////////////////////////////////////////
  // We set the SurfaceSide to atCenterOfSurface.                       //
  ////////////////////////////////////////////////////////////////////////
  LogDebug("Geant4e") << "G4e -  SurfaceSide is always atCenterOfSurface after propagation";
  SurfaceSideDefinition::SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface;
  //
  ////////////////////////////////////////////////////////

  return TrajectoryStateOnSurface(tParsDest, curvError, pDest, side);
}

//Require method with input TrajectoryStateOnSurface to be used in track fitting
//Don't need extra info about starting surface; use regular propagation method
TrajectoryStateOnSurface
Geant4ePropagator::propagate (const TrajectoryStateOnSurface& tsos, const Plane& plane) const {
  const FreeTrajectoryState ftsStart = *tsos.freeState();
  return propagate(ftsStart,plane);
}


/** Propagate from a free state (e.g. position and momentum in 
 *  in global cartesian coordinates) to a cylinder.
 */
TrajectoryStateOnSurface 
Geant4ePropagator::propagate (const FreeTrajectoryState& ftsStart, 
			      const Cylinder& cDest) const {

  if(theG4eManager->PrintG4ErrorState() == "G4ErrorState_PreInit")
    theG4eManager->InitGeant4e();
  if (!theSteppingAction) {
    theSteppingAction = new Geant4eSteppingAction;
    theG4eManager->SetUserAction(theSteppingAction);
  }

  //Get Cylinder parameters.
  //CMS uses cm and GeV while Geant4 uses mm and MeV.
  // - Radius
  G4float radCyl = cDest.radius()*cm;
  // - Position: PositionType & GlobalPoint are Basic3DPoint<float,GlobalTag>
  G4ThreeVector posCyl = 
    TrackPropagation::globalPointToHep3Vector(cDest.position());
  // - Rotation: Type in CMSSW is RotationType == TkRotation<T>, T=float
  G4RotationMatrix rotCyl = 
    TrackPropagation::tkRotationFToHepRotation(cDest.rotation());

  //DEBUG
  TkRotation<float>  rotation = cDest.rotation();
  LogDebug("Geant4e") << "G4e -  TkRotation" << rotation;
  LogDebug("Geant4e") << "G4e -  G4Rotation" << rotCyl << "mm";


  //Set the target surface
  G4ErrorSurfaceTarget* g4eTarget = new G4ErrorCylSurfaceTarget(radCyl,	posCyl,
								rotCyl);

  //DEBUG
  LogDebug("Geant4e") << "G4e -  Destination CMS cylinder position:" << cDest.position() << "cm\n"
		      << "G4e -  Destination CMS cylinder radius:" << cDest.radius() << "cm\n"
		      << "G4e -  Destination CMS cylinder rotation:" << cDest.rotation() << "\n";
  LogDebug("Geant4e") << "G4e -  Destination G4  cylinder position: " << posCyl << "mm\n"
	              << "G4e -  Destination G4  cylinder radius:" << radCyl << "mm\n"
		      << "G4e -  Destination G4  cylinder rotation:" << rotCyl << "\n";


  //Get the starting point and direction and convert them to CLHEP::Hep3Vector for G4
  //CMS uses cm and GeV while Geant4 uses mm and MeV
  GlobalPoint  cmsInitPos = ftsStart.position();
  GlobalVector cmsInitMom = ftsStart.momentum();

  CLHEP::Hep3Vector g4InitMom = 
    TrackPropagation::globalVectorToHep3Vector(cmsInitMom*GeV);
  CLHEP::Hep3Vector g4InitPos = 
    TrackPropagation::globalPointToHep3Vector(cmsInitPos);

  //DEBUG
  LogDebug("Geant4e") << "G4e -  Initial CMS point position:" << cmsInitPos 
		      << "cm\n"
		      << "G4e -              (Ro, eta, phi): (" 
		      << cmsInitPos.perp() << " cm, " 
		      << cmsInitPos.eta() << ", " 
		      << cmsInitPos.phi().degrees() << " deg)\n"
		      << "G4e -  Initial G4  point position: " << g4InitPos 
		      << " mm, Ro = " << g4InitPos.perp() << " mm";
  LogDebug("Geant4e") << "G4e -  Initial CMS momentum      :" << cmsInitMom 
		      << "GeV\n"
		      << "G4e -  Initial G4  momentum      : " << g4InitMom 
		      << " MeV";

  //Set particle name
  int charge = ftsStart.charge();
  std::string particleName  = theParticleName;
  if (charge > 0)
    particleName += "+";
  else
    particleName += "-";
  LogDebug("Geant4e") << "G4e -  Particle name: " << particleName;

  //Set the error and trajectories, and finally propagate
  G4ErrorTrajErr g4error( 5, 1 );
  if(ftsStart.hasError()) {
    const CurvilinearTrajectoryError initErr = ftsStart.curvilinearError();
    g4error = TrackPropagation::algebraicSymMatrix55ToG4ErrorTrajErr( initErr , charge); //The error matrix
  }
  LogDebug("Geant4e") << "G4e -  Error matrix: " << g4error;

  G4ErrorFreeTrajState* g4eTrajState = 
    new G4ErrorFreeTrajState(particleName, g4InitPos, g4InitMom, g4error);
  LogDebug("Geant4e") << "G4e -  Traj. State: " << (*g4eTrajState);

  //Set the mode of propagation according to the propagation direction
  G4ErrorMode mode = G4ErrorMode_PropForwards;

  if (propagationDirection() == oppositeToMomentum) {
    mode = G4ErrorMode_PropBackwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'backwards\'";
  } else if(propagationDirection() == alongMomentum) {
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'forwards\'";
  } else {
    //------------------------------------
    //For cylinder assume outside is backwards, inside is along
    //General use for particles from collisions
    LocalPoint lpos = cDest.toLocal(cmsInitPos);
    Surface::Side theSide = cDest.side(lpos,0);
    if(theSide==SurfaceOrientation::positiveSide){  //outside cylinder
      mode = G4ErrorMode_PropBackwards;
      LogDebug("Geant4e") << "G4e -  Propagator mode is \'backwards\'";
    } else { //inside cylinder
      LogDebug("Geant4e") << "G4e -  Propagator mode is \'forwards\'";
    }

  }

  //////////////////////////////
  // Propagate

  int ierr;
  if(mode == G4ErrorMode_PropBackwards) {
    //To make geant transport the particle correctly need to give it the opposite momentum
    //because geant flips the B field bending and adds energy instead of subtracting it
    //but still wants the momentum "backwards"
    g4eTrajState->SetMomentum( -g4eTrajState->GetMomentum());
    ierr = theG4eManager->Propagate( g4eTrajState, g4eTarget, mode);
    g4eTrajState->SetMomentum( -g4eTrajState->GetMomentum());
  } else {
    ierr = theG4eManager->Propagate( g4eTrajState, g4eTarget, mode);
  }
  LogDebug("Geant4e") << "G4e -  Return error from propagation: " << ierr;

  if(ierr!=0) {
    LogDebug("Geant4e") << "G4e - Error is not 0, returning invalid trajectory";
    return TrajectoryStateOnSurface();
  }

  // Retrieve the state in the end from Geant4e, converte them to CMS vectors
  // and points, and build global trajectory parameters
  // CMS uses cm and GeV while Geant4 uses mm and MeV
  HepGeom::Point3D<double>  posEnd = g4eTrajState->GetPosition();
  HepGeom::Vector3D<double>  momEnd = g4eTrajState->GetMomentum();

  GlobalPoint  posEndGV = TrackPropagation::hepPoint3DToGlobalPoint(posEnd);
  GlobalVector momEndGV = TrackPropagation::hep3VectorToGlobalVector(momEnd)/GeV;


  //DEBUG
  LogDebug("Geant4e") << "G4e -  Final CMS point position:" << posEndGV 
		      << "cm\n"
		      << "G4e -            (Ro, eta, phi): (" 
		      << posEndGV.perp() << " cm, " 
		      << posEndGV.eta() << ", " 
		      << posEndGV.phi().degrees() << " deg)\n"
		      << "G4e -  Final G4  point position: " << posEnd 
		      << " mm,\tRo =" << posEnd.perp()  << " mm";
  LogDebug("Geant4e") << "G4e -  Final CMS momentum      :" << momEndGV
		      << "GeV\n"
		      << "G4e -  Final G4  momentum      : " << momEnd 
		      << " MeV";

  GlobalTrajectoryParameters tParsDest(posEndGV, momEndGV, charge, theField);


  // Get the error covariance matrix from Geant4e. It comes in curvilinear
  // coordinates so use the appropiate CMS class  
  G4ErrorTrajErr g4errorEnd = g4eTrajState->GetError();
  CurvilinearTrajectoryError 
    curvError(TrackPropagation::g4ErrorTrajErrToAlgebraicSymMatrix55(g4errorEnd, charge));
  LogDebug("Geant4e") << "G4e -  Error matrix after propagation: " << g4errorEnd;

  ////////////////////////////////////////////////////////////////////////
  // We set the SurfaceSide to atCenterOfSurface.                       //
  ////////////////////////////////////////////////////////////////////////

  SurfaceSideDefinition::SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface;

  return TrajectoryStateOnSurface(tParsDest, curvError, cDest, side);
}


//Require method with input TrajectoryStateOnSurface to be used in track fitting
//Don't need extra info about starting surface; use regular propagation method
TrajectoryStateOnSurface
Geant4ePropagator::propagate (const TrajectoryStateOnSurface& tsos, const Cylinder& cyl) const {
  const FreeTrajectoryState ftsStart = *tsos.freeState();
  return propagate(ftsStart,cyl);
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

std::pair< TrajectoryStateOnSurface, double> 
Geant4ePropagator::propagateWithPath (const TrajectoryStateOnSurface& tsosStart, 
				      const Plane& pDest) const {

  theSteppingAction->reset();

  //Finally build the pair<...> that needs to be returned where the second
  //parameter is the exact path length. Currently calculated with a stepping
  //action that adds up the length of every step
  return TsosPP(propagate(tsosStart,pDest), theSteppingAction->trackLength());
}

std::pair< TrajectoryStateOnSurface, double> 
Geant4ePropagator::propagateWithPath (const TrajectoryStateOnSurface& tsosStart,
				      const Cylinder& cDest) const {
  theSteppingAction->reset();

  //Finally build the pair<...> that needs to be returned where the second
  //parameter is the exact path length. Currently calculated with a stepping
  //action that adds up the length of every step
  return TsosPP(propagate(tsosStart,cDest), theSteppingAction->trackLength());
}
