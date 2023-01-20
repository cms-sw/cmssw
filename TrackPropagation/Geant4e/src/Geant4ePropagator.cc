#include <sstream>

// Geant4e
#include "TrackPropagation/Geant4e/interface/ConvertFromToCLHEP.h"
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"

// CMSSW
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"

// Geant4
#include "G4Box.hh"
#include "G4ErrorCylSurfaceTarget.hh"
#include "G4ErrorFreeTrajState.hh"
#include "G4ErrorPlaneSurfaceTarget.hh"
#include "G4ErrorPropagatorData.hh"
#include "G4ErrorRunManagerHelper.hh"
#include "G4EventManager.hh"
#include "G4Field.hh"
#include "G4FieldManager.hh"
#include "G4GeometryTolerance.hh"
#include "G4SteppingControl.hh"
#include "G4TransportationManager.hh"
#include "G4Tubs.hh"
#include "G4UImanager.hh"
#include "G4ErrorPropagationNavigator.hh"
#include "G4RunManagerKernel.hh"

// CLHEP
#include "CLHEP/Units/GlobalSystemOfUnits.h"

/** Constructor.
 */
Geant4ePropagator::Geant4ePropagator(const MagneticField *field,
                                     std::string particleName,
                                     PropagationDirection dir,
                                     double plimit)
    : Propagator(dir),
      theField(field),
      theParticleName(particleName),
      theG4eManager(G4ErrorPropagatorManager::GetErrorPropagatorManager()),
      theG4eData(G4ErrorPropagatorData::GetErrorPropagatorData()),
      plimit_(plimit) {
  LogDebug("Geant4e") << "Geant4e Propagator initialized";

  // has to be called here, doing it later will not load the G4 physics list
  // properly when using the G4 ES Producer. Reason: unclear
  ensureGeant4eIsInitilized(true);
}

/** Destructor.
 */
Geant4ePropagator::~Geant4ePropagator() {
  LogDebug("Geant4e") << "Geant4ePropagator::~Geant4ePropagator()" << std::endl;

  // don't close the g4 Geometry here, because the propagator might have been
  // cloned
  // but there is only one, globally-shared Geometry
}

//
////////////////////////////////////////////////////////////////////////////
//

/** Propagate from a free state (e.g. position and momentum in
 *  in global cartesian coordinates) to a plane.
 */

void Geant4ePropagator::ensureGeant4eIsInitilized(bool forceInit) const {
  LogDebug("Geant4e") << "ensureGeant4eIsInitilized called" << std::endl;
  if (forceInit) {
    LogDebug("Geant4e") << "Initializing G4 propagator" << std::endl;

    //G4UImanager::GetUIpointer()->ApplyCommand("/exerror/setField -10. kilogauss");

    auto man = G4RunManagerKernel::GetRunManagerKernel();
    man->SetVerboseLevel(0);
    theG4eManager->InitGeant4e();

    const G4Field *field = G4TransportationManager::GetTransportationManager()->GetFieldManager()->GetDetectorField();
    if (field == nullptr) {
      edm::LogError("Geant4e") << "No G4 magnetic field defined";
    }
    LogDebug("Geant4e") << "G4 propagator initialized" << std::endl;
  }
  // define 10 mm step limit for propagator
  G4UImanager::GetUIpointer()->ApplyCommand("/geant4e/limits/stepLength 10.0 mm");
}

template <>
Geant4ePropagator::ErrorTargetPair Geant4ePropagator::transformToG4SurfaceTarget(const Plane &pDest,
                                                                                 bool moveTargetToEndOfSurface) const {
  //* Get position and normal (orientation) of the destination plane
  GlobalPoint posPlane = pDest.toGlobal(LocalPoint(0, 0, 0));
  GlobalVector normalPlane = pDest.toGlobal(LocalVector(0, 0, 1.));
  normalPlane = normalPlane.unit();

  //* Transform this into HepGeom::Point3D<double>  and
  // HepGeom::Normal3D<double>  that define a plane for
  //  Geant4e.
  //  CMS uses cm and GeV while Geant4 uses mm and MeV
  HepGeom::Point3D<double> surfPos = TrackPropagation::globalPointToHepPoint3D(posPlane);
  HepGeom::Normal3D<double> surfNorm = TrackPropagation::globalVectorToHepNormal3D(normalPlane);

  //* Set the target surface
  return ErrorTargetPair(false, std::make_shared<G4ErrorPlaneSurfaceTarget>(surfNorm, surfPos));
}

template <>
Geant4ePropagator::ErrorTargetPair Geant4ePropagator::transformToG4SurfaceTarget(const Cylinder &pDest,
                                                                                 bool moveTargetToEndOfSurface) const {
  // Get Cylinder parameters.
  // CMS uses cm and GeV while Geant4 uses mm and MeV.
  // - Radius
  G4float radCyl = pDest.radius() * cm;
  // - Position: PositionType & GlobalPoint are Basic3DPoint<float,GlobalTag>
  G4ThreeVector posCyl = TrackPropagation::globalPointToHep3Vector(pDest.position());
  // - Rotation: Type in CMSSW is RotationType == TkRotation<T>, T=float
  G4RotationMatrix rotCyl = TrackPropagation::tkRotationFToHepRotation(pDest.rotation());

  // DEBUG
  TkRotation<float> rotation = pDest.rotation();
  LogDebug("Geant4e") << "G4e -  TkRotation" << rotation;
  LogDebug("Geant4e") << "G4e -  G4Rotation" << rotCyl << "mm";

  return ErrorTargetPair(!moveTargetToEndOfSurface, std::make_shared<G4ErrorCylSurfaceTarget>(radCyl, posCyl, rotCyl));
}

template <>
std::string Geant4ePropagator::getSurfaceType(Cylinder const &c) const {
  return "Cylinder";
}

template <>
std::string Geant4ePropagator::getSurfaceType(Plane const &c) const {
  return "Plane";
}

std::string Geant4ePropagator::generateParticleName(int charge) const {
  std::string particleName = theParticleName;

  if (charge > 0) {
    particleName += "+";
  }
  if (charge < 0) {
    particleName += "-";
  }

  LogDebug("Geant4e") << "G4e -  Particle name: " << particleName;

  return particleName;
}

template <>
bool Geant4ePropagator::configureAnyPropagation(G4ErrorMode &mode,
                                                Plane const &pDest,
                                                GlobalPoint const &cmsInitPos,
                                                GlobalVector const &cmsInitMom) const {
  if (cmsInitMom.mag() < plimit_)
    return false;
  if (pDest.localZ(cmsInitPos) * pDest.localZ(cmsInitMom) < 0) {
    mode = G4ErrorMode_PropForwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'forwards\' indirect "
                           "via the Any direction"
                        << std::endl;
  } else {
    mode = G4ErrorMode_PropBackwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'backwards\' indirect "
                           "via the Any direction"
                        << std::endl;
  }

  return true;
}

template <>
bool Geant4ePropagator::configureAnyPropagation(G4ErrorMode &mode,
                                                Cylinder const &pDest,
                                                GlobalPoint const &cmsInitPos,
                                                GlobalVector const &cmsInitMom) const {
  if (cmsInitMom.mag() < plimit_)
    return false;
  //------------------------------------
  // For cylinder assume outside is backwards, inside is along
  // General use for particles from collisions
  LocalPoint lpos = pDest.toLocal(cmsInitPos);
  Surface::Side theSide = pDest.side(lpos, 0);
  if (theSide == SurfaceOrientation::positiveSide) {  // outside cylinder
    mode = G4ErrorMode_PropBackwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'backwards\' indirect "
                           "via the Any direction";
  } else {  // inside cylinder
    mode = G4ErrorMode_PropForwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'forwards\' indirect "
                           "via the Any direction";
  }

  return true;
}

template <class SurfaceType>
bool Geant4ePropagator::configurePropagation(G4ErrorMode &mode,
                                             SurfaceType const &pDest,
                                             GlobalPoint const &cmsInitPos,
                                             GlobalVector const &cmsInitMom) const {
  if (cmsInitMom.mag() < plimit_)
    return false;
  if (propagationDirection() == oppositeToMomentum) {
    mode = G4ErrorMode_PropBackwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'backwards\' " << std::endl;
  } else if (propagationDirection() == alongMomentum) {
    mode = G4ErrorMode_PropForwards;
    LogDebug("Geant4e") << "G4e -  Propagator mode is \'forwards\'" << std::endl;
  } else if (propagationDirection() == anyDirection) {
    if (configureAnyPropagation(mode, pDest, cmsInitPos, cmsInitMom) == false)
      return false;
  } else {
    edm::LogError("Geant4e") << "G4e - Unsupported propagation mode";
    return false;
  }
  return true;
}

template <class SurfaceType>
std::pair<TrajectoryStateOnSurface, double> Geant4ePropagator::propagateGeneric(const FreeTrajectoryState &ftsStart,
                                                                                const SurfaceType &pDest) const {
  ///////////////////////////////
  // Construct the target surface
  //
  //* Set the target surface

  ErrorTargetPair g4eTarget_center = transformToG4SurfaceTarget(pDest, false);

  // * Get the starting point and direction and convert them to
  // CLHEP::Hep3Vector
  //   for G4. CMS uses cm and GeV while Geant4 uses mm and MeV
  GlobalPoint cmsInitPos = ftsStart.position();
  GlobalVector cmsInitMom = ftsStart.momentum();
  bool flipped = false;
  if (propagationDirection() == oppositeToMomentum) {
    // flip the momentum vector as Geant4 will not do this
    // on it's own in a backward propagation
    cmsInitMom = -cmsInitMom;
    flipped = true;
  }

  // Set the mode of propagation according to the propagation direction
  G4ErrorMode mode = G4ErrorMode_PropForwards;
  if (!configurePropagation(mode, pDest, cmsInitPos, cmsInitMom))
    return TsosPP(TrajectoryStateOnSurface(), 0.0f);

  // re-check propagation direction chosen in case of AnyDirection
  if (mode == G4ErrorMode_PropBackwards && !flipped)
    cmsInitMom = -cmsInitMom;

  CLHEP::Hep3Vector g4InitPos = TrackPropagation::globalPointToHep3Vector(cmsInitPos);
  CLHEP::Hep3Vector g4InitMom = TrackPropagation::globalVectorToHep3Vector(cmsInitMom * GeV);

  debugReportTrackState("intitial", cmsInitPos, g4InitPos, cmsInitMom, g4InitMom, pDest);

  // Set the mode of propagation according to the propagation direction
  // G4ErrorMode mode = G4ErrorMode_PropForwards;

  // if (!configurePropagation(mode, pDest, cmsInitPos, cmsInitMom))
  //	return TsosPP(TrajectoryStateOnSurface(), 0.0f);

  ///////////////////////////////
  // Set the error and trajectories, and finally propagate
  //
  G4ErrorTrajErr g4error(5, 1);
  if (ftsStart.hasError()) {
    CurvilinearTrajectoryError initErr;
    initErr = ftsStart.curvilinearError();
    g4error = TrackPropagation::algebraicSymMatrix55ToG4ErrorTrajErr(initErr, ftsStart.charge());
    LogDebug("Geant4e") << "CMS -  Error matrix: " << std::endl << initErr.matrix();
  } else {
    LogDebug("Geant4e") << "No error matrix available" << std::endl;
    return TsosPP(TrajectoryStateOnSurface(), 0.0f);
  }

  LogDebug("Geant4e") << "G4e -  Error matrix: " << std::endl << g4error;

  // in CMSSW, the state errors are deflated when performing the backward
  // propagation
  if (mode == G4ErrorMode_PropForwards) {
    G4ErrorPropagatorData::GetErrorPropagatorData()->SetStage(G4ErrorStage_Inflation);
  } else if (mode == G4ErrorMode_PropBackwards) {
    G4ErrorPropagatorData::GetErrorPropagatorData()->SetStage(G4ErrorStage_Deflation);
  }

  G4ErrorFreeTrajState g4eTrajState(generateParticleName(ftsStart.charge()), g4InitPos, g4InitMom, g4error);
  LogDebug("Geant4e") << "G4e -  Traj. State: " << (g4eTrajState);

  //////////////////////////////
  // Propagate
  int iterations = 0;
  double finalPathLength = 0;

  HepGeom::Point3D<double> finalRecoPos;

  G4ErrorPropagatorData::GetErrorPropagatorData()->SetMode(mode);

  theG4eData->SetTarget(g4eTarget_center.second.get());
  LogDebug("Geant4e") << "Running Propagation to the RECO surface" << std::endl;

  theG4eManager->InitTrackPropagation();

  // re-initialize navigator to avoid mismatches and/or segfaults
  theG4eManager->GetErrorPropagationNavigator()->LocateGlobalPointAndSetup(
      g4InitPos, &g4InitMom, /*pRelativeSearch = */ false, /*ignoreDirection = */ false);

  bool continuePropagation = true;
  while (continuePropagation) {
    iterations++;
    LogDebug("Geant4e") << std::endl << "step count " << iterations << " step length " << finalPathLength;

    // re-initialize navigator to avoid mismatches and/or segfaults
    theG4eManager->GetErrorPropagationNavigator()->LocateGlobalPointWithinVolume(g4eTrajState.GetPosition());

    const int ierr = theG4eManager->PropagateOneStep(&g4eTrajState, mode);

    if (ierr != 0) {
      // propagation failed, return invalid track state
      return TsosPP(TrajectoryStateOnSurface(), 0.0f);
    }

    const float thisPathLength = TrackPropagation::g4doubleToCmsDouble(g4eTrajState.GetG4Track()->GetStepLength());

    LogDebug("Geant4e") << "step Length was " << thisPathLength << " cm, current global position: "
                        << TrackPropagation::hepPoint3DToGlobalPoint(g4eTrajState.GetPosition()) << std::endl;

    finalPathLength += thisPathLength;

    // if (std::fabs(finalPathLength) > 10000.0f)
    if (std::fabs(finalPathLength) > 200.0f) {
      LogDebug("Geant4e") << "ERROR: Quitting propagation: path length mega large" << std::endl;
      theG4eManager->GetPropagator()->InvokePostUserTrackingAction(g4eTrajState.GetG4Track());
      continuePropagation = false;
      LogDebug("Geant4e") << "WARNING: Quitting propagation: max path length "
                             "exceeded, returning invalid state"
                          << std::endl;

      // reached maximum path length, bail out
      return TsosPP(TrajectoryStateOnSurface(), 0.0f);
    }

    if (theG4eManager->GetPropagator()->CheckIfLastStep(g4eTrajState.GetG4Track())) {
      theG4eManager->GetPropagator()->InvokePostUserTrackingAction(g4eTrajState.GetG4Track());
      continuePropagation = false;
    }
  }

  // CMSSW Tracking convention, backward propagations have negative path length
  if (propagationDirection() == oppositeToMomentum)
    finalPathLength = -finalPathLength;

  // store the correct location for the hit on the RECO surface
  LogDebug("Geant4e") << "Position on the RECO surface" << g4eTrajState.GetPosition() << std::endl;
  finalRecoPos = g4eTrajState.GetPosition();

  theG4eManager->EventTermination();

  LogDebug("Geant4e") << "Final position of the Track :" << g4eTrajState.GetPosition() << std::endl;

  //////////////////////////////
  // Retrieve the state in the end from Geant4e, convert them to CMS vectors
  // and points, and build global trajectory parameters.
  // CMS uses cm and GeV while Geant4 uses mm and MeV
  //
  const HepGeom::Vector3D<double> momEnd = g4eTrajState.GetMomentum();

  // use the hit on the the RECO plane as the final position to be d'accor with
  // the RecHit measurements
  const GlobalPoint posEndGV = TrackPropagation::hepPoint3DToGlobalPoint(finalRecoPos);
  GlobalVector momEndGV = TrackPropagation::hep3VectorToGlobalVector(momEnd) / GeV;

  debugReportTrackState("final", posEndGV, finalRecoPos, momEndGV, momEnd, pDest);

  // Get the error covariance matrix from Geant4e. It comes in curvilinear
  // coordinates so use the appropiate CMS class
  G4ErrorTrajErr g4errorEnd = g4eTrajState.GetError();

  CurvilinearTrajectoryError curvError(
      TrackPropagation::g4ErrorTrajErrToAlgebraicSymMatrix55(g4errorEnd, ftsStart.charge()));

  if (mode == G4ErrorMode_PropBackwards) {
    GlobalTrajectoryParameters endParm(
        posEndGV, momEndGV, ftsStart.parameters().charge(), &ftsStart.parameters().magneticField());

    // flip the momentum direction because it has been flipped before running
    // G4's backwards prop
    momEndGV = -momEndGV;
  }

  LogDebug("Geant4e") << "G4e -  Error matrix after propagation: " << std::endl << g4errorEnd;

  LogDebug("Geant4e") << "CMS -  Error matrix after propagation: " << std::endl << curvError.matrix();

  GlobalTrajectoryParameters tParsDest(posEndGV, momEndGV, ftsStart.charge(), theField);

  SurfaceSideDefinition::SurfaceSide side;

  side = propagationDirection() == alongMomentum ? SurfaceSideDefinition::afterSurface
                                                 : SurfaceSideDefinition::beforeSurface;

  return TsosPP(TrajectoryStateOnSurface(tParsDest, curvError, pDest, side), finalPathLength);
}

//
////////////////////////////////////////////////////////////////////////////
//

/** The methods propagateWithPath() are identical to the corresponding
 *  methods propagate() in what concerns the resulting
 *  TrajectoryStateOnSurface, but they provide in addition the
 *  exact path length along the trajectory.
 */

std::pair<TrajectoryStateOnSurface, double> Geant4ePropagator::propagateWithPath(const FreeTrajectoryState &ftsStart,
                                                                                 const Plane &pDest) const {
  // Finally build the pair<...> that needs to be returned where the second
  // parameter is the exact path length. Currently calculated with a stepping
  // action that adds up the length of every step
  return propagateGeneric(ftsStart, pDest);
}

std::pair<TrajectoryStateOnSurface, double> Geant4ePropagator::propagateWithPath(const FreeTrajectoryState &ftsStart,
                                                                                 const Cylinder &cDest) const {
  // Finally build the pair<...> that needs to be returned where the second
  // parameter is the exact path length.
  return propagateGeneric(ftsStart, cDest);
}

std::pair<TrajectoryStateOnSurface, double> Geant4ePropagator::propagateWithPath(
    const TrajectoryStateOnSurface &tsosStart, const Plane &pDest) const {
  // Finally build the pair<...> that needs to be returned where the second
  // parameter is the exact path length.
  const FreeTrajectoryState ftsStart = *tsosStart.freeState();
  return propagateGeneric(ftsStart, pDest);
}

std::pair<TrajectoryStateOnSurface, double> Geant4ePropagator::propagateWithPath(
    const TrajectoryStateOnSurface &tsosStart, const Cylinder &cDest) const {
  const FreeTrajectoryState ftsStart = *tsosStart.freeState();
  // Finally build the pair<...> that needs to be returned where the second
  // parameter is the exact path length.
  return propagateGeneric(ftsStart, cDest);
}

void Geant4ePropagator::debugReportPlaneSetup(GlobalPoint const &posPlane,
                                              HepGeom::Point3D<double> const &surfPos,
                                              GlobalVector const &normalPlane,
                                              HepGeom::Normal3D<double> const &surfNorm,
                                              const Plane &pDest) const {
  LogDebug("Geant4e") << "G4e -  Destination CMS plane position:" << posPlane << "cm\n"
                      << "G4e -                  (Ro, eta, phi): (" << posPlane.perp() << " cm, " << posPlane.eta()
                      << ", " << posPlane.phi().degrees() << " deg)\n"
                      << "G4e -  Destination G4  plane position: " << surfPos << " mm, Ro = " << surfPos.perp()
                      << " mm";
  LogDebug("Geant4e") << "G4e -  Destination CMS plane normal  : " << normalPlane << "\n"
                      << "G4e -  Destination G4  plane normal  : " << normalPlane;
  LogDebug("Geant4e") << "G4e -  Distance from plane position to plane: " << pDest.localZ(posPlane) << " cm";
}

template <class SurfaceType>
void Geant4ePropagator::debugReportTrackState(std::string const &currentContext,
                                              GlobalPoint const &cmsInitPos,
                                              CLHEP::Hep3Vector const &g4InitPos,
                                              GlobalVector const &cmsInitMom,
                                              CLHEP::Hep3Vector const &g4InitMom,
                                              const SurfaceType &pDest) const {
  LogDebug("Geant4e") << "G4e - Current Context: " << currentContext;
  LogDebug("Geant4e") << "G4e -  CMS point position:" << cmsInitPos << "cm\n"
                      << "G4e -              (Ro, eta, phi): (" << cmsInitPos.perp() << " cm, " << cmsInitPos.eta()
                      << ", " << cmsInitPos.phi().degrees() << " deg)\n"
                      << "G4e -   G4  point position: " << g4InitPos << " mm, Ro = " << g4InitPos.perp() << " mm";
  LogDebug("Geant4e") << "G4e -   CMS momentum      :" << cmsInitMom << "GeV\n"
                      << " pt: " << cmsInitMom.perp() << "G4e -  G4  momentum      : " << g4InitMom << " MeV";
}
