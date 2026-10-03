#include "SimG4Core/CustomPhysics/interface/QuirkTransportation.h"
#include "SimG4Core/CustomPhysics/interface/QuirkHyperbolaStepper.h"
#include "SimG4Core/CustomPhysics/interface/QuirkStringStore.h"
#include "SimG4Core/CustomPhysics/interface/Quirk.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4ProductionCutsTable.hh"
#include "G4ChordFinder.hh"
#include "G4SafetyHelper.hh"
#include "G4FieldManagerStore.hh"
#include "G4MagIntegratorDriver.hh"
#include "G4TransportationProcessType.hh"
#include "G4Exception.hh"
#include <CLHEP/Units/SystemOfUnits.h>

QuirkTransportation::QuirkTransportation(G4int verb, G4bool moveTinySteps, G4double crossingLength)
    : G4VProcess("QuirkTransportation", fTransportation),
      fThreshold_Warning_Energy(100 * CLHEP::MeV),
      fThreshold_Important_Energy(250 * CLHEP::MeV),
      fThresholdTrials(10),
      fMoveTinySteps(moveTinySteps),
      fCrossingLength(crossingLength) {
  verboseLevel = verb;
  SetProcessSubType(TRANSPORTATION);

#ifdef G4MULTITHREADED
  // the master never tracks
  if (G4Threading::IsMasterThread()) {
    return;
  }
#endif

  G4TransportationManager* transportMgr = G4TransportationManager::GetTransportationManager();
  fLinearNavigator = transportMgr->GetNavigatorForTracking();
  fFieldPropagator = transportMgr->GetPropagatorInField();
  fpSafetyHelper = transportMgr->GetSafetyHelper();
}

QuirkTransportation::~QuirkTransportation() {
  if (verboseLevel > 0 && fSumEnergyKilled > 0.0) {
    edm::LogVerbatim("SimG4CoreCustomPhysics")
        << "QuirkTransportation: loopers killed, sum of energy " << fSumEnergyKilled / CLHEP::GeV << " GeV, max "
        << fMaxEnergyKilled / CLHEP::GeV << " GeV";
  }
}

G4double QuirkTransportation::AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                                    G4double,  // previousStepSize
                                                                    G4double currentMinimumStep,
                                                                    G4double& currentSafety,
                                                                    G4GPILSelection* selection) {
  fParticleIsLooping = false;
  *selection = CandidateForSelection;

  const G4DynamicParticle* pParticle = track.GetDynamicParticle();
  const G4ThreeVector& startPosition = track.GetPosition();

  // isotropic safety at the start point
  G4ThreeVector OriginShift = startPosition - fPreviousSftOrigin;
  G4double MagSqShift = OriginShift.mag2();
  if (MagSqShift >= sqr(fPreviousSafety)) {
    currentSafety = 0.0;
  } else {
    currentSafety = fPreviousSafety - std::sqrt(MagSqShift);
  }

  // field manager of this volume (the global CMSFieldManager in CMS)
  G4FieldManager* fieldMgr = fFieldPropagator->FindAndSetFieldManager(track.GetVolume());
  if (nullptr == fieldMgr) {
    G4Exception("QuirkTransportation::AlongStepGPIL", "QuirkNoFieldMgr", RunMustBeAborted, "no field manager");
    return 0.0;
  }
  fieldMgr->ConfigureForTrack(&track);

  // a head-on pair approaches its crossing point in ever shorter steps: restart the string there
  QuirkInfracolorForce& string = QuirkStringStore::instance().stringFor(pParticle->GetPDGcode());
  if (fCrossingLength > 0. && string.HasNextStringVector() && string.IsSourceAlive() && string.IsSourceInitialized() &&
      string.GetStringVectors().front().t() < fCrossingLength && string.GetStringVectors().size() == 1) {
    G4LorentzVector u = pParticle->Get4Momentum();
    string.Restart(u / u.m());
  }

  // stepper replaying the quirk trajectory
  QuirkHyperbolaStepper quirkStepper(
      string, track, fieldMgr->DoesFieldExist() ? fieldMgr->GetDetectorField() : nullptr);

  // swap in the quirk chord finder for this step only
  G4ChordFinder* oldChordFinder = fieldMgr->GetChordFinder();
  G4ChordFinder quirkChordFinder(new G4MagInt_Driver(0.0, &quirkStepper, quirkStepper.GetNumberOfVariables()));
  fieldMgr->SetChordFinder(&quirkChordFinder);

  const G4ThreeVector& spin = track.GetPolarization();
  G4FieldTrack aFieldTrack(startPosition,
                           track.GetMomentumDirection(),
                           0.0,
                           track.GetKineticEnergy(),
                           pParticle->GetMass(),
                           track.GetVelocity(),
                           track.GetGlobalTime(),
                           0.0,  // step length in place of proper time
                           &spin);

  currentMinimumStep = std::min(currentMinimumStep, quirkStepper.GetMaxLength());
  if (currentMinimumStep > 0) {
    // field propagator handles the boundary crossings
    G4double lengthAlongCurve =
        fFieldPropagator->ComputeStep(aFieldTrack, currentMinimumStep, currentSafety, track.GetVolume());
    fGeometryLimitedStep = lengthAlongCurve < currentMinimumStep;

    // update stepper and strings with the length actually taken
    quirkStepper.Update(aFieldTrack, aFieldTrack.GetCurveLength() == 0 && !fGeometryLimitedStep, fMoveTinySteps);
  } else {
    fGeometryLimitedStep = false;
  }

  fPreviousSftOrigin = startPosition;
  fPreviousSafety = currentSafety;

  // end-of-step quantities
  G4double geometryStepLength = aFieldTrack.GetCurveLength();
  fTransportEndPosition = aFieldTrack.GetPosition();
  fMomentumChanged = true;
  fTransportEndMomentumDir = aFieldTrack.GetMomentumDir();
  fTransportEndKineticEnergy = aFieldTrack.GetKineticEnergy();
  fCandidateEndGlobalTime = aFieldTrack.GetLabTimeOfFlight();
  fTransportEndSpin = aFieldTrack.GetSpin();
  {
    // remember the 4-velocity for a later restart of the string
    const G4double m = pParticle->GetMass();
    const G4double e = fTransportEndKineticEnergy + m;
    string.SetVelocity(G4LorentzVector(std::sqrt(std::max(e * e - m * m, 0.)) * fTransportEndMomentumDir, e) / m);
  }
  fParticleIsLooping = fFieldPropagator->IsParticleLooping();
  fEndpointDistance = (fTransportEndPosition - startPosition).mag();

  // zero step on a boundary is limited by the boundary
  if (currentMinimumStep == 0.0 && currentSafety == 0.0) {
    fGeometryLimitedStep = true;
  }

  // safety from the end point if it becomes negative there
  if (currentSafety < fEndpointDistance) {
    G4double endSafety = fLinearNavigator->ComputeSafety(fTransportEndPosition);
    currentSafety = endSafety;
    fPreviousSftOrigin = fTransportEndPosition;
    fPreviousSafety = currentSafety;
    fpSafetyHelper->SetCurrentSafety(currentSafety, fTransportEndPosition);
    // the stepping manager assumes it is from the start point
    currentSafety += fEndpointDistance;
  }

  fParticleChange.ProposeTrueStepLength(geometryStepLength);

  // restore the CMS chord finder
  fieldMgr->SetChordFinder(oldChordFinder);

  return geometryStepLength;
}

G4VParticleChange* QuirkTransportation::AlongStepDoIt(const G4Track& track, const G4Step&) {
  fParticleChange.Initialize(track);

  fParticleChange.ProposePosition(fTransportEndPosition);
  fParticleChange.ProposeMomentumDirection(fTransportEndMomentumDir);
  fParticleChange.ProposeEnergy(fTransportEndKineticEnergy);
  fParticleChange.SetMomentumChanged(fMomentumChanged);
  fParticleChange.ProposePolarization(fTransportEndSpin);

  G4double deltaTime = fCandidateEndGlobalTime - track.GetGlobalTime();
  fParticleChange.ProposeGlobalTime(fCandidateEndGlobalTime);

  // proper time from the Lorentz factor
  G4double deltaProperTime = deltaTime * (track.GetDynamicParticle()->GetMass() / track.GetTotalEnergy());
  fParticleChange.ProposeProperTime(track.GetProperTime() + deltaProperTime);

  // kill a particle stuck looping in the field
  if (fParticleIsLooping) {
    G4double endEnergy = fTransportEndKineticEnergy;
    if ((endEnergy < fThreshold_Important_Energy) || (fNoLooperTrials >= fThresholdTrials)) {
      fParticleChange.ProposeTrackStatus(fStopAndKill);
      fSumEnergyKilled += endEnergy;
      fMaxEnergyKilled = std::max(fMaxEnergyKilled, endEnergy);
      if (verboseLevel > 1 || endEnergy > fThreshold_Warning_Energy) {
        edm::LogWarning("SimG4CoreCustomPhysics")
            << "QuirkTransportation is killing a looping quirk with Ekin(GeV)= "
            << track.GetKineticEnergy() / CLHEP::GeV << " after " << fNoLooperTrials << " trials";
      }
      fNoLooperTrials = 0;
    } else {
      ++fNoLooperTrials;
    }
  } else {
    fNoLooperTrials = 0;
  }

  fParticleChange.SetPointerToVectorOfAuxiliaryPoints(fFieldPropagator->GimmeTrajectoryVectorAndForgetIt());

  return &fParticleChange;
}

G4double QuirkTransportation::PostStepGetPhysicalInteractionLength(const G4Track&,
                                                                   G4double,  // previousStepSize
                                                                   G4ForceCondition* pForceCond) {
  *pForceCond = Forced;
  return DBL_MAX;
}

G4VParticleChange* QuirkTransportation::PostStepDoIt(const G4Track& track, const G4Step&) {
  G4TouchableHandle retCurrentTouchable;
  G4bool isLastStep = false;

  fParticleChange.ProposeTrackStatus(track.GetTrackStatus());

  if (fGeometryLimitedStep) {
    // relocate the particle after a boundary-limited step
    fLinearNavigator->SetGeometricallyLimitedStep();
    fLinearNavigator->LocateGlobalPointAndUpdateTouchableHandle(
        track.GetPosition(), track.GetMomentumDirection(), fCurrentTouchableHandle, true);
    // out of the world volume
    if (fCurrentTouchableHandle->GetVolume() == nullptr) {
      fParticleChange.ProposeTrackStatus(fStopAndKill);
      if (verboseLevel > 0) {
        edm::LogVerbatim("SimG4CoreCustomPhysics")
            << "QuirkTransportation: quirk " << track.GetTrackID() << " left the world after "
            << track.GetCurrentStepNumber() << " steps";
      }
    }
    retCurrentTouchable = fCurrentTouchableHandle;
    fParticleChange.SetTouchableHandle(fCurrentTouchableHandle);
    isLastStep = fLinearNavigator->ExitedMotherVolume() || fLinearNavigator->EnteredDaughterVolume();
  } else {
    // only moves the navigator's location
    fLinearNavigator->LocateGlobalPointWithinVolume(track.GetPosition());
    fParticleChange.SetTouchableHandle(track.GetTouchableHandle());
    retCurrentTouchable = track.GetTouchableHandle();
  }
  fParticleChange.ProposeLastStepInVolume(isLastStep);

  const G4VPhysicalVolume* pNewVol = retCurrentTouchable->GetVolume();
  const G4Material* pNewMaterial = nullptr;
  G4VSensitiveDetector* pNewSensitiveDetector = nullptr;
  const G4MaterialCutsCouple* pNewMaterialCutsCouple = nullptr;
  if (pNewVol != nullptr) {
    pNewMaterial = pNewVol->GetLogicalVolume()->GetMaterial();
    pNewSensitiveDetector = pNewVol->GetLogicalVolume()->GetSensitiveDetector();
    pNewMaterialCutsCouple = pNewVol->GetLogicalVolume()->GetMaterialCutsCouple();
  }
  fParticleChange.SetMaterialInTouchable(const_cast<G4Material*>(pNewMaterial));
  fParticleChange.SetSensitiveDetectorInTouchable(pNewSensitiveDetector);

  // parameterized volumes
  if (pNewVol != nullptr && pNewMaterialCutsCouple != nullptr &&
      pNewMaterialCutsCouple->GetMaterial() != pNewMaterial) {
    pNewMaterialCutsCouple = G4ProductionCutsTable::GetProductionCutsTable()->GetMaterialCutsCouple(
        pNewMaterial, pNewMaterialCutsCouple->GetProductionCuts());
  }
  fParticleChange.SetMaterialCutsCoupleInTouchable(pNewMaterialCutsCouple);
  fParticleChange.SetTouchableHandle(retCurrentTouchable);

  return &fParticleChange;
}

void QuirkTransportation::StartTracking(G4Track* aTrack) {
  G4VProcess::StartTracking(aTrack);

  // reset safety and looper counter, also on resumption of a suspended quirk
  fPreviousSafety = 0.0;
  fPreviousSftOrigin = G4ThreeVector(0., 0., 0.);
  fNoLooperTrials = 0;

  fFieldPropagator->ClearPropagatorState();
  G4FieldManagerStore::GetInstance()->ClearAllChordFindersState();

  fCurrentTouchableHandle = aTrack->GetTouchableHandle();

  // first string vector at the first step of the pair
  QuirkStringStore::instance().stringFor(aTrack->GetDefinition()->GetPDGEncoding()).StartTracking(aTrack);
}
