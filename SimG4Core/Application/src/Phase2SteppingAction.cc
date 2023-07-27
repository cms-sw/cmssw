#define EDM_ML_DEBUG

#include "SimG4Core/Application/interface/Phase2SteppingAction.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "G4LogicalVolumeStore.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4RegionStore.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#define DebugLog

Phase2SteppingAction::Phase2SteppingAction(const CMSSteppingVerbose* sv, const edm::ParameterSet& p, bool hasW)
    : steppingVerbose(sv), hasWatcher(hasW) {
  theCriticalEnergyForVacuum = (p.getParameter<double>("CriticalEnergyForVacuum") * CLHEP::MeV);
  if (0.0 < theCriticalEnergyForVacuum) {
    killBeamPipe = true;
  }
  theCriticalDensity = (p.getParameter<double>("CriticalDensity") * CLHEP::g / CLHEP::cm3);
  maxZCentralCMS = p.getParameter<double>("MaxZCentralCMS") * CLHEP::m;
  maxTrackTime = p.getParameter<double>("MaxTrackTime") * CLHEP::ns;
  maxTrackTimeForward = p.getParameter<double>("MaxTrackTimeForward") * CLHEP::ns;
  maxTrackTimes = p.getParameter<std::vector<double> >("MaxTrackTimes");
  maxTimeNames = p.getParameter<std::vector<std::string> >("MaxTimeNames");
  deadRegionNames = p.getParameter<std::vector<std::string> >("DeadRegions");
  maxNumberOfSteps = p.getParameter<int>("MaxNumberOfSteps");
  ekinMins = p.getParameter<std::vector<double> >("EkinThresholds");
  ekinNames = p.getParameter<std::vector<std::string> >("EkinNames");
  ekinParticles = p.getParameter<std::vector<std::string> >("EkinParticles");

  edm::LogVerbatim("SimG4CoreApplication")
      << "Phase2SteppingAction:: KillBeamPipe = " << killBeamPipe
      << " CriticalDensity = " << theCriticalDensity * CLHEP::cm3 / CLHEP::g << " g/cm3\n"
      << "                 CriticalEnergyForVacuum = " << theCriticalEnergyForVacuum / CLHEP::MeV << " Mev;"
      << " MaxTrackTime = " << maxTrackTime / CLHEP::ns << " ns;"
      << " MaxZCentralCMS = " << maxZCentralCMS / CLHEP::m << " m"
      << " MaxTrackTimeForward = " << maxTrackTimeForward / CLHEP::ns << " ns"
      << " MaxNumberOfSteps = " << maxNumberOfSteps;

  numberTimes = maxTrackTimes.size();
  if (numberTimes > 0) {
    for (unsigned int i = 0; i < numberTimes; i++) {
      edm::LogVerbatim("SimG4CoreApplication")
          << "Phase2SteppingAction::MaxTrackTime for " << maxTimeNames[i] << " is " << maxTrackTimes[i] << " ns ";
      maxTrackTimes[i] *= ns;
    }
  }

  ndeadRegions = deadRegionNames.size();
  if (ndeadRegions > 0) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "Phase2SteppingAction: Number of DeadRegions where all trackes are killed " << ndeadRegions;
    for (unsigned int i = 0; i < ndeadRegions; ++i) {
      edm::LogVerbatim("SimG4CoreApplication")
          << "Phase2SteppingAction: DeadRegion " << i << ".  " << deadRegionNames[i];
    }
  }
  numberEkins = ekinNames.size();
  numberPart = ekinParticles.size();
  if (0 == numberPart) {
    numberEkins = 0;
  }

  if (numberEkins > 0) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "Phase2SteppingAction::Kill following " << numberPart << " particles in " << numberEkins << " volumes";
    for (unsigned int i = 0; i < numberPart; ++i) {
      edm::LogVerbatim("SimG4CoreApplication") << "Phase2SteppingAction::Particle " << i << " " << ekinParticles[i]
                                               << "  Threshold = " << ekinMins[i] << " MeV";
      ekinMins[i] *= CLHEP::MeV;
    }
    for (unsigned int i = 0; i < numberEkins; ++i) {
      edm::LogVerbatim("SimG4CoreApplication") << "Phase2SteppingAction::LogVolume[" << i << "] = " << ekinNames[i];
    }
  }
}

void Phase2SteppingAction::UserSteppingAction(const G4Step* aStep) {
  if (!initialized) {
    initialized = initPointer();
  }

  m_g4StepSignal(aStep);

  G4Track* theTrack = aStep->GetTrack();
  TrackStatus tstat = (theTrack->GetTrackStatus() == fAlive) ? sAlive : sKilledByProcess;

  if (theTrack->GetKineticEnergy() < 0.0) {
    if (nWarnings < 2) {
      ++nWarnings;
      edm::LogWarning("SimG4CoreApplication")
          << "Phase2SteppingAction::UserPhase2SteppingAction: Track #" << theTrack->GetTrackID() << " "
          << theTrack->GetDefinition()->GetParticleName() << " Ekin(MeV)= " << theTrack->GetKineticEnergy() / MeV;
    }
    theTrack->SetKineticEnergy(0.0);
  }

  const G4StepPoint* preStep = aStep->GetPreStepPoint();
  const G4StepPoint* postStep = aStep->GetPostStepPoint();

  // the track is killed by the process
  if (tstat == sKilledByProcess) {
    if (nullptr != steppingVerbose) {
      steppingVerbose->nextStep(aStep, fpSteppingManager, false);
    }
    return;
  }

  if (sAlive == tstat && theTrack->GetCurrentStepNumber() > maxNumberOfSteps) {
    tstat = sNumberOfSteps;
    if (nWarnings < 5) {
      ++nWarnings;
      edm::LogWarning("SimG4CoreApplication")
          << "Track #" << theTrack->GetTrackID() << " " << theTrack->GetDefinition()->GetParticleName()
          << " E(MeV)= " << preStep->GetKineticEnergy() / MeV << " Nstep= " << theTrack->GetCurrentStepNumber()
          << " is killed due to limit on number of steps;/n  PV: " << preStep->GetPhysicalVolume()->GetName() << " at "
          << theTrack->GetPosition() << " StepLen(mm)= " << aStep->GetStepLength();
    }
  }
  const double time = theTrack->GetGlobalTime();

  // check Z-coordinate
  if (sAlive == tstat && std::abs(theTrack->GetPosition().z()) >= maxZCentralCMS) {
    tstat = (time > maxTrackTimeForward) ? sOutOfTime : sVeryForward;
  }

  // check G4Region
  if (sAlive == tstat) {
    // next logical volume and next region
    const G4LogicalVolume* lv = postStep->GetPhysicalVolume()->GetLogicalVolume();
    const G4Region* theRegion = lv->GetRegion();

    // kill in dead regions
    if (isInsideDeadRegion(theRegion))
      tstat = sDeadRegion;

    // kill out of time
    if (sAlive == tstat) {
      if (isOutOfTimeWindow(theRegion, time))
        tstat = sOutOfTime;
    }

    // kill low-energy in volumes on demand
    if (sAlive == tstat && numberEkins > 0) {
      if (isLowEnergy(lv, theTrack))
        tstat = sLowEnergy;
    }

    // kill low-energy in vacuum
    if (sAlive == tstat && killBeamPipe) {
      if (theTrack->GetKineticEnergy() < theCriticalEnergyForVacuum &&
          theTrack->GetDefinition()->GetPDGCharge() != 0.0 && lv->GetMaterial()->GetDensity() <= theCriticalDensity) {
        tstat = sLowEnergyInVacuum;
      }
    }
  }
  // check transition tracker/btl and tracker/calo
  bool isKilled = false;
  if (sAlive == tstat || sVeryForward == tstat) {
    // store TrackInformation about transition from one envelope to another
    if (preStep->GetPhysicalVolume() == tracker && postStep->GetPhysicalVolume() == btl) {
      // store transition tracker -> BTL only for tracks entering BTL for the first time
      TrackInformation* trkinfo = static_cast<TrackInformation*>(theTrack->GetUserInformation());
      if (!trkinfo->isFromTtoBTL() && !trkinfo->isFromBTLtoT()) {
        trkinfo->setFromTtoBTL();
        trkinfo->setIdAtBTLentrance(trkinfo->mcTruthID());
#ifdef DebugLog
        LogDebug("SimG4CoreApplication") << "Setting flag for Tracker -> BTL " << trkinfo->isFromTtoBTL()
                                         << " IdAtBTLentrance = " << trkinfo->idAtBTLentrance();
#endif
      } else {
        trkinfo->setBTLlooper();
        trkinfo->setIdAtBTLentrance(trkinfo->mcTruthID());
#ifdef DebugLog
        LogDebug("SimG4CoreApplication") << "Setting flag for BTL looper " << trkinfo->isBTLlooper();
        trkinfo->Print();
#endif
      }
    } else if (preStep->GetPhysicalVolume() == btl && postStep->GetPhysicalVolume() == tracker) {
      // store transition BTL -> tracker
      TrackInformation* trkinfo = static_cast<TrackInformation*>(theTrack->GetUserInformation());
      if (!trkinfo->isFromBTLtoT()) {
        trkinfo->setFromBTLtoT();
#ifdef DebugLog
        LogDebug("SimG4CoreApplication") << "Setting flag for BTL -> Tracker " << trkinfo->isFromBTLtoT();
#endif
      }
    } else if (preStep->GetPhysicalVolume() == tracker && postStep->GetPhysicalVolume() == calo) {
      // store transition tracker -> calo
      TrackInformation* trkinfo = static_cast<TrackInformation*>(theTrack->GetUserInformation());
      if (!trkinfo->crossedBoundary()) {
        trkinfo->setCrossedBoundary(theTrack);
      }
    } else if (preStep->GetPhysicalVolume() == calo && postStep->GetPhysicalVolume() == cmse) {
      // store transition calo -> cmse to tag backscattering
      TrackInformation* trkinfo = static_cast<TrackInformation*>(theTrack->GetUserInformation());
      if (!trkinfo->isInTrkFromBackscattering()) {
        trkinfo->setInTrkFromBackscattering();
#ifdef DebugLog
        LogDebug("SimG4CoreApplication") << "Setting flag for backscattering from CALO "
                                         << trkinfo->isInTrkFromBackscattering();
#endif
      }
    }
  } else {
    theTrack->SetTrackStatus(fStopAndKill);
    isKilled = true;
#ifdef DebugLog
    PrintKilledTrack(theTrack, tstat);
#endif
  }
  if (nullptr != steppingVerbose) {
    steppingVerbose->nextStep(aStep, fpSteppingManager, isKilled);
  }
}

bool Phase2SteppingAction::isLowEnergy(const G4LogicalVolume* lv, const G4Track* theTrack) const {
  const double ekin = theTrack->GetKineticEnergy();
  int pCode = theTrack->GetDefinition()->GetPDGEncoding();

  for (auto& vol : ekinVolumes) {
    if (lv == vol) {
      for (unsigned int i = 0; i < numberPart; ++i) {
        if (pCode == ekinPDG[i]) {
          return (ekin <= ekinMins[i]);
        }
      }
      break;
    }
  }
  return false;
}

bool Phase2SteppingAction::initPointer() {
  const G4PhysicalVolumeStore* pvs = G4PhysicalVolumeStore::GetInstance();
  for (auto const& pvcite : *pvs) {
    const G4String& pvname = pvcite->GetName();
    if (pvname == "Tracker" || pvname == "tracker:Tracker_1") {
      tracker = pvcite;
    } else if (pvname == "CALO" || pvname == "caloBase:CALO_1") {
      calo = pvcite;
    } else if (pvname == "BarrelTimingLayer" || pvname == "btl:BarrelTimingLayer_1") {
      btl = pvcite;
    } else if (pvname == "CMSE" || pvname == "cms:CMSE_1") {
      cmse = pvcite;
    }
    if (tracker && calo && btl && cmse)
      break;
  }
  edm::LogVerbatim("SimG4CoreApplication")
      << "Phase2SteppingAction: pointer for Tracker " << tracker << " and for Calo " << calo << " and for BTL " << btl;

  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  if (numberEkins > 0) {
    ekinVolumes.resize(numberEkins, nullptr);
    for (auto const& lvcite : *lvs) {
      const G4String& lvname = lvcite->GetName();
      for (unsigned int i = 0; i < numberEkins; ++i) {
        if (lvname == (G4String)(ekinNames[i])) {
          ekinVolumes[i] = lvcite;
          break;
        }
      }
    }
    for (unsigned int i = 0; i < numberEkins; ++i) {
      edm::LogVerbatim("SimG4CoreApplication") << ekinVolumes[i]->GetName() << " with pointer " << ekinVolumes[i];
    }
  }

  if (numberPart > 0) {
    G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
    ekinPDG.resize(numberPart, 0);
    for (unsigned int i = 0; i < numberPart; ++i) {
      const G4ParticleDefinition* part = theParticleTable->FindParticle(ekinParticles[i]);
      if (nullptr != part)
        ekinPDG[i] = part->GetPDGEncoding();
      edm::LogVerbatim("SimG4CoreApplication") << "Particle " << ekinParticles[i] << " with PDG code " << ekinPDG[i]
                                               << " and KE cut off " << ekinMins[i] / MeV << " MeV";
    }
  }

  const G4RegionStore* rs = G4RegionStore::GetInstance();
  if (numberTimes > 0) {
    maxTimeRegions.resize(numberTimes, nullptr);
    for (auto const& rcite : *rs) {
      const G4String& rname = rcite->GetName();
      for (unsigned int i = 0; i < numberTimes; ++i) {
        if (rname == (G4String)(maxTimeNames[i])) {
          maxTimeRegions[i] = rcite;
          break;
        }
      }
    }
  }
  if (ndeadRegions > 0) {
    deadRegions.resize(ndeadRegions, nullptr);
    for (auto const& rcite : *rs) {
      const G4String& rname = rcite->GetName();
      for (unsigned int i = 0; i < ndeadRegions; ++i) {
        if (rname == (G4String)(deadRegionNames[i])) {
          deadRegions[i] = rcite;
          break;
        }
      }
    }
  }
  return true;
}

void Phase2SteppingAction::PrintKilledTrack(const G4Track* aTrack, const TrackStatus& tst) const {
  std::string vname = "";
  std::string rname = "";
  std::string typ = " ";
  switch (tst) {
    case sDeadRegion:
      typ = " in dead region ";
      break;
    case sOutOfTime:
      typ = " out of time window ";
      break;
    case sLowEnergy:
      typ = " low energy limit ";
      break;
    case sLowEnergyInVacuum:
      typ = " low energy limit in vacuum ";
      break;
    case sEnergyDepNaN:
      typ = " energy deposition is NaN ";
      break;
    case sVeryForward:
      typ = " very forward track ";
      break;
    case sNumberOfSteps:
      typ = " too many steps ";
      break;
    default:
      break;
  }
  G4VPhysicalVolume* pv = aTrack->GetNextVolume();
  vname = pv->GetLogicalVolume()->GetName();
  rname = pv->GetLogicalVolume()->GetRegion()->GetName();

  const double ekin = aTrack->GetKineticEnergy();
  if (ekin < 2 * CLHEP::MeV) {
    return;
  }
  edm::LogWarning("SimG4CoreApplication")
      << "Track #" << aTrack->GetTrackID() << " StepN= " << aTrack->GetCurrentStepNumber() << " "
      << aTrack->GetDefinition()->GetParticleName() << " E(MeV)=" << ekin / CLHEP::MeV
      << " T(ns)=" << aTrack->GetGlobalTime() / CLHEP::ns << " is killed due to " << typ << "\n  LV: " << vname << " ("
      << rname << ") at " << aTrack->GetPosition() << " step(cm)=" << aTrack->GetStep()->GetStepLength() / CLHEP::cm;
}
