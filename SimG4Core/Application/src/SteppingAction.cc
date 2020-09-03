
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "G4LogicalVolumeStore.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4RegionStore.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

//#define DebugLog

SteppingAction::SteppingAction(EventAction* e, const edm::ParameterSet& p, const CMSSteppingVerbose* sv, bool hasW)
    : eventAction_(e),
      tracker(nullptr),
      calo(nullptr),
      steppingVerbose(sv),
      nWarnings(0),
      initialized(false),
      killBeamPipe(false),
      hasWatcher(hasW) {
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
  ekinMins = p.getParameter<std::vector<double> >("EkinThresholds");
  ekinNames = p.getParameter<std::vector<std::string> >("EkinNames");
  ekinParticles = p.getParameter<std::vector<std::string> >("EkinParticles");

  edm::LogVerbatim("SimG4CoreApplication")
      << "SteppingAction:: KillBeamPipe = " << killBeamPipe
      << " CriticalDensity = " << theCriticalDensity * CLHEP::cm3 / CLHEP::g << " g/cm3\n"
      << "                 CriticalEnergyForVacuum = " << theCriticalEnergyForVacuum / CLHEP::MeV << " Mev;"
      << " MaxTrackTime = " << maxTrackTime / CLHEP::ns << " ns;"
      << " MaxZCentralCMS = " << maxZCentralCMS / CLHEP::m << " m"
      << " MaxTrackTimeForward = " << maxTrackTimeForward / CLHEP::ns << " ns";

  numberTimes = maxTrackTimes.size();
  if (numberTimes > 0) {
    for (unsigned int i = 0; i < numberTimes; i++) {
      edm::LogVerbatim("SimG4CoreApplication")
          << "SteppingAction::MaxTrackTime for " << maxTimeNames[i] << " is " << maxTrackTimes[i] << " ns ";
      maxTrackTimes[i] *= ns;
    }
  }

  ndeadRegions = deadRegionNames.size();
  if (ndeadRegions > 0) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "SteppingAction: Number of DeadRegions where all trackes are killed " << ndeadRegions;
    for (unsigned int i = 0; i < ndeadRegions; ++i) {
      edm::LogVerbatim("SimG4CoreApplication") << "SteppingAction: DeadRegion " << i << ".  " << deadRegionNames[i];
    }
  }
  numberEkins = ekinNames.size();
  numberPart = ekinParticles.size();
  if (0 == numberPart) {
    numberEkins = 0;
  }

  if (numberEkins > 0) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "SteppingAction::Kill following " << numberPart << " particles in " << numberEkins << " volumes";
    for (unsigned int i = 0; i < numberPart; ++i) {
      edm::LogVerbatim("SimG4CoreApplication")
          << "SteppingAction::Particle " << i << " " << ekinParticles[i] << "  Threshold = " << ekinMins[i] << " MeV";
      ekinMins[i] *= CLHEP::MeV;
    }
    for (unsigned int i = 0; i < numberEkins; ++i) {
      edm::LogVerbatim("SimG4CoreApplication") << "SteppingAction::LogVolume[" << i << "] = " << ekinNames[i];
    }
  }
}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step* aStep) {
  if (!initialized) {
    initialized = initPointer();
  }

  //if(hasWatcher) { m_g4StepSignal(aStep); }
  m_g4StepSignal(aStep);

  G4Track* theTrack = aStep->GetTrack();
  TrackStatus tstat = (theTrack->GetTrackStatus() == fAlive) ? sAlive : sKilledByProcess;

  const G4StepPoint* preStep = aStep->GetPreStepPoint();
  const G4StepPoint* postStep = aStep->GetPostStepPoint();

  // NaN energy deposit
  if (sAlive == tstat && edm::isNotFinite(aStep->GetTotalEnergyDeposit())) {
    tstat = sEnergyDepNaN;
    if (nWarnings < 20) {
      ++nWarnings;
      edm::LogWarning("SimG4CoreApplication")
          << "Track #" << theTrack->GetTrackID() << " " << theTrack->GetDefinition()->GetParticleName()
          << " E(MeV)= " << preStep->GetKineticEnergy() / MeV
          << " is killed due to edep=NaN inside PV: " << preStep->GetPhysicalVolume()->GetName() << " at "
          << theTrack->GetPosition() << " StepLen(mm)= " << aStep->GetStepLength();
    }
  }

  // check Z-coordinate
  if (sAlive == tstat && std::abs(theTrack->GetPosition().z()) >= maxZCentralCMS) {
    tstat = (theTrack->GetGlobalTime() > maxTrackTimeForward) ? sOutOfTime : sVeryForward;
  }

  // check G4Region
  if (sAlive == tstat && postStep->GetPhysicalVolume() != nullptr) {
    const G4Region* theRegion = preStep->GetPhysicalVolume()->GetLogicalVolume()->GetRegion();

    // kill in dead regions
    if (isInsideDeadRegion(theRegion)) {
      tstat = sDeadRegion;
    }

    // kill out of time
    if (sAlive == tstat && isOutOfTimeWindow(theTrack, theRegion)) {
      tstat = sOutOfTime;
    }

    // kill low-energy in volumes on demand
    if (sAlive == tstat && numberEkins > 0 && isLowEnergy(aStep)) {
      tstat = sLowEnergy;
    }

    // kill low-energy in vacuum
    G4double kinEnergy = theTrack->GetKineticEnergy();
    if (sAlive == tstat && killBeamPipe && kinEnergy < theCriticalEnergyForVacuum &&
        theTrack->GetDefinition()->GetPDGCharge() != 0.0 && kinEnergy > 0.0 &&
        theTrack->GetNextVolume()->GetLogicalVolume()->GetMaterial()->GetDensity() <= theCriticalDensity) {
      tstat = sLowEnergyInVacuum;
    }

    // check transition tracker/calo
    if (sAlive == tstat || sVeryForward == tstat) {
      if (isThisVolume(preStep->GetTouchable(), tracker) && isThisVolume(postStep->GetTouchable(), calo)) {
        math::XYZVectorD pos((preStep->GetPosition()).x(), (preStep->GetPosition()).y(), (preStep->GetPosition()).z());

        math::XYZTLorentzVectorD mom((preStep->GetMomentum()).x(),
                                     (preStep->GetMomentum()).y(),
                                     (preStep->GetMomentum()).z(),
                                     preStep->GetTotalEnergy());

        uint32_t id = theTrack->GetTrackID();

        std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> p(pos, mom);
        eventAction_->addTkCaloStateInfo(id, p);
      }
    } else {
      theTrack->SetTrackStatus(fStopAndKill);
#ifdef DebugLog
      PrintKilledTrack(theTrack, tstat);
#endif
    }
  }
  if (nullptr != steppingVerbose) {
    steppingVerbose->NextStep(aStep, fpSteppingManager, (1 < tstat));
  }
}

bool SteppingAction::isLowEnergy(const G4Step* aStep) const {
  const G4StepPoint* sp = aStep->GetPostStepPoint();
  const G4LogicalVolume* lv = sp->GetPhysicalVolume()->GetLogicalVolume();
  double ekin = sp->GetKineticEnergy();
  int pCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();

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

bool SteppingAction::initPointer() {
  const G4PhysicalVolumeStore* pvs = G4PhysicalVolumeStore::GetInstance();
  if (pvs) {
    std::vector<G4VPhysicalVolume*>::const_iterator pvcite;
    for (pvcite = pvs->begin(); pvcite != pvs->end(); ++pvcite) {
      if ((*pvcite)->GetName() == "Tracker")
        tracker = (*pvcite);
      if ((*pvcite)->GetName() == "CALO")
        calo = (*pvcite);
      if (tracker && calo)
        break;
    }
    if (tracker || calo) {
      edm::LogVerbatim("SimG4CoreApplication") << "Pointer for Tracker " << tracker << " and for Calo " << calo;
      if (tracker)
        LogDebug("SimG4CoreApplication") << "Tracker vol name " << tracker->GetName();
      if (calo)
        LogDebug("SimG4CoreApplication") << "Calorimeter vol name " << calo->GetName();
    }
  }

  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  if (numberEkins > 0) {
    if (lvs) {
      ekinVolumes.resize(numberEkins, nullptr);
      std::vector<G4LogicalVolume*>::const_iterator lvcite;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); ++lvcite) {
        for (unsigned int i = 0; i < numberEkins; ++i) {
          if ((*lvcite)->GetName() == (G4String)(ekinNames[i])) {
            ekinVolumes[i] = (*lvcite);
            break;
          }
        }
      }
    }
    for (unsigned int i = 0; i < numberEkins; ++i) {
      edm::LogVerbatim("SimG4CoreApplication") << ekinVolumes[i]->GetName() << " with pointer " << ekinVolumes[i];
    }
  }

  if (numberPart > 0) {
    G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
    G4String partName;
    ekinPDG.resize(numberPart, 0);
    for (unsigned int i = 0; i < numberPart; ++i) {
      ekinPDG[i] = theParticleTable->FindParticle(partName = ekinParticles[i])->GetPDGEncoding();
      edm::LogVerbatim("SimG4CoreApplication") << "Particle " << ekinParticles[i] << " with PDG code " << ekinPDG[i]
                                               << " and KE cut off " << ekinMins[i] / MeV << " MeV";
    }
  }

  const G4RegionStore* rs = G4RegionStore::GetInstance();
  if (numberTimes > 0) {
    maxTimeRegions.resize(numberTimes, nullptr);
    std::vector<G4Region*>::const_iterator rcite;
    for (rcite = rs->begin(); rcite != rs->end(); ++rcite) {
      for (unsigned int i = 0; i < numberTimes; ++i) {
        if ((*rcite)->GetName() == (G4String)(maxTimeNames[i])) {
          maxTimeRegions[i] = (*rcite);
          break;
        }
      }
    }
  }
  if (ndeadRegions > 0) {
    deadRegions.resize(ndeadRegions, nullptr);
    std::vector<G4Region*>::const_iterator rcite;
    for (rcite = rs->begin(); rcite != rs->end(); ++rcite) {
      for (unsigned int i = 0; i < ndeadRegions; ++i) {
        if ((*rcite)->GetName() == (G4String)(deadRegionNames[i])) {
          deadRegions[i] = (*rcite);
          break;
        }
      }
    }
  }
  return true;
}

void SteppingAction::PrintKilledTrack(const G4Track* aTrack, const TrackStatus& tst) const {
  std::string vname = "";
  std::string rname = "";
  std::string typ = " ";
  switch (tst) {
    case sKilledByProcess:
      typ = " G4Process ";
      break;
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
    default:
      break;
  }
  G4VPhysicalVolume* pv = aTrack->GetNextVolume();
  if (pv) {
    vname = pv->GetLogicalVolume()->GetName();
    rname = pv->GetLogicalVolume()->GetRegion()->GetName();
  }

  edm::LogVerbatim("SimG4CoreApplication")
      << "Track #" << aTrack->GetTrackID() << " " << aTrack->GetDefinition()->GetParticleName()
      << " E(MeV)= " << aTrack->GetKineticEnergy() / MeV << " T(ns)= " << aTrack->GetGlobalTime() / ns
      << " is killed due to " << typ << " inside LV: " << vname << " (" << rname << ") at " << aTrack->GetPosition();
}
