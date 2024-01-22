#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"
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

//#define EDM_ML_DEBUG

SteppingAction::SteppingAction(const CMSSteppingVerbose* sv, const edm::ParameterSet& p, bool hasW, bool dd4hep)
    : steppingVerbose(sv), hasWatcher(hasW), dd4hep_(dd4hep) {
  theCriticalEnergyForVacuum = (p.getParameter<double>("CriticalEnergyForVacuum") * CLHEP::MeV);
  if (0.0 < theCriticalEnergyForVacuum) {
    killBeamPipe = true;
  }
  m_CMStoZDCtransport = (p.getParameter<bool>("CMStoZDCtransport"));
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
  trackerName_ = p.getParameter<std::string>("TrackerName");
  caloName_ = p.getParameter<std::string>("CaloName");
  cms2ZDCName_ = p.getParameter<std::string>("CMS2ZDCName");

  edm::LogVerbatim("SimG4CoreApplication")
      << "SteppingAction:: KillBeamPipe = " << killBeamPipe
      << " CriticalDensity = " << theCriticalDensity * CLHEP::cm3 / CLHEP::g << " g/cm3\n"
      << "                 CriticalEnergyForVacuum = " << theCriticalEnergyForVacuum / CLHEP::MeV << " Mev;"
      << " MaxTrackTime = " << maxTrackTime / CLHEP::ns << " ns;"
      << " MaxZCentralCMS = " << maxZCentralCMS / CLHEP::m << " m"
      << " MaxTrackTimeForward = " << maxTrackTimeForward / CLHEP::ns << " ns"
      << " MaxNumberOfSteps = " << maxNumberOfSteps << " ZDC: " << m_CMStoZDCtransport << "\n"
      << "                 Names of special volumes: " << trackerName_ << "  " << caloName_;

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

void SteppingAction::UserSteppingAction(const G4Step* aStep) {
  if (!initialized) {
    initialized = initPointer();
  }

  m_g4StepSignal(aStep);

  G4Track* theTrack = aStep->GetTrack();
  TrackStatus tstat = (theTrack->GetTrackStatus() == fAlive) ? sAlive : sKilledByProcess;
  const double ekin = theTrack->GetKineticEnergy();

  if (ekin < 0.0) {
    if (nWarnings < 2) {
      ++nWarnings;
      edm::LogWarning("SimG4CoreApplication")
          << "SteppingAction::UserSteppingAction: Track #" << theTrack->GetTrackID() << " "
          << theTrack->GetDefinition()->GetParticleName() << " Ekin(MeV)=" << ekin;
    }
    theTrack->SetKineticEnergy(0.0);
  }

  // the track is killed by the process
  if (tstat == sKilledByProcess) {
    if (nullptr != steppingVerbose) {
      steppingVerbose->nextStep(aStep, fpSteppingManager, false);
    }
    return;
  }

  const G4StepPoint* preStep = aStep->GetPreStepPoint();
  const G4StepPoint* postStep = aStep->GetPostStepPoint();
  if (sAlive == tstat && theTrack->GetCurrentStepNumber() > maxNumberOfSteps) {
    tstat = sNumberOfSteps;
    if (nWarnings < 5) {
      ++nWarnings;
      edm::LogWarning("SimG4CoreApplication")
          << "Track #" << theTrack->GetTrackID() << " " << theTrack->GetDefinition()->GetParticleName()
          << " E(MeV)=" << ekin << " Nstep=" << theTrack->GetCurrentStepNumber()
          << " is killed due to limit on number of steps;/n  PV:" << preStep->GetPhysicalVolume()->GetName() << " at "
          << theTrack->GetPosition() << " StepLen(mm)=" << aStep->GetStepLength();
    }
  }

  const double time = theTrack->GetGlobalTime();

  // check Z-coordinate
  if (sAlive == tstat && std::abs(theTrack->GetPosition().z()) >= maxZCentralCMS) {
    tstat = (time > maxTrackTimeForward) ? sOutOfTime : sVeryForward;
  }

  // check G4Region
  if (sAlive == tstat || sVeryForward == tstat) {
    // next logical volume and next region
    const G4LogicalVolume* lv = postStep->GetPhysicalVolume()->GetLogicalVolume();
    const G4Region* theRegion = lv->GetRegion();

    // kill in dead regions except CMStoZDC volume
    if (isInsideDeadRegion(theRegion) && !isForZDC(lv, std::abs(theTrack->GetParticleDefinition()->GetPDGEncoding()))) {
      tstat = sDeadRegion;
    }

    // kill particles leaving ZDC
    if (sAlive == sVeryForward && m_CMStoZDCtransport) {
      const G4Region* preRegion = preStep->GetPhysicalVolume()->GetLogicalVolume()->GetRegion();
      if (preRegion == m_ZDCRegion && preRegion != theRegion)
        tstat = sDeadRegion;
    }

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
      if (ekin < theCriticalEnergyForVacuum && theTrack->GetDefinition()->GetPDGCharge() != 0.0 &&
          lv->GetMaterial()->GetDensity() <= theCriticalDensity) {
        tstat = sLowEnergyInVacuum;
      }
    }
  }
  // check transition tracker/calo
  bool isKilled = false;
  if (sAlive == tstat || sVeryForward == tstat) {
    if (preStep->GetPhysicalVolume() == tracker && postStep->GetPhysicalVolume() == calo) {
      TrackInformation* trkinfo = static_cast<TrackInformation*>(theTrack->GetUserInformation());
      if (!trkinfo->crossedBoundary()) {
        trkinfo->setCrossedBoundary(theTrack);
      }
    }
  } else {
    theTrack->SetTrackStatus(fStopAndKill);
    isKilled = true;
#ifdef EDM_ML_DEBUG
    PrintKilledTrack(theTrack, tstat);
#endif
  }
  if (nullptr != steppingVerbose) {
    steppingVerbose->nextStep(aStep, fpSteppingManager, isKilled);
  }
}

bool SteppingAction::isLowEnergy(const G4LogicalVolume* lv, const G4Track* theTrack) const {
  const double ekin = theTrack->GetKineticEnergy();
  int pCode = theTrack->GetDefinition()->GetPDGEncoding();

  for (auto const& vol : ekinVolumes) {
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
  for (auto const& pvcite : *pvs) {
    const std::string& pvname = (std::string)(DD4hep2DDDName::namePV(pvcite->GetName(), dd4hep_));
    if (pvname == trackerName_) {
      tracker = pvcite;
    } else if (pvname == caloName_) {
      calo = pvcite;
    }
    if (tracker && calo)
      break;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("SimG4CoreApplication") << pvs->size() << " Physical volume in the store";
  for (auto const& pvcite : *pvs)
    edm::LogVerbatim("SimG4CoreApplication") << pvcite << " corresponds to " << pvcite->GetName();
#endif

  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  ekinVolumes.resize(numberEkins, nullptr);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("SimG4CoreApplication") << lvs->size() << " Logical volume in the store";
  for (auto const& lvcite : *lvs)
    edm::LogVerbatim("SimG4CoreApplication") << lvcite << " corresponds to " << lvcite->GetName();
#endif
  for (auto const& lvcite : *lvs) {
    std::string lvname = (std::string)(DD4hep2DDDName::nameMatterLV(lvcite->GetName(), dd4hep_));
    if (lvname == cms2ZDCName_) {
      m_CMStoZDC = lvcite;
    }
    for (unsigned int i = 0; i < numberEkins; ++i) {
      if (lvname == ekinNames[i]) {
        ekinVolumes[i] = lvcite;
        break;
      }
    }
  }
  edm::LogVerbatim("SimG4CoreApplication")
      << "SteppingAction: pointer for Tracker: " << tracker << "; Calo: " << calo << "; to CMStoZDC: " << m_CMStoZDC;
  for (unsigned int i = 0; i < numberEkins; ++i) {
    edm::LogVerbatim("SimG4CoreApplication") << ekinVolumes[i]->GetName() << " with pointer " << ekinVolumes[i];
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
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("SimG4CoreApplication") << rs->size() << " Regions in the store";
  for (auto const& rcite : *rs)
    edm::LogVerbatim("SimG4CoreApplication") << rcite << " corresponds to " << rcite->GetName();
#endif
  for (auto const& rcite : *rs) {
    const G4String& rname = rcite->GetName();
    if (numberTimes > 0) {
      maxTimeRegions.resize(numberTimes, nullptr);
      for (unsigned int i = 0; i < numberTimes; ++i) {
        if (rname == (G4String)(maxTimeNames[i])) {
          maxTimeRegions[i] = rcite;
          break;
        }
      }
    }
    if (ndeadRegions > 0) {
      deadRegions.resize(ndeadRegions, nullptr);
      for (unsigned int i = 0; i < ndeadRegions; ++i) {
        if (rname == (G4String)(deadRegionNames[i])) {
          deadRegions[i] = rcite;
          break;
        }
      }
    }
    if (m_CMStoZDCtransport && rname == "ZDCRegion") {
      m_ZDCRegion = rcite;
    }
  }
  return true;
}

void SteppingAction::PrintKilledTrack(const G4Track* aTrack, const TrackStatus& tst) const {
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
