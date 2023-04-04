///////////////////////////////////////////////////////////////////////////////
// File: TimingSD.cc
// Date: 02.2006
// Description: Sensitive Detector class for Timing
// Modifications:
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/TimingSD.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <vector>
#include <iostream>

//#define EDM_ML_DEBUG

static const float invgev = 1.0 / CLHEP::GeV;
static const double invns = 1.0 / CLHEP::nanosecond;
static const double invdeg = 1.0 / CLHEP::deg;

//-------------------------------------------------------------------
TimingSD::TimingSD(const std::string& name, const SensitiveDetectorCatalog& clg, const SimTrackManager* manager)
    : SensitiveTkDetector(name, clg),
      theManager(manager),
      theHC(nullptr),
      currentHit(nullptr),
      theTrack(nullptr),
      preStepPoint(nullptr),
      postStepPoint(nullptr),
      unitID(0),
      previousUnitID(0),
      primID(-2),
      hcID(-1),
      tsID(-2),
      primaryID(0),
      tSliceID(-1),
      timeFactor(1.0),
      energyCut(1.e+9),
      energyHistoryCut(1.e+9) {
  slave = new TrackingSlaveSD(name);
  theEnumerator = new G4ProcessTypeEnumerator();
}

TimingSD::~TimingSD() {
  delete slave;
  delete theEnumerator;
}

void TimingSD::Initialize(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("TimingSim") << "TimingSD : Initialize called for " << GetName() << " time slice factor "
                                << timeFactor << "\n MC truth cuts in are " << energyCut / CLHEP::GeV << " GeV and "
                                << energyHistoryCut / CLHEP::GeV << " GeV";

  theHC = new BscG4HitCollection(GetName(), collectionName[0]);
  if (hcID < 0) {
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  }
  HCE->AddHitsCollection(hcID, theHC);

  tsID = -2;
  primID = -2;
}

void TimingSD::setTimeFactor(double val) {
  if (val <= 0.0) {
    return;
  }
  timeFactor = val;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimingSim") << "TimingSD : for " << GetName() << " time slice factor is set to " << timeFactor;
#endif
}

void TimingSD::setCuts(double eCut, double historyCut) {
  if (eCut > 0.) {
    energyCut = eCut;
  }
  if (historyCut > 0.) {
    energyHistoryCut = historyCut;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimingSim") << "TimingSD : for " << GetName() << " MC truth cuts in are " << energyCut / CLHEP::GeV
                                << " GeV and " << energyHistoryCut / CLHEP::GeV << " GeV";
#endif
}

bool TimingSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  edeposit = aStep->GetTotalEnergyDeposit();
  if (edeposit > 0.f) {
    getStepInfo(aStep);
    if (!hitExists(aStep)) {
      createNewHit(aStep);
    }
  }
  return true;
}

void TimingSD::getStepInfo(const G4Step* aStep) {
  preStepPoint = aStep->GetPreStepPoint();
  postStepPoint = aStep->GetPostStepPoint();
  hitPointExit = postStepPoint->GetPosition();
  setToLocal(preStepPoint, hitPointExit, hitPointLocalExit);
  const G4Track* newTrack = aStep->GetTrack();

  // neutral particles deliver energy post step
  // charged particle start deliver energy pre step
  if (newTrack->GetDefinition()->GetPDGCharge() == 0.0) {
    hitPoint = hitPointExit;
    hitPointLocal = hitPointLocalExit;
    tof = (float)(postStepPoint->GetGlobalTime() * invns);
  } else {
    hitPoint = preStepPoint->GetPosition();
    setToLocal(preStepPoint, hitPoint, hitPointLocal);
    tof = (float)(preStepPoint->GetGlobalTime() * invns);
  }

#ifdef EDM_ML_DEBUG
  double distGlobal =
      std::sqrt(std::pow(hitPoint.x() - hitPointExit.x(), 2) + std::pow(hitPoint.y() - hitPointExit.y(), 2) +
                std::pow(hitPoint.z() - hitPointExit.z(), 2));
  double distLocal = std::sqrt(std::pow(hitPointLocal.x() - hitPointLocalExit.x(), 2) +
                               std::pow(hitPointLocal.y() - hitPointLocalExit.y(), 2) +
                               std::pow(hitPointLocal.z() - hitPointLocalExit.z(), 2));
  LogDebug("TimingSim") << "TimingSD:"
                        << "\n Global entry point: " << hitPoint << "\n Global exit  point: " << hitPointExit
                        << "\n Global step length: " << distGlobal << "\n Local  entry point: " << hitPointLocal
                        << "\n Local  exit  point: " << hitPointLocalExit << "\n Local  step length: " << distLocal;
  if (std::fabs(distGlobal - distLocal) > 1.e-6) {
    LogDebug("TimingSim") << "DIFFERENCE IN DISTANCE \n";
  }
#endif

  incidentEnergy = preStepPoint->GetKineticEnergy();

  // should MC truth be saved
  if (newTrack != theTrack) {
    theTrack = newTrack;
    TrackInformation* info = nullptr;
    if (incidentEnergy > energyCut) {
      info = cmsTrackInformation(theTrack);
      info->setStoreTrack();
    }
    if (incidentEnergy > energyHistoryCut) {
      if (nullptr == info) {
        info = cmsTrackInformation(theTrack);
      }
      info->putInHistory();
    }
#ifdef EDM_ML_DEBUG
    if (info != nullptr) {
      LogDebug("TimingSim") << "TrackInformation for ID = " << theTrack->GetTrackID();
      info->Print();
    }
#endif
  }

  edeposit *= invgev;
  if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
    edepositEM = edeposit;
    edepositHAD = 0.f;
  } else {
    edepositEM = 0.f;
    edepositHAD = edeposit;
  }
  // time slice is defined for the entry point
  tSlice = timeFactor * preStepPoint->GetGlobalTime() * invns;
  tSliceID = (int)tSlice;

  unitID = setDetUnitId(aStep);
  primaryID = theTrack->GetTrackID();
}

bool TimingSD::hitExists(const G4Step* aStep) {
  if (!currentHit) {
    return false;
  }

  // Update if in the same detector and time-slice
  if (tSliceID == tsID && unitID == previousUnitID) {
    updateHit();
    return true;
  }

  //look in the HitContainer whether a hit with the same primID, unitID,
  //tSliceID already exists:

  bool found = false;
  int thehc_entries = theHC->entries();
  for (int j = 0; j < thehc_entries; ++j) {
    BscG4Hit* aHit = (*theHC)[j];
    if (aHit->getTimeSliceID() == tSliceID && aHit->getUnitID() == unitID) {
      if (checkHit(aStep, aHit)) {
        currentHit = aHit;
        found = true;
        break;
      }
    }
  }
  if (found) {
    updateHit();
  }
  return found;
}

bool TimingSD::checkHit(const G4Step*, BscG4Hit* hit) {
  // change hit info to fastest primary particle
  if (tof < hit->getTof()) {
    hit->setTrackID(primaryID);
    hit->setIncidentEnergy((float)incidentEnergy);
    hit->setPabs(float(preStepPoint->GetMomentum().mag()) * invgev);
    hit->setTof(tof);
    hit->setParticleType(theTrack->GetDefinition()->GetPDGEncoding());

    float ThetaAtEntry = (float)(hitPointLocal.theta() * invdeg);
    float PhiAtEntry = (float)(hitPointLocal.phi() * invdeg);

    hit->setThetaAtEntry(ThetaAtEntry);
    hit->setPhiAtEntry(PhiAtEntry);

    hit->setEntry(hitPoint);
    hit->setEntryLocalP(hitPointLocal);
    hit->setExitLocalP(hitPointLocalExit);

    hit->setParentId(theTrack->GetParentID());
    hit->setProcessId(theEnumerator->processId(theTrack->GetCreatorProcess()));

    hit->setVertexPosition(theTrack->GetVertexPosition());
  }
  return true;
}

void TimingSD::storeHit(BscG4Hit* hit) {
  if (primID < 0)
    return;
  if (hit == nullptr) {
    edm::LogWarning("BscSim") << "BscSD: hit to be stored is NULL !!";
    return;
  }

  theHC->insert(hit);
}

void TimingSD::createNewHit(const G4Step* aStep) {
#ifdef EDM_ML_DEBUG
  const G4VPhysicalVolume* currentPV = preStepPoint->GetPhysicalVolume();
  edm::LogVerbatim("TimingSim") << "TimingSD CreateNewHit for " << GetName() << " PV " << currentPV->GetName()
                                << " PVid = " << currentPV->GetCopyNo() << " Unit " << unitID << "\n primary "
                                << primaryID << " Tof(ns)= " << tof << " time slice " << tSliceID
                                << " E(MeV)= " << incidentEnergy << " trackID " << theTrack->GetTrackID() << " "
                                << theTrack->GetDefinition()->GetParticleName() << " parentID "
                                << theTrack->GetParentID();

  if (theTrack->GetCreatorProcess() != nullptr) {
    edm::LogVerbatim("TimingSim") << theTrack->GetCreatorProcess()->GetProcessName();
  } else {
    edm::LogVerbatim("TimingSim") << " is primary particle";
  }
#endif

  currentHit = new BscG4Hit;
  currentHit->setTrackID(primaryID);
  currentHit->setTimeSlice(tSlice);
  currentHit->setUnitID(unitID);
  currentHit->setIncidentEnergy((float)incidentEnergy);

  currentHit->setPabs(float(preStepPoint->GetMomentum().mag() * invgev));
  currentHit->setTof(tof);
  currentHit->setParticleType(theTrack->GetDefinition()->GetPDGEncoding());

  float ThetaAtEntry = hitPointLocal.theta() * invdeg;
  float PhiAtEntry = hitPointLocal.phi() * invdeg;

  currentHit->setThetaAtEntry(ThetaAtEntry);
  currentHit->setPhiAtEntry(PhiAtEntry);

  currentHit->setEntry(hitPoint);
  currentHit->setEntryLocalP(hitPointLocal);
  currentHit->setExitLocalP(hitPointLocalExit);

  currentHit->setParentId(theTrack->GetParentID());
  currentHit->setProcessId(theEnumerator->processId(theTrack->GetCreatorProcess()));

  currentHit->setVertexPosition(theTrack->GetVertexPosition());

  updateHit();
  storeHit(currentHit);
}

void TimingSD::updateHit() {
  currentHit->addEnergyDeposit(edepositEM, edepositHAD);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimingSim") << "updateHit: " << GetName() << " add eloss(GeV) " << edeposit
                                << "CurrentHit=" << currentHit << ", PostStepPoint= " << postStepPoint->GetPosition();
#endif

  // buffer for next steps:
  tsID = tSliceID;
  primID = primaryID;
  previousUnitID = unitID;
}

void TimingSD::setToLocal(const G4StepPoint* stepPoint, const G4ThreeVector& globalPoint, G4ThreeVector& localPoint) {
  const G4VTouchable* touch = stepPoint->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(globalPoint);
}

void TimingSD::EndOfEvent(G4HCofThisEvent*) {
  int nhits = theHC->entries();
  if (0 == nhits) {
    return;
  }
  // here we loop over transient hits and make them persistent
  for (int j = 0; j < nhits; ++j) {
    BscG4Hit* aHit = (*theHC)[j];
    Local3DPoint locEntryPoint = ConvertToLocal3DPoint(aHit->getEntryLocalP());
    Local3DPoint locExitPoint = ConvertToLocal3DPoint(aHit->getExitLocalP());

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("TimingSim") << "TimingSD: Hit for storage \n"
                                  << *aHit << "\n Entry point: " << locEntryPoint << "\n Exit  point: " << locExitPoint;
#endif

    slave->processHits(PSimHit(locEntryPoint,
                               locExitPoint,
                               aHit->getPabs(),
                               aHit->getTof(),
                               aHit->getEnergyLoss(),
                               aHit->getParticleType(),
                               aHit->getUnitID(),
                               aHit->getTrackID(),
                               aHit->getThetaAtEntry(),
                               aHit->getPhiAtEntry(),
                               aHit->getProcessId()));
  }
}

void TimingSD::PrintAll() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimingSim") << "TimingSD: Collection " << theHC->GetName();
#endif
  theHC->PrintAllHits();
}

void TimingSD::fillHits(edm::PSimHitContainer& cc, const std::string& hname) {
  if (slave->name() == hname) {
    cc = slave->hits();
  }
}

void TimingSD::update(const BeginOfEvent* i) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimingSim") << " Dispatched BeginOfEvent for " << GetName();
#endif
  clearHits();
}

void TimingSD::clearHits() { slave->Initialize(); }
