#include "SimG4CMS/ShowerLibraryProducer/interface/HFWedgeSensitiveDetector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4VSolid.hh"
#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4SDManager.hh"
#include "G4ios.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//#define EDM_ML_DEBUG

HFWedgeSensitiveDetector::HFWedgeSensitiveDetector(const std::string& iname,
						   const SensitiveDetectorCatalog& clg,
						   const SimTrackManager* manager)
    : SensitiveCaloDetector(iname, clg), hcID(-1), theHC(nullptr), currentHit(nullptr) {
  edm::LogVerbatim("FiberSim") << "HFWedgeSensitiveDetector : Instantiated for " << iname;
}

HFWedgeSensitiveDetector::~HFWedgeSensitiveDetector() { delete theHC; }

void HFWedgeSensitiveDetector::Initialize(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "HFWedgeSensitiveDetector : Initialize called for " << GetName() << " in collection " << HCE;
  theHC = new HFShowerG4HitsCollection(GetName(), collectionName[0]);
  if (hcID < 0)
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);
  edm::LogVerbatim("FiberSim") << "HFWedgeSensitiveDetector : Add hit collectrion for " << collectionName[0] << ":" << hcID << ":" << theHC;

  clearHits();
}

G4bool HFWedgeSensitiveDetector::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  currentID = setDetUnitId(aStep);
  trackID = aStep->GetTrack()->GetTrackID();
  edep = aStep->GetTotalEnergyDeposit();
  time = (preStepPoint->GetGlobalTime()) / ns;

  globalPos = preStepPoint->GetPosition();
  localPos = touch->GetHistory()->GetTopTransform().TransformPoint(globalPos);
  const G4DynamicParticle* particle = aStep->GetTrack()->GetDynamicParticle();
  momDir = particle->GetMomentumDirection();

  if (hitExists() == false && edep > 0.)
    currentHit = createNewHit();

  return true;
}

void HFWedgeSensitiveDetector::EndOfEvent(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "HFWedgeSensitiveDetector: Sees" << theHC->entries() << " hits";
  clear();
}

void HFWedgeSensitiveDetector::clear() {}

void HFWedgeSensitiveDetector::DrawAll() {}

void HFWedgeSensitiveDetector::PrintAll() {}

G4bool HFWedgeSensitiveDetector::hitExists() {
  // Update if in the same detector, time-slice and for same track
  if (currentID == previousID) {
    updateHit(currentHit);
    return true;
  }

  std::map<int, HFShowerG4Hit*>::const_iterator it = hitMap.find(currentID);
  if (it != hitMap.end()) {
    updateHit(currentHit);
    return true;
  }

  return false;
}

HFShowerG4Hit* HFWedgeSensitiveDetector::createNewHit() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FiberSim") << "HFWedgeSensitiveDetector::CreateNewHit for ID " << currentID << " Track " << trackID
                               << " Edep: " << edep / CLHEP::MeV << " MeV; Time: " << time << " ns; Position (local) "
                               << localPos << " (global ) " << globalPos << " direction " << momDir;
#endif
  HFShowerG4Hit* aHit = new HFShowerG4Hit;
  aHit->setHitId(currentID);
  aHit->setTrackId(trackID);
  aHit->setTime(time);
  aHit->setLocalPos(localPos);
  aHit->setGlobalPos(globalPos);
  aHit->setPrimMomDir(momDir);
  updateHit(aHit);

  theHC->insert(aHit);
  hitMap.insert(std::pair<int, HFShowerG4Hit*>(previousID, aHit));

  return aHit;
}

void HFWedgeSensitiveDetector::updateHit(HFShowerG4Hit* aHit) {
  if (edep != 0) {
    aHit->updateEnergy(edep);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FiberSim") << "HFWedgeSensitiveDetector: Add energy deposit in " << currentID << " edep " << edep / CLHEP::MeV << " MeV";
#endif
  }
  previousID = currentID;
}

void HFWedgeSensitiveDetector::clearHits() {
  hitMap.erase(hitMap.begin(), hitMap.end());
  previousID = -1;
}

uint32_t HFWedgeSensitiveDetector::setDetUnitId(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  return (touch->GetReplicaNumber(0));
}

void HFWedgeSensitiveDetector::fillHits(edm::PCaloHitContainer&, const std::string&) {}
