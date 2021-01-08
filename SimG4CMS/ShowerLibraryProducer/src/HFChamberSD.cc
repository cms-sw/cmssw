#include "SimG4CMS/ShowerLibraryProducer/interface/HFChamberSD.h"
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

HFChamberSD::HFChamberSD(const std::string& name,
                         const edm::EventSetup& es,
                         const SensitiveDetectorCatalog& clg,
                         edm::ParameterSet const& p,
                         const SimTrackManager* manager)
    : SensitiveCaloDetector(name, es, clg, p), m_trackManager(manager), theHCID(-1), theHC(nullptr), theNSteps(0) {
  edm::LogVerbatim("FiberSim") << "HFChamberSD : Instantiated for " << name;
}

HFChamberSD::~HFChamberSD() { delete theHC; }

void HFChamberSD::Initialize(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "HFChamberSD : Initialize called for " << GetName() << " in collection " << HCE;
  theHC = new HFShowerG4HitsCollection(GetName(), collectionName[0]);
  if (theHCID < 0)
    theHCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(theHCID, theHC);
  edm::LogVerbatim("FiberSim") << "HFChamberSD : Add hit collectrion for " << collectionName[0] << ":" << theHCID << ":"
                               << theHC;
}

G4bool HFChamberSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  //do not process hits other than primary particle hits:
  double charge = aStep->GetTrack()->GetDefinition()->GetPDGCharge();
  int trackID = aStep->GetTrack()->GetTrackID();
  if (charge == 0. || trackID != 1 || aStep->GetTrack()->GetParentID() != 0 ||
      aStep->GetTrack()->GetCreatorProcess() != nullptr)
    return false;
  ++theNSteps;
  //if(theNSteps>1)return false;

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  int detID = setDetUnitId(aStep);

  double edep = aStep->GetTotalEnergyDeposit();
  double time = (preStepPoint->GetGlobalTime()) / ns;

  const G4ThreeVector& globalPos = preStepPoint->GetPosition();
  G4ThreeVector localPos = touch->GetHistory()->GetTopTransform().TransformPoint(globalPos);
  const G4DynamicParticle* particle = aStep->GetTrack()->GetDynamicParticle();
  const G4ThreeVector& momDir = particle->GetMomentumDirection();

  HFShowerG4Hit* aHit = new HFShowerG4Hit(detID, trackID, edep, time);
  aHit->setLocalPos(localPos);
  aHit->setGlobalPos(globalPos);
  aHit->setPrimMomDir(momDir);

  edm::LogVerbatim("FiberSim") << "HFChamberSD: Hit created in (" << touch->GetVolume(0)->GetLogicalVolume()->GetName()
                               << ")  ID " << detID << " Track " << trackID << " Edep: " << edep / CLHEP::MeV
                               << " MeV; Time: " << time << " ns; Position (local) " << localPos << " (global ) "
                               << globalPos << " direction " << momDir;

  theHC->insert(aHit);
  return true;
}

void HFChamberSD::EndOfEvent(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "HFChamberSD: Finds " << theHC->entries() << " hits";
  clear();
}

void HFChamberSD::clear() { theNSteps = 0; }

void HFChamberSD::DrawAll() {}

void HFChamberSD::PrintAll() {}

void HFChamberSD::clearHits() {}

uint32_t HFChamberSD::setDetUnitId(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  return (touch->GetReplicaNumber(0));
}

void HFChamberSD::fillHits(edm::PCaloHitContainer&, const std::string&) {}
