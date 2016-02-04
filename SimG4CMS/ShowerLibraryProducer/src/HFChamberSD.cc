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

HFChamberSD::HFChamberSD(std::string name, const DDCompactView & cpv,
		 SensitiveDetectorCatalog & clg, edm::ParameterSet const & p,
		 const SimTrackManager* manager) :
  SensitiveCaloDetector(name, cpv, clg, p), theName(name),
  m_trackManager(manager), theHCID(-1), theHC(0), theNSteps(0) {

  collectionName.insert(name);
  LogDebug("FiberSim") << "***************************************************"
		       << "\n"
		       << "*                                                 *"
		       << "\n"
		       << "* Constructing a HFChamberSD  with name " << GetName()
		       << "\n"
		       << "*                                                 *"
		       << "\n"
		       << "***************************************************";
  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  std::vector<std::string> lvNames = clg.logicalNames(name);
  this->Register();
  for (std::vector<std::string>::iterator it=lvNames.begin();
       it !=lvNames.end(); it++){
    this->AssignSD(*it);
    LogDebug("FiberSim") << "HFChamberSD : Assigns SD to LV " << (*it);
  }
}

HFChamberSD::~HFChamberSD() {
  if (theHC)    delete theHC;
}

void HFChamberSD::Initialize(G4HCofThisEvent * HCE) {

  LogDebug("FiberSim") << "HFChamberSD : Initialize called for " << GetName();
  theHC = new HFShowerG4HitsCollection(GetName(), collectionName[0]);
  if (theHCID<0)
    theHCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(theHCID, theHC);

}

G4bool HFChamberSD::ProcessHits(G4Step * aStep, G4TouchableHistory*) {

  //do not process hits other than primary particle hits:
  double charge = aStep->GetTrack()->GetDefinition()->GetPDGCharge();
  int trackID = aStep->GetTrack()->GetTrackID();
  if(charge == 0. || trackID != 1 ||aStep->GetTrack()->GetParentID() != 0 || aStep->GetTrack()->GetCreatorProcess() != NULL) return false;
  ++theNSteps;
  //if(theNSteps>1)return false;

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  int detID   = setDetUnitId(aStep);

  double edep = aStep->GetTotalEnergyDeposit();
  double time = (preStepPoint->GetGlobalTime())/ns;

  G4ThreeVector globalPos = preStepPoint->GetPosition();
  G4ThreeVector localPos  = touch->GetHistory()->GetTopTransform().TransformPoint(globalPos);
  const G4DynamicParticle* particle =  aStep->GetTrack()->GetDynamicParticle();
  G4ThreeVector momDir   = particle->GetMomentumDirection();

  HFShowerG4Hit *aHit = new HFShowerG4Hit(detID, trackID, edep, time);
  aHit->setLocalPos(localPos);
  aHit->setGlobalPos(globalPos);
  aHit->setPrimMomDir(momDir);

  LogDebug("FiberSim") << "HFChamberSD: Hit created in ("
		       << touch->GetVolume(0)->GetLogicalVolume()->GetName()
		       << ") " << " ID " << detID << " Track " << trackID
		       << " Edep: " << edep/MeV << " MeV; Time: " << time
		       << " ns; Position (local) " << localPos << " (global ) "
		       << globalPos << " direction " << momDir;

  theHC->insert(aHit);
  return true;
}

void HFChamberSD::EndOfEvent(G4HCofThisEvent * HCE) {

  LogDebug("FiberSim") << "HFChamberSD: Sees" << theHC->entries() << " hits";
  clear();
}

void HFChamberSD::clear() {
  theNSteps = 0;
}

void HFChamberSD::DrawAll()  {}

void HFChamberSD::PrintAll() {}

void HFChamberSD::clearHits() {}

uint32_t HFChamberSD::setDetUnitId(G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  return (touch->GetReplicaNumber(0));
}

void HFChamberSD::fillHits(edm::PCaloHitContainer&, std::string) {}
