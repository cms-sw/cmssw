#include "SimG4CMS/ShowerLibraryProducer/interface/HFWedgeSD.h"
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

HFWedgeSD::HFWedgeSD(std::string name, const DDCompactView & cpv, 
		 SensitiveDetectorCatalog & clg, edm::ParameterSet const & p, 
		 const SimTrackManager* manager) :
  SensitiveCaloDetector(name, cpv, clg, p), theName(name),
  m_trackManager(manager), hcID(-1), theHC(0), currentHit(0) {

  collectionName.insert(name);
  LogDebug("FiberSim") << "***************************************************"
		       << "\n"
		       << "*                                                 *"
		       << "\n"
		       << "* Constructing a HFWedgeSD  with name " << GetName()
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
    LogDebug("FiberSim") << "HFWedgeSD : Assigns SD to LV " << (*it);
  }
}

HFWedgeSD::~HFWedgeSD() {
  if (theHC)    delete theHC;
}

void HFWedgeSD::Initialize(G4HCofThisEvent * HCE) {

  LogDebug("FiberSim") << "HFWedgeSD : Initialize called for " << GetName();
  theHC = new HFShowerG4HitsCollection(GetName(), collectionName[0]);
  if (hcID<0)
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  clearHits();
}

G4bool HFWedgeSD::ProcessHits(G4Step * aStep, G4TouchableHistory*) {

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  currentID = setDetUnitId(aStep);
  trackID   = aStep->GetTrack()->GetTrackID();
  edep      = aStep->GetTotalEnergyDeposit();
  time      = (preStepPoint->GetGlobalTime())/ns;

  globalPos = preStepPoint->GetPosition();
  localPos  = touch->GetHistory()->GetTopTransform().TransformPoint(globalPos);
  const G4DynamicParticle* particle =  aStep->GetTrack()->GetDynamicParticle();
  momDir    = particle->GetMomentumDirection();

  if (hitExists() == false && edep>0.) 
    currentHit = createNewHit();

  return true;
}

void HFWedgeSD::EndOfEvent(G4HCofThisEvent * HCE) {
 
  LogDebug("FiberSim") << "HFWedgeSD: Sees" << theHC->entries() << " hits";
  clear();
}

void HFWedgeSD::clear() {}

void HFWedgeSD::DrawAll()  {}

void HFWedgeSD::PrintAll() {}

G4bool HFWedgeSD::hitExists() {
   
  // Update if in the same detector, time-slice and for same track   
  if (currentID == previousID) {
    updateHit(currentHit);
    return true;
  }
   
  std::map<int,HFShowerG4Hit*>::const_iterator it = hitMap.find(currentID);
  if (it != hitMap.end()) {
    updateHit(currentHit);
    return true;
  }
  
  return false;
}

HFShowerG4Hit* HFWedgeSD::createNewHit() {

  LogDebug("FiberSim") << "HFWedgeSD::CreateNewHit for ID " << currentID
		       << " Track " << trackID << " Edep: " << edep/MeV 
		       << " MeV; Time: " << time << " ns; Position (local) " 
		       << localPos << " (global ) " << globalPos 
		       << " direction " << momDir;
  HFShowerG4Hit* aHit = new HFShowerG4Hit;
  aHit->setHitId(currentID);
  aHit->setTrackId(trackID);
  aHit->setTime(time);
  aHit->setLocalPos(localPos);
  aHit->setGlobalPos(globalPos);
  aHit->setPrimMomDir(momDir);
  updateHit(aHit);

  theHC->insert(aHit);
  hitMap.insert(std::pair<int,HFShowerG4Hit*>(previousID,aHit));

  return aHit;
}	 

void HFWedgeSD::updateHit(HFShowerG4Hit* aHit) {

  if (edep != 0) {
    aHit->updateEnergy(edep);
    LogDebug("FiberSim") << "HFWedgeSD: Add energy deposit in " << currentID 
			 << " edep " << edep/MeV << " MeV"; 
  }
  previousID = currentID;
}

void HFWedgeSD::clearHits() {
  hitMap.erase (hitMap.begin(), hitMap.end());
  previousID = -1;
}

uint32_t HFWedgeSD::setDetUnitId(G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  return (touch->GetReplicaNumber(0));
}

void HFWedgeSD::fillHits(edm::PCaloHitContainer&, std::string) {}
