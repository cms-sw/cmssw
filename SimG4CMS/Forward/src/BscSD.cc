///////////////////////////////////////////////////////////////////////////////
// File: BscSD.cc
// Date: 02.2006
// Description: Sensitive Detector class for Bsc
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
 
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"


#include "SimG4CMS/Forward/interface/BscSD.h"
#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"
#include "SimG4CMS/Forward/interface/BscNumberingScheme.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4Step.hh"
#include "G4ParticleTable.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <string>
#include <vector>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define debug
//-------------------------------------------------------------------
BscSD::BscSD(const std::string& name, const DDCompactView & cpv,
	     const SensitiveDetectorCatalog & clg,
	     edm::ParameterSet const & p, const SimTrackManager* manager) :
  SensitiveTkDetector(name, cpv, clg, p), numberingScheme(nullptr),
  hcID(-1), theHC(nullptr), theManager(manager), currentHit(nullptr), theTrack(nullptr), 
  currentPV(nullptr), unitID(0),  previousUnitID(0), preStepPoint(nullptr), 
  postStepPoint(nullptr){
    
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("BscSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
    
  SetVerboseLevel(verbn);
    
  slave  = new TrackingSlaveSD(name);
        
  if(name == "BSCHits") {
    if (verbn > 0) {
      edm::LogInfo("BscSim") << "name = BSCHits and  new BscNumberingSchem";
    }
    numberingScheme = new BscNumberingScheme() ;
  } else {
    edm::LogWarning("BscSim") << "BscSD: ReadoutName "<<name<<" not supported";
  }
}

BscSD::~BscSD() { 
  delete slave; 
  delete numberingScheme;
}

void BscSD::Initialize(G4HCofThisEvent * HCE) { 
#ifdef debug
  LogDebug("BscSim") << "BscSD : Initialize called for " << GetName();
#endif

  theHC = new BscG4HitCollection(GetName(), collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID   = -2;
  primID = -2;
}

bool BscSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  edeposit = aStep->GetTotalEnergyDeposit();
  if (edeposit>0.f){ 
    getStepInfo(aStep);
    LogDebug("BscSim") << "BscSD :  number of hits = " << theHC->entries();
    if (!hitExists()){ 
      createNewHit();
    }
  }
  return true;
} 

void BscSD::getStepInfo(const G4Step* aStep) {
  
  preStepPoint = aStep->GetPreStepPoint(); 
  postStepPoint= aStep->GetPostStepPoint(); 
  theTrack     = aStep->GetTrack();   
  hitPoint     = preStepPoint->GetPosition();	
  currentPV    = preStepPoint->GetPhysicalVolume();
  hitPointExit = postStepPoint->GetPosition();	

  hitPointLocal = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  hitPointLocalExit = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPointExit);

  particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  LogDebug("BscSim") << "BscSD:particleType =  " << theTrack->GetDefinition()->GetParticleName();
  edeposit /= GeV;
  if ( G4TrackToParticleID::isGammaElectronPositron(theTrack) ) {
    edepositEM  = edeposit; edepositHAD = 0.f;
  } else {
    edepositEM  = 0.f; edepositHAD = edeposit;
  }
  tSlice    = (postStepPoint->GetGlobalTime() )/nanosecond;
  tSliceID  = (int) tSlice;
  unitID    = setDetUnitId(aStep);
#ifdef debug
  LogDebug("BscSim") << "unitID=" << unitID;
#endif
  primaryID    = theTrack->GetTrackID();
  //  Position     = hitPoint;
  Pabs         = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
  Tof          = aStep->GetPostStepPoint()->GetGlobalTime()/nanosecond;
  ThetaAtEntry = aStep->GetPreStepPoint()->GetPosition().theta()/deg;
  PhiAtEntry   = aStep->GetPreStepPoint()->GetPosition().phi()/deg;

  ParentId = theTrack->GetParentID();
  Vx = theTrack->GetVertexPosition().x();
  Vy = theTrack->GetVertexPosition().y();
  Vz = theTrack->GetVertexPosition().z();
  X  = hitPoint.x();
  Y  = hitPoint.y();
  Z  = hitPoint.z();
}

uint32_t BscSD::setDetUnitId(const G4Step * aStep) { 
  return (numberingScheme == nullptr ? 0 : numberingScheme->getUnitID(aStep));
}

bool BscSD::hitExists() {
  if (primaryID<1) {
    edm::LogWarning("BscSim") << "***** BscSD error: primaryID = " 
				  << primaryID
				  << " maybe detector name changed";
  }

  // Update if in the same detector, time-slice and for same track   
  //  if (primaryID == primID && tSliceID == tsID && unitID==previousUnitID) {
  if (tSliceID == tsID && unitID==previousUnitID) {
    //AZ:
    updateHit();
    return true;
  }
  // Reset entry point for new primary
  if (primaryID != primID)
    resetForNewPrimary();
   
  //look in the HitContainer whether a hit with the same primID, unitID,
  //tSliceID already exists:
   
  bool found = false;

  //LogDebug("BscSim") << "BscSD: HCollection=  " << theHC->entries();
  
  for (int j=0; j<theHC->entries()&&!found; j++) {
    BscG4Hit* aPreviousHit = (*theHC)[j];
    if (aPreviousHit->getTrackID()     == primaryID &&
	aPreviousHit->getTimeSliceID() == tSliceID  &&
	aPreviousHit->getUnitID()      == unitID       ) {
      currentHit = aPreviousHit;
      found      = true;
      break;
    }
  }          

  if (found) { updateHit(); }
  return found;
}

void BscSD::resetForNewPrimary() {
  
  entrancePoint  = setToLocal(hitPoint);
  exitPoint      = setToLocalExit(hitPointExit);
  incidentEnergy = preStepPoint->GetKineticEnergy();

}

void BscSD::storeHit(BscG4Hit* hit){

  if (primID<0) return;
  if (hit == nullptr ) {
    edm::LogWarning("BscSim") << "BscSD: hit to be stored is NULL !!";
    return;
  }

  theHC->insert( hit );
}

void BscSD::createNewHit() {

#ifdef debug
  LogDebug("BscSim") << "BscSD CreateNewHit for"
		     << " PV "     << currentPV->GetName()
		     << " PVid = " << currentPV->GetCopyNo()
		     << " Unit "   << unitID <<std::endl;
  LogDebug("BscSim") << " primary "    << primaryID
		     << " time slice " << tSliceID 
		     << " For Track  " << theTrack->GetTrackID()
		     << " which is a " << theTrack->GetDefinition()->GetParticleName();
	   
  if (theTrack->GetTrackID()==1) {
    LogDebug("BscSim") << " of energy "     << theTrack->GetTotalEnergy();
  } else {
    LogDebug("BscSim") << " daughter of part. " << theTrack->GetParentID();
  }

  LogDebug("BscSim")  << " and created by " ;
  if (theTrack->GetCreatorProcess()!=nullptr)
    LogDebug("BscSim") << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    LogDebug("BscSim") << "NO process";
  LogDebug("BscSim") << std::endl;
#endif          
    
  currentHit = new BscG4Hit;
  currentHit->setTrackID(primaryID);
  currentHit->setTimeSlice(tSlice);
  currentHit->setUnitID(unitID);
  currentHit->setIncidentEnergy(incidentEnergy);

  currentHit->setPabs(Pabs);
  currentHit->setTof(Tof);
  currentHit->setEnergyLoss(edeposit);
  currentHit->setParticleType(particleCode);
  currentHit->setThetaAtEntry(ThetaAtEntry);
  currentHit->setPhiAtEntry(PhiAtEntry);

  currentHit->setEntry(hitPoint);

  currentHit->setEntryLocalP(hitPointLocal);
  currentHit->setExitLocalP(hitPointLocalExit);

  currentHit->setParentId(ParentId);
  currentHit->setVx(Vx);
  currentHit->setVy(Vy);
  currentHit->setVz(Vz);

  currentHit->setX(X);
  currentHit->setY(Y);
  currentHit->setZ(Z);

  updateHit();
  
  storeHit(currentHit);
}	 
 
void BscSD::updateHit() {

  currentHit->addEnergyDeposit(edepositEM,edepositHAD);

#ifdef debug
  LogDebug("BscSim") << "updateHit: add eloss " << edeposit 
		     << "CurrentHit=" << currentHit
		     << ", PostStepPoint=" << postStepPoint->GetPosition();
#endif

  // buffer for next steps:
  tsID           = tSliceID;
  primID         = primaryID;
  previousUnitID = unitID;
}

G4ThreeVector BscSD::setToLocal(const G4ThreeVector& global){

  const G4VTouchable* touch= preStepPoint->GetTouchable();
  theEntryPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return theEntryPoint;  
}
     
G4ThreeVector BscSD::setToLocalExit(const G4ThreeVector& globalPoint){

  const G4VTouchable* touch= postStepPoint->GetTouchable();
  theExitPoint = touch->GetHistory()->GetTopTransform().TransformPoint(globalPoint);
  return theExitPoint;  
}
     

void BscSD::EndOfEvent(G4HCofThisEvent* ) {

  // here we loop over transient hits and make them persistent
  for (int j=0; j<theHC->entries(); j++) {
    //AZ:
    BscG4Hit* aHit = (*theHC)[j];
    LogDebug("BscSim") << "hit number" << j << "unit ID = "<<aHit->getUnitID()<< "\n";
    LogDebug("BscSim") << "entry z " << aHit->getEntry().z()<< "\n";
    LogDebug("BscSim") << "entr theta " << aHit->getThetaAtEntry()<< "\n";

    Local3DPoint locExitPoint(0,0,0);
    Local3DPoint locEntryPoint(aHit->getEntry().x(),
			       aHit->getEntry().y(),
			       aHit->getEntry().z());
    slave->processHits(PSimHit(locEntryPoint,locExitPoint,
			       aHit->getPabs(),
			       aHit->getTof(),
			       aHit->getEnergyLoss(),
			       aHit->getParticleType(),
			       aHit->getUnitID(),
			       aHit->getTrackID(),
			       aHit->getThetaAtEntry(),
			       aHit->getPhiAtEntry()));
  }
}
     
void BscSD::PrintAll() {
  LogDebug("BscSim") << "BscSD: Collection " << theHC->GetName() << "\n";
  theHC->PrintAllHits();
} 

void BscSD::fillHits(edm::PSimHitContainer& cc, const std::string& hname) {
  if (slave->name() == hname) { cc=slave->hits(); }
}

void BscSD::update (const BeginOfEvent * i) {
  LogDebug("BscSim") << " Dispatched BeginOfEvent for " << GetName();
  clearHits();
}

void BscSD::clearHits(){
  slave->Initialize();
}
