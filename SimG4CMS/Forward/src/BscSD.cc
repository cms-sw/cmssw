///////////////////////////////////////////////////////////////////////////////
// File: BscSD.cc
// Date: 02.2006
// Description: Sensitive Detector class for Bsc
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
 
//#include "Geometry/Vector/interface/LocalPoint.h"
//#include "Geometry/Vector/interface/LocalVector.h"

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

#include <string>
#include <vector>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define debug
//-------------------------------------------------------------------
BscSD::BscSD(std::string name, const DDCompactView & cpv,
	     SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, const SimTrackManager* manager) :
  SensitiveTkDetector(name, cpv, clg, p), numberingScheme(0), name(name),
  hcID(-1), theHC(0), theManager(manager), currentHit(0), theTrack(0), 
  currentPV(0), unitID(0),  previousUnitID(0), preStepPoint(0), 
  postStepPoint(0), eventno(0){
    
  //Add Bsc Sentitive Detector Name
  collectionName.insert(name);
    
    
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("BscSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
  //int verbn = 1;
    
  SetVerboseLevel(verbn);
  LogDebug("BscSim") 
    << "*******************************************************\n"
    << "*                                                     *\n"
    << "* Constructing a BscSD  with name " << name << "\n"
    << "*                                                     *\n"
    << "*******************************************************";
    
    
  slave  = new TrackingSlaveSD(name);
    
  //
  // attach detectors (LogicalVolumes)
  //
  std::vector<std::string> lvNames = clg.logicalNames(name);

  this->Register();

  for (std::vector<std::string>::iterator it=lvNames.begin();  
       it !=lvNames.end(); it++) {
    this->AssignSD(*it);
    edm::LogInfo("BscSim") << "BscSD : Assigns SD to LV " << (*it);
  }
    
  if      (name == "BSCHits") {
    if (verbn > 0) {
      edm::LogInfo("BscSim") << "name = BSCHits and  new BscNumberingSchem";
    }
    numberingScheme = new BscNumberingScheme() ;
  } else {
    edm::LogWarning("BscSim") << "BscSD: ReadoutName "<<name<<" not supported";
  }
  
  edm::LogInfo("BscSim") << "BscSD: Instantiation completed";
}




BscSD::~BscSD() { 
  //AZ:
  if (slave) delete slave; 

  if (numberingScheme)
    delete numberingScheme;

}

double BscSD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void BscSD::Initialize(G4HCofThisEvent * HCE) { 
#ifdef debug
  LogDebug("BscSim") << "BscSD : Initialize called for " << name << std::endl;
#endif

  theHC = new BscG4HitCollection(name, collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID   = -2;
  primID = -2;

  ////    slave->Initialize();
}


bool BscSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  if (aStep == NULL) {
    return true;
  } else {
    GetStepInfo(aStep);
    //   LogDebug("BscSim") << edeposit <<std::endl;

    //AZ
#ifdef debug
    LogDebug("BscSim") << "BscSD :  number of hits = " << theHC->entries() << std::endl;
#endif

    if (HitExists() == false && edeposit>0. ){ 
      CreateNewHit();
      return true;
    }
  }
  return true;
} 

void BscSD::GetStepInfo(G4Step* aStep) {
  
  preStepPoint = aStep->GetPreStepPoint(); 
  postStepPoint= aStep->GetPostStepPoint(); 
  theTrack     = aStep->GetTrack();   
  hitPoint     = preStepPoint->GetPosition();	
  currentPV    = preStepPoint->GetPhysicalVolume();
  hitPointExit = postStepPoint->GetPosition();	

  hitPointLocal = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  hitPointLocalExit = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPointExit);


  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  LogDebug("BscSim") <<  "  BscSD :particleType =  " << theTrack->GetDefinition()->GetParticleName() <<std::endl;
  if (particleCode == emPDG ||
      particleCode == epPDG ||
      particleCode == gammaPDG ) {
    edepositEM  = getEnergyDeposit(aStep); edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = getEnergyDeposit(aStep);
  }
  edeposit = aStep->GetTotalEnergyDeposit();
  tSlice    = (postStepPoint->GetGlobalTime() )/nanosecond;
  tSliceID  = (int) tSlice;
  unitID    = setDetUnitId(aStep);
#ifdef debug
  LogDebug("BscSim") << "unitID=" << unitID <<std::endl;
#endif
  primaryID    = theTrack->GetTrackID();
  //  Position     = hitPoint;
  Pabs         = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
  Tof          = aStep->GetPostStepPoint()->GetGlobalTime()/nanosecond;
  Eloss        = aStep->GetTotalEnergyDeposit()/GeV;
  ParticleType = theTrack->GetDefinition()->GetPDGEncoding();      
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

uint32_t BscSD::setDetUnitId(G4Step * aStep) { 

  return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(aStep));
}


G4bool BscSD::HitExists() {
  if (primaryID<1) {
    edm::LogWarning("BscSim") << "***** BscSD error: primaryID = " 
				  << primaryID
				  << " maybe detector name changed";
  }

  // Update if in the same detector, time-slice and for same track   
  //  if (primaryID == primID && tSliceID == tsID && unitID==previousUnitID) {
  if (tSliceID == tsID && unitID==previousUnitID) {
    //AZ:
    UpdateHit();
    return true;
  }
  // Reset entry point for new primary
  if (primaryID != primID)
    ResetForNewPrimary();
   
  //look in the HitContainer whether a hit with the same primID, unitID,
  //tSliceID already exists:
   
  G4bool found = false;

  //    LogDebug("BscSim") << "BscSD: HCollection=  " << theHC->entries()    <<std::endl;
  
  for (int j=0; j<theHC->entries()&&!found; j++) {
    BscG4Hit* aPreviousHit = (*theHC)[j];
    if (aPreviousHit->getTrackID()     == primaryID &&
	aPreviousHit->getTimeSliceID() == tSliceID  &&
	aPreviousHit->getUnitID()      == unitID       ) {
      //AZ:
      currentHit = aPreviousHit;
      found      = true;
    }
  }          

  if (found) {
    //AZ:
    UpdateHit();
    return true;
  } else {
    return false;
  }    
}


void BscSD::ResetForNewPrimary() {
  
  entrancePoint  = SetToLocal(hitPoint);
  exitPoint      = SetToLocalExit(hitPointExit);
  incidentEnergy = preStepPoint->GetKineticEnergy();

}


void BscSD::StoreHit(BscG4Hit* hit){

  if (primID<0) return;
  if (hit == 0 ) {
    edm::LogWarning("BscSim") << "BscSD: hit to be stored is NULL !!";
    return;
  }

  theHC->insert( hit );
}


void BscSD::CreateNewHit() {

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
  if (theTrack->GetCreatorProcess()!=NULL)
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
  currentHit->setEnergyLoss(Eloss);
  currentHit->setParticleType(ParticleType);
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

  UpdateHit();
  
  StoreHit(currentHit);
}	 
 

void BscSD::UpdateHit() {

  if (Eloss > 0.) {
    currentHit->addEnergyDeposit(edepositEM,edepositHAD);

#ifdef debug
    LogDebug("BscSim") << "updateHit: add eloss " << Eloss <<std::endl;
    LogDebug("BscSim") << "CurrentHit=" << currentHit
		       << ", PostStepPoint=" << postStepPoint->GetPosition();
#endif
    //AZ
    currentHit->setEnergyLoss(Eloss);
  }  

  // buffer for next steps:
  tsID           = tSliceID;
  primID         = primaryID;
  previousUnitID = unitID;
}


G4ThreeVector BscSD::SetToLocal(G4ThreeVector global){

  const G4VTouchable* touch= preStepPoint->GetTouchable();
  theEntryPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return theEntryPoint;  
}
     

G4ThreeVector BscSD::SetToLocalExit(G4ThreeVector globalPoint){

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
  Summarize();
}
     

void BscSD::Summarize() {
}


void BscSD::clear() {
} 


void BscSD::DrawAll() {
} 


void BscSD::PrintAll() {
  LogDebug("BscSim") << "BscSD: Collection " << theHC->GetName() << "\n";
  theHC->PrintAllHits();
} 


void BscSD::fillHits(edm::PSimHitContainer& c, std::string n) {
  if (slave->name() == n) c=slave->hits();
}

void BscSD::update (const BeginOfEvent * i) {
  LogDebug("BscSim") << " Dispatched BeginOfEvent for " << GetName()
                       << " !" ;
   clearHits();
   eventno = (*i)()->GetEventID();
}

void BscSD::update(const BeginOfRun *) {

  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  emPDG = theParticleTable->FindParticle(particleName="e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle(particleName="e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle(particleName="gamma")->GetPDGEncoding();

} 

void BscSD::update (const ::EndOfEvent*) {
}

void BscSD::clearHits(){
  //AZ:
  slave->Initialize();
}

std::vector<std::string> BscSD::getNames(){
  std::vector<std::string> temp;
  temp.push_back(slave->name());
  return temp;
}

