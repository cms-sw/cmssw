#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimG4CMS/Forward/interface/FastTimerSD.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

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

//#define DebugLog
//-------------------------------------------------------------------
FastTimerSD::FastTimerSD(std::string name, const DDCompactView & cpv,
			 const SensitiveDetectorCatalog & clg, 
			 edm::ParameterSet const & p, 
			 const SimTrackManager* manager) :
  SensitiveTkDetector(name, cpv, clg, p), ftcons(0), name(name),
  hcID(-1), theHC(0), theManager(manager), currentHit(0), theTrack(0), 
  currentPV(0), unitID(0),  previousUnitID(0), preStepPoint(0), 
  postStepPoint(0), eventno(0){
    
  //Add FastTimer Sentitive Detector Name
  collectionName.insert(name);
    
    
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("FastTimerSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
  //int verbn = 1;
    
  SetVerboseLevel(verbn);
#ifdef DebugLog
  LogDebug("FastTimerSim") 
    << "*******************************************************\n"
    << "*                                                     *\n"
    << "* Constructing a FastTimerSD  with name " << name << "\n"
    << "*                                                     *\n"
    << "*******************************************************";
#endif    
    
  slave  = new TrackingSlaveSD(name);
    
  //
  // attach detectors (LogicalVolumes)
  //
  std::vector<std::string> lvNames = clg.logicalNames(name);

  this->Register();

  for (std::vector<std::string>::iterator it=lvNames.begin();  
       it !=lvNames.end(); it++) {
    this->AssignSD(*it);
    edm::LogInfo("FastTimerSim") << "FastTimerSD : Assigns SD to LV " << (*it);
  }
    
  ftcons = new FastTimeDDDConstants(cpv) ;
  type   = ftcons->getType();
  
  edm::LogInfo("FastTimerSim") << "FastTimerSD: Instantiation completed";
}


FastTimerSD::~FastTimerSD() { 

  if (slave)  delete slave; 
  if (ftcons) delete ftcons;
}

double FastTimerSD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void FastTimerSD::Initialize(G4HCofThisEvent * HCE) { 
#ifdef DebugLog
  LogDebug("FastTimerSim") << "FastTimerSD : Initialize called for " << name;
#endif

  theHC = new BscG4HitCollection(name, collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID   = -2;
  primID = -2;
}


bool FastTimerSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  if (aStep == NULL) {
    return true;
  } else {
    GetStepInfo(aStep);
#ifdef DebugLog
    LogDebug("FastTimerSim") << "FastTimerSD :  number of hits = " 
			     << theHC->entries();
#endif
    if (HitExists() == false && edeposit>0. ){ 
      CreateNewHit();
      return true;
    }
  }
  return true;
} 

void FastTimerSD::GetStepInfo(G4Step* aStep) {
  
  preStepPoint = aStep->GetPreStepPoint(); 
  postStepPoint= aStep->GetPostStepPoint(); 
  theTrack     = aStep->GetTrack();   
  hitPoint     = preStepPoint->GetPosition();	
  currentPV    = preStepPoint->GetPhysicalVolume();
  hitPointExit = postStepPoint->GetPosition();	

  hitPointLocal = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  hitPointLocalExit = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPointExit);


  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
#ifdef DebugLog
  edm::LogInfo("FastTimerSim") << "FastTimerSD :particleType =  " 
			       << theTrack->GetDefinition()->GetParticleName();
#endif
  if (particleCode == emPDG ||
      particleCode == epPDG ||
      particleCode == gammaPDG ) {
    edepositEM  = getEnergyDeposit(aStep); edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = getEnergyDeposit(aStep);
  }
  edeposit = aStep->GetTotalEnergyDeposit();
  tSlice    = (100*postStepPoint->GetGlobalTime() )/CLHEP::nanosecond;
  tSliceID  = (int) tSlice;
  unitID    = setDetUnitId(aStep);
#ifdef DebugLog
  LogDebug("FastTimerSim") << "FastTimerSD:unitID = " << std::hex << unitID
			   << std::dec;
#endif
  primaryID    = theTrack->GetTrackID();
  //  Position     = hitPoint;
  Pabs         = aStep->GetPreStepPoint()->GetMomentum().mag()/CLHEP::GeV;
  Tof          = aStep->GetPostStepPoint()->GetGlobalTime()/CLHEP::nanosecond;
  Eloss        = aStep->GetTotalEnergyDeposit()/CLHEP::GeV;
  ParticleType = theTrack->GetDefinition()->GetPDGEncoding();      
  ThetaAtEntry = aStep->GetPreStepPoint()->GetPosition().theta()/CLHEP::deg;
  PhiAtEntry   = aStep->GetPreStepPoint()->GetPosition().phi()/CLHEP::deg;

  ParentId = theTrack->GetParentID();
  Vx = theTrack->GetVertexPosition().x();
  Vy = theTrack->GetVertexPosition().y();
  Vz = theTrack->GetVertexPosition().z();
  X  = hitPoint.x();
  Y  = hitPoint.y();
  Z  = hitPoint.z();
}

uint32_t FastTimerSD::setDetUnitId(G4Step * aStep) { 

  //Find the depth segment
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  float globalZ = aStep->GetPreStepPoint()->GetPosition().z();
  int        iz = (globalZ > 0) ? 1 : -1;
  std::pair<int,int> ixy;
  if (type == 1) {
    ixy = ftcons->getXY(touch->GetReplicaNumber(0));
  } else {
    ixy = ftcons->getXY(hitPointLocal.x(),hitPointLocal.y());
  }
  uint32_t id = FastTimeDetId(ixy.first,ixy.second,iz).rawId();
#ifdef DebugLog
  edm::LogInfo("FastTimerSim") << "Levels " << touch->GetReplicaNumber(0) <<":"
			       << globalZ << " Ixyz " << ixy.first << ":" 
			       << ixy.second << ":" << iz  << " id " 
			       << std::hex << id << std::dec;
#endif
  return id;
}


G4bool FastTimerSD::HitExists() {
  if (primaryID<1) {
    edm::LogWarning("FastTimerSim") << "***** FastTimerSD error: primaryID = " 
				    << primaryID
				    << " maybe detector name changed";
  }

  // Update if in the same detector, time-slice and for same track   
  if (tSliceID == tsID && unitID==previousUnitID) {
    UpdateHit();
    return true;
  }
  // Reset entry point for new primary
  if (primaryID != primID)
    ResetForNewPrimary();
   
  //look in the HitContainer whether a hit with the same primID, unitID,
  //tSliceID already exists:
   
  G4bool found = false;

  for (int j=0; j<theHC->entries()&&!found; j++) {
    BscG4Hit* aPreviousHit = (*theHC)[j];
    if (aPreviousHit->getTrackID()     == primaryID &&
	aPreviousHit->getTimeSliceID() == tSliceID  &&
	aPreviousHit->getUnitID()      == unitID       ) {
      currentHit = aPreviousHit;
      found      = true;
    }
  }          

  if (found) {
    UpdateHit();
    return true;
  } else {
    return false;
  }    
}


void FastTimerSD::ResetForNewPrimary() {
  
  entrancePoint  = SetToLocal(hitPoint);
  exitPoint      = SetToLocalExit(hitPointExit);
  incidentEnergy = preStepPoint->GetKineticEnergy();
}


void FastTimerSD::StoreHit(BscG4Hit* hit){

  if (primID<0) return;
  if (hit == 0) {
    edm::LogWarning("FastTimerSim") << "FastTimerSD: hit to be stored is NULL !!";
  } else {
    theHC->insert( hit );
  }
}


void FastTimerSD::CreateNewHit() {

#ifdef DebugLog
  LogDebug("FastTimerSim") << "FastTimerSD CreateNewHit for"
			   << " PV "     << currentPV->GetName()
			   << " PVid = " << currentPV->GetCopyNo()
			   << " Unit "   << unitID <<std::endl;
  LogDebug("FastTimerSim") << " primary "    << primaryID
			   << " time slice " << tSliceID 
			   << " For Track  " << theTrack->GetTrackID()
			   << " which is a " << theTrack->GetDefinition()->GetParticleName();
	   
  if (theTrack->GetTrackID()==1) {
    LogDebug("FastTimerSim") << " of energy "     << theTrack->GetTotalEnergy();
  } else {
    LogDebug("FastTimerSim") << " daughter of part. " << theTrack->GetParentID();
  }

  LogDebug("FastTimerSim")  << " and created by " ;
  if (theTrack->GetCreatorProcess()!=NULL)
    LogDebug("FastTimerSim") << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    LogDebug("FastTimerSim") << "NO process";
  LogDebug("FastTimerSim") << std::endl;
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
 

void FastTimerSD::UpdateHit() {

  if (Eloss > 0.) {
    currentHit->addEnergyDeposit(edepositEM,edepositHAD);

#ifdef DebugLog
    LogDebug("FastTimerSim") << "updateHit: add eloss " << Eloss <<std::endl;
    LogDebug("FastTimerSim") << "CurrentHit="<< currentHit<< ", PostStepPoint="
			     << postStepPoint->GetPosition();
#endif
    currentHit->setEnergyLoss(Eloss);
  }  

  // buffer for next steps:
  tsID           = tSliceID;
  primID         = primaryID;
  previousUnitID = unitID;
}


G4ThreeVector FastTimerSD::SetToLocal(const G4ThreeVector& global){

  const G4VTouchable* touch= preStepPoint->GetTouchable();
  theEntryPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return theEntryPoint;  
}
     

G4ThreeVector FastTimerSD::SetToLocalExit(const G4ThreeVector& globalPoint){

  const G4VTouchable* touch= postStepPoint->GetTouchable();
  theExitPoint = touch->GetHistory()->GetTopTransform().TransformPoint(globalPoint);
  return theExitPoint;  
}
     

void FastTimerSD::EndOfEvent(G4HCofThisEvent* ) {

  // here we loop over transient hits and make them persistent
  for (int j=0; j<theHC->entries(); j++) {
    BscG4Hit* aHit = (*theHC)[j];
#ifdef DebugLog
    edm::LogInfo("FastTimerSim") << "hit number " << j << " unit ID = "
				 << std::hex << aHit->getUnitID() << std::dec
				 << " entry z " << aHit->getEntry().z()
				 << " entry theta " << aHit->getThetaAtEntry();
#endif
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
     
void FastTimerSD::Summarize() {}

void FastTimerSD::clear() {} 

void FastTimerSD::DrawAll() {} 

void FastTimerSD::PrintAll() {
#ifdef DebugLog
  edm::LogInfo("FastTimerSim") << "FastTimerSD: Collection "<<theHC->GetName();
#endif
  theHC->PrintAllHits();
} 

void FastTimerSD::fillHits(edm::PSimHitContainer& c, std::string n) {
  if (slave->name() == n) c=slave->hits();
}

void FastTimerSD::update (const BeginOfEvent * i) {
#ifdef DebugLog
  edm::LogInfo("FastTimerSim") << "Dispatched BeginOfEvent for " << GetName()
			       << " !" ;
#endif
   clearHits();
   eventno = (*i)()->GetEventID();
}

void FastTimerSD::update(const BeginOfRun *) {

  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  emPDG = theParticleTable->FindParticle(particleName="e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle(particleName="e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle(particleName="gamma")->GetPDGEncoding();

} 

void FastTimerSD::update (const ::EndOfEvent*) {}

void FastTimerSD::clearHits(){
  slave->Initialize();
}

std::vector<std::string> FastTimerSD::getNames(){
  std::vector<std::string> temp;
  temp.push_back(slave->name());
  return temp;
}
