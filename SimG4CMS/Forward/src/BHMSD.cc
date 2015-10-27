#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimG4CMS/Forward/interface/BHMSD.h"

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
BHMSD::BHMSD(std::string name, const DDCompactView & cpv,
	     const SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, const SimTrackManager* manager) :
  SensitiveTkDetector(name, cpv, clg, p), numberingScheme(0), name(name),
  hcID(-1), theHC(0), theManager(manager), currentHit(0), theTrack(0), 
  currentPV(0), unitID(0),  previousUnitID(0), preStepPoint(0), 
  postStepPoint(0), eventno(0){
    
  //Add BHM Sentitive Detector Name
  collectionName.insert(name);
    
    
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("BHMSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
  //int verbn = 1;
    
  SetVerboseLevel(verbn);
  LogDebug("BHMSim") 
    << "*******************************************************\n"
    << "*                                                     *\n"
    << "* Constructing a BHMSD  with name " << name << "\n"
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
    edm::LogInfo("BHMSim") << "BHMSD : Assigns SD to LV " << (*it);
  }
    
  if (verbn > 0) {
    edm::LogInfo("BHMSim") << "name = " <<name <<" and new BHMNumberingScheme";
  }
  numberingScheme = new BHMNumberingScheme() ;
  
  edm::LogInfo("BHMSim") << "BHMSD: Instantiation completed";
}


BHMSD::~BHMSD() { 

  if (slave)           delete slave; 
  if (numberingScheme) delete numberingScheme;
}

double BHMSD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void BHMSD::Initialize(G4HCofThisEvent * HCE) { 
#ifdef debug
  LogDebug("BHMSim") << "BHMSD : Initialize called for " << name << std::endl;
#endif

  theHC = new BscG4HitCollection(name, collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID   = -2;
  primID = -2;
}


bool BHMSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  if (aStep == NULL) {
    return true;
  } else {
    GetStepInfo(aStep);
#ifdef debug
    LogDebug("BHMSim") << "BHMSD :  number of hits = " << theHC->entries() << std::endl;
#endif
    if (HitExists() == false && edeposit>0. ){ 
      CreateNewHit();
      return true;
    }
  }
  return true;
} 

void BHMSD::GetStepInfo(G4Step* aStep) {
  
  preStepPoint = aStep->GetPreStepPoint(); 
  postStepPoint= aStep->GetPostStepPoint(); 
  theTrack     = aStep->GetTrack();   
  hitPoint     = preStepPoint->GetPosition();	
  currentPV    = preStepPoint->GetPhysicalVolume();
  hitPointExit = postStepPoint->GetPosition();	

  hitPointLocal = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  hitPointLocalExit = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPointExit);


  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  LogDebug("BHMSim") <<  "  BHMSD :particleType =  " << theTrack->GetDefinition()->GetParticleName() <<std::endl;
  if (particleCode == emPDG ||
      particleCode == epPDG ||
      particleCode == gammaPDG ) {
    edepositEM  = getEnergyDeposit(aStep); edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = getEnergyDeposit(aStep);
  }
  edeposit = aStep->GetTotalEnergyDeposit();
  tSlice    = (postStepPoint->GetGlobalTime() )/CLHEP::nanosecond;
  tSliceID  = (int) tSlice;
  unitID    = setDetUnitId(aStep);
#ifdef debug
  LogDebug("BHMSim") << "unitID=" << unitID <<std::endl;
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

uint32_t BHMSD::setDetUnitId(G4Step * aStep) { 
  return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(aStep));
}


G4bool BHMSD::HitExists() {
  if (primaryID<1) {
    edm::LogWarning("BHMSim") << "***** BHMSD error: primaryID = " 
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


void BHMSD::ResetForNewPrimary() {
  
  entrancePoint  = SetToLocal(hitPoint);
  exitPoint      = SetToLocalExit(hitPointExit);
  incidentEnergy = preStepPoint->GetKineticEnergy();
}


void BHMSD::StoreHit(BscG4Hit* hit){

  if (primID<0) return;
  if (hit == 0 ) {
    edm::LogWarning("BHMSim") << "BHMSD: hit to be stored is NULL !!";
    return;
  }

  theHC->insert( hit );
}


void BHMSD::CreateNewHit() {

#ifdef debug
  LogDebug("BHMSim") << "BHMSD CreateNewHit for"
		     << " PV "     << currentPV->GetName()
		     << " PVid = " << currentPV->GetCopyNo()
		     << " Unit "   << unitID <<std::endl;
  LogDebug("BHMSim") << " primary "    << primaryID
		     << " time slice " << tSliceID 
		     << " For Track  " << theTrack->GetTrackID()
		     << " which is a " << theTrack->GetDefinition()->GetParticleName();
	   
  if (theTrack->GetTrackID()==1) {
    LogDebug("BHMSim") << " of energy "     << theTrack->GetTotalEnergy();
  } else {
    LogDebug("BHMSim") << " daughter of part. " << theTrack->GetParentID();
  }

  LogDebug("BHMSim")  << " and created by " ;
  if (theTrack->GetCreatorProcess()!=NULL)
    LogDebug("BHMSim") << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    LogDebug("BHMSim") << "NO process";
  LogDebug("BHMSim") << std::endl;
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
 

void BHMSD::UpdateHit() {

  if (Eloss > 0.) {
    currentHit->addEnergyDeposit(edepositEM,edepositHAD);

#ifdef debug
    LogDebug("BHMSim") << "updateHit: add eloss " << Eloss <<std::endl;
    LogDebug("BHMSim") << "CurrentHit=" << currentHit
		       << ", PostStepPoint=" << postStepPoint->GetPosition();
#endif
    currentHit->setEnergyLoss(Eloss);
  }  

  // buffer for next steps:
  tsID           = tSliceID;
  primID         = primaryID;
  previousUnitID = unitID;
}


G4ThreeVector BHMSD::SetToLocal(const G4ThreeVector& global) {

  const G4VTouchable* touch= preStepPoint->GetTouchable();
  theEntryPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return theEntryPoint;  
}
     

G4ThreeVector BHMSD::SetToLocalExit(const G4ThreeVector& globalPoint) {

  const G4VTouchable* touch= postStepPoint->GetTouchable();
  theExitPoint = touch->GetHistory()->GetTopTransform().TransformPoint(globalPoint);
  return theExitPoint;  
}
     

void BHMSD::EndOfEvent(G4HCofThisEvent* ) {

  // here we loop over transient hits and make them persistent
  for (int j=0; j<theHC->entries(); j++) {
    BscG4Hit* aHit = (*theHC)[j];
    LogDebug("BHMSim") << "hit number" << j << "unit ID = "<<aHit->getUnitID()<< "\n";
    LogDebug("BHMSim") << "entry z " << aHit->getEntry().z()<< "\n";
    LogDebug("BHMSim") << "entr theta " << aHit->getThetaAtEntry()<< "\n";

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
     

void BHMSD::Summarize() {
}


void BHMSD::clear() {
} 


void BHMSD::DrawAll() {
} 


void BHMSD::PrintAll() {
  LogDebug("BHMSim") << "BHMSD: Collection " << theHC->GetName() << "\n";
  theHC->PrintAllHits();
} 


void BHMSD::fillHits(edm::PSimHitContainer& c, std::string n) {
  if (slave->name() == n) c=slave->hits();
}

void BHMSD::update (const BeginOfEvent * i) {
  LogDebug("BHMSim") << " Dispatched BeginOfEvent for " << GetName()
                       << " !" ;
   clearHits();
   eventno = (*i)()->GetEventID();
}

void BHMSD::update(const BeginOfRun *) {

  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  emPDG = theParticleTable->FindParticle(particleName="e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle(particleName="e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle(particleName="gamma")->GetPDGEncoding();

} 

void BHMSD::update (const ::EndOfEvent*) {
}

void BHMSD::clearHits(){
  slave->Initialize();
}

std::vector<std::string> BHMSD::getNames(){
  std::vector<std::string> temp;
  temp.push_back(slave->name());
  return temp;
}
