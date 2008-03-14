///////////////////////////////////////////////////////////////////////////////
// File: FP420SD.cc
// Date: 02.2006
// Description: Sensitive Detector class for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
 
//#include "Geometry/Vector/interface/LocalPoint.h"
//#include "Geometry/Vector/interface/LocalVector.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"


#include "SimG4CMS/FP420/interface/FP420SD.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4Step.hh"

#include <string>
#include <vector>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define debug
//-------------------------------------------------------------------
FP420SD::FP420SD(std::string name, const DDCompactView & cpv,
		 SensitiveDetectorCatalog & clg, 
		 edm::ParameterSet const & p, const SimTrackManager* manager) :
  SensitiveTkDetector(name, cpv, clg, p), numberingScheme(0), name(name),
  hcID(-1), theHC(0), theManager(manager), currentHit(0), theTrack(0), 
  currentPV(0), unitID(0),  previousUnitID(0), preStepPoint(0), 
  postStepPoint(0), eventno(0){
//-------------------------------------------------------------------
/*
FP420SD::FP420SD(G4String name, const DDCompactView & cpv,
		 edm::ParameterSet const & p, const SimTrackManager* manager) :
  CaloSD(name, cpv, p, manager), numberingScheme(0), name(name),hcID(-1),
  theHC(0), currentHit(0), theTrack(0), currentPV(0), 
  unitID(0),  previousUnitID(0), preStepPoint(0), postStepPoint(0), eventno(0){
*/
//-------------------------------------------------------------------


    
    //Add FP420 Sentitive Detector Name
    collectionName.insert(name);
    
    
    //Parameters
      edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("FP420SD");
       int verbn = m_p.getUntrackedParameter<int>("Verbosity");
    //int verbn = 1;
    
    SetVerboseLevel(verbn);
    LogDebug("FP420Sim") 
      << "*******************************************************\n"
      << "*                                                     *\n"
      << "* Constructing a FP420SD  with name " << name << "\n"
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
      edm::LogInfo("FP420Sim") << "FP420SD : Assigns SD to LV " << (*it);
    }
    
    if      (name == "FP420SI") {
      if (verbn > 0) {
      edm::LogInfo("FP420Sim") << "name = FP420SI and  new FP420NumberingSchem";
      }
      numberingScheme = new FP420NumberingScheme() ;
    } else {
      edm::LogWarning("FP420Sim") << "FP420SD: ReadoutName not supported\n";
    }
    
    edm::LogInfo("FP420Sim") << "FP420SD: Instantiation completed";
  }




FP420SD::~FP420SD() { 
  //AZ:
  if (slave) delete slave; 

  if (numberingScheme)
    delete numberingScheme;

}

double FP420SD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void FP420SD::Initialize(G4HCofThisEvent * HCE) { 
#ifdef debug
    LogDebug("FP420Sim") << "FP420SD : Initialize called for " << name << std::endl;
#endif

  theHC = new FP420G4HitCollection(name, collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID   = -2;
  primID = -2;

  ////    slave->Initialize();
}


bool FP420SD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  if (aStep == NULL) {
    return true;
  } else {
    GetStepInfo(aStep);
  //   LogDebug("FP420Sim") << edeposit <<std::endl;

    //AZ
#ifdef debug
  LogDebug("FP420Sim") << "FP420SD :  number of hits = " << theHC->entries() << std::endl;
#endif

    if (HitExists() == false && edeposit>0. &&  theHC->entries()< 100 ){ 
      CreateNewHit();
    return true;
    }
  }
    return true;
} 

void FP420SD::GetStepInfo(G4Step* aStep) {
  
  preStepPoint = aStep->GetPreStepPoint(); 
  postStepPoint= aStep->GetPostStepPoint(); 
  theTrack     = aStep->GetTrack();   
  hitPoint     = preStepPoint->GetPosition();	
  currentPV    = preStepPoint->GetPhysicalVolume();
  hitPointExit = postStepPoint->GetPosition();	

  hitPointLocal = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  hitPointLocalExit = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPointExit);


  G4String particleType = theTrack->GetDefinition()->GetParticleName();
//     LogDebug("FP420Sim") <<  "  FP420SD :particleType =  " << particleType <<std::endl;
  if (particleType == "e-" ||
      particleType == "e+" ||
      particleType == "gamma" ){
    edepositEM  = getEnergyDeposit(aStep); edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = getEnergyDeposit(aStep);
  }
  edeposit = aStep->GetTotalEnergyDeposit();
  tSlice    = (postStepPoint->GetGlobalTime() )/nanosecond;
  tSliceID  = (int) tSlice;
  unitID    = setDetUnitId(aStep);
#ifdef debug
  LogDebug("FP420Sim") << "unitID=" << unitID <<std::endl;
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

uint32_t FP420SD::setDetUnitId(G4Step * aStep) { 

  return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(aStep));
}


G4bool FP420SD::HitExists() {
  if (primaryID<1) {
    edm::LogWarning("FP420Sim") << "***** FP420SD error: primaryID = " 
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

  //    LogDebug("FP420Sim") << "FP420SD: HCollection=  " << theHC->entries()    <<std::endl;
  
  for (int j=0; j<theHC->entries()&&!found; j++) {
    FP420G4Hit* aPreviousHit = (*theHC)[j];
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


void FP420SD::ResetForNewPrimary() {
  
  entrancePoint  = SetToLocal(hitPoint);
  exitPoint      = SetToLocalExit(hitPointExit);
  incidentEnergy = preStepPoint->GetKineticEnergy();

}


void FP420SD::StoreHit(FP420G4Hit* hit){

  if (primID<0) return;
  if (hit == 0 ) {
    edm::LogWarning("FP420Sim") << "FP420SD: hit to be stored is NULL !!";
    return;
  }

  theHC->insert( hit );
}


void FP420SD::CreateNewHit() {

#ifdef debug
  //       << " MVid = " << currentPV->GetMother()->GetCopyNo()
  LogDebug("FP420Sim") << "FP420SD CreateNewHit for"
       << " PV "     << currentPV->GetName()
       << " PVid = " << currentPV->GetCopyNo()
       << " Unit "   << unitID <<std::endl;
  LogDebug("FP420Sim") << " primary "    << primaryID
       << " time slice " << tSliceID 
       << " For Track  " << theTrack->GetTrackID()
       << " which is a " << theTrack->GetDefinition()->GetParticleName();
	   
  if (theTrack->GetTrackID()==1) {
    LogDebug("FP420Sim") << " of energy "     << theTrack->GetTotalEnergy();
  } else {
    LogDebug("FP420Sim") << " daughter of part. " << theTrack->GetParentID();
  }

  LogDebug("FP420Sim")  << " and created by " ;
  if (theTrack->GetCreatorProcess()!=NULL)
    LogDebug("FP420Sim") << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    LogDebug("FP420Sim") << "NO process";
  LogDebug("FP420Sim") << std::endl;
#endif          
    

  currentHit = new FP420G4Hit;
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

// currentHit->setEntry(entrancePoint);
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
  //AZ:12.10.2007
  //  UpdateHit();
  // buffer for next steps:
  tsID           = tSliceID;
  primID         = primaryID;
  previousUnitID = unitID;
  
  StoreHit(currentHit);
}	 
 

void FP420SD::UpdateHit() {
  //
  if (Eloss > 0.) {
  currentHit->addEnergyDeposit(edepositEM,edepositHAD);

#ifdef debug
    LogDebug("FP420Sim") << "updateHit: add eloss " << Eloss <<std::endl;
    LogDebug("FP420Sim") << "CurrentHit=" << currentHit
         << ", PostStepPoint=" << postStepPoint->GetPosition() << std::endl;
#endif
    //AZ
    //   currentHit->setEnergyLoss(Eloss);
        currentHit->addEnergyLoss(Eloss);
  }  



  // buffer for next steps:
  tsID           = tSliceID;
  primID         = primaryID;
  previousUnitID = unitID;
}


G4ThreeVector FP420SD::SetToLocal(G4ThreeVector global){

  const G4VTouchable* touch= preStepPoint->GetTouchable();
  theEntryPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return theEntryPoint;  
}
     

G4ThreeVector FP420SD::SetToLocalExit(G4ThreeVector globalPoint){

  const G4VTouchable* touch= postStepPoint->GetTouchable();
  theExitPoint = touch->GetHistory()->GetTopTransform().TransformPoint(globalPoint);
  return theExitPoint;  
}
     

void FP420SD::EndOfEvent(G4HCofThisEvent* ) {

  // here we loop over transient hits and make them persistent
  if(theHC->entries() > 100){
    LogDebug("FP420Sim") << "FP420SD: warning!!! Number of hits exceed 100 and =" << theHC->entries() << "\n";
  }
  for (int j=0; j<theHC->entries() && j<100; j++) {
    //AZ:
              FP420G4Hit* aHit = (*theHC)[j];
#ifdef ddebug
    //    LogDebug("FP420SD") << " FP420Hit " << j << " " << *aHit << std::endl;
  //  slave->ProcessHits(aHit->getUnitID(), aHit->getEnergyDeposit()/GeV,
  //	       aHit->getTimeSlice(), aHit->getTrackID());
    LogDebug("FP420Sim") << "hit number" << j << "unit ID = "<<aHit->getUnitID()<< "\n";
    LogDebug("FP420Sim") << "entry z " << aHit->getEntry().z()<< "\n";
    LogDebug("FP420Sim") << "entr theta " << aHit->getThetaAtEntry()<< "\n";
#endif

    //    Local3DPoint locExitPoint(0,0,0);
    //  Local3DPoint locEntryPoint(aHit->getEntry().x(),
    //	 aHit->getEntry().y(),
    //	 aHit->getEntry().z());
    Local3DPoint locExitPoint(aHit->getExitLocalP().x(),
			 aHit->getExitLocalP().y(),
			 aHit->getExitLocalP().z());
    Local3DPoint locEntryPoint(aHit->getEntryLocalP().x(),
			 aHit->getEntryLocalP().y(),
			 aHit->getEntryLocalP().z());
    slave->processHits(PSimHit(locEntryPoint,locExitPoint,
			       aHit->getPabs(),
			       aHit->getTof(),
			       aHit->getEnergyLoss(),
			       aHit->getParticleType(),
			       aHit->getUnitID(),
			       aHit->getTrackID(),
			       aHit->getThetaAtEntry(),
			       aHit->getPhiAtEntry()));
    /*      
			       aHit->getEM(),               -
			       aHit->getHadr(),             -
			       aHit->getIncidentEnergy(),   -
			       aHit->getTimeSlice(),       -
			       aHit->getEntry(),     -
			       aHit->getParentId(),     
			       aHit->getEntryLocalP(),  -
			       aHit->getExitLocalP(),   -
			       aHit->getX(),    -
			       aHit->getY(),   -
			       aHit->getZ(),   -
			       aHit->getVx(),  -
			       aHit->getVy(),  -
			       aHit->getVz()));  -
*/

  }
  Summarize();
}


void FP420SD::Summarize() {
}


void FP420SD::clear() {
} 


void FP420SD::DrawAll() {
} 


void FP420SD::PrintAll() {
  LogDebug("FP420Sim") << "FP420SD: Collection " << theHC->GetName() << "\n";
  theHC->PrintAllHits();
} 


//void FP420SD::SetNumberingScheme(FP420NumberingScheme* scheme){
//
//  if (numberingScheme)
//    delete numberingScheme;
//  numberingScheme = scheme;
//
//}

void FP420SD::fillHits(edm::PSimHitContainer& c, std::string n) {
  if (slave->name() == n) c=slave->hits();
}

void FP420SD::update (const BeginOfEvent * i) {
  LogDebug("ForwardSim") << " Dispatched BeginOfEvent for " << GetName()
                       << " !" ;
   clearHits();
   eventno = (*i)()->GetEventID();
}

void FP420SD::update (const ::EndOfEvent*) {
}

void FP420SD::clearHits(){
  //AZ:
    slave->Initialize();
}

std::vector<std::string> FP420SD::getNames(){
  std::vector<std::string> temp;
  temp.push_back(slave->name());
  return temp;
}

