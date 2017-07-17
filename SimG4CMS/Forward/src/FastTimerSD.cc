#include "SimG4CMS/Forward/interface/FastTimerSD.h"

#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

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

//#define EDM_ML_DEBUG
//-------------------------------------------------------------------
FastTimerSD::FastTimerSD(std::string name, const DDCompactView & cpv,
			 const SensitiveDetectorCatalog & clg, 
			 edm::ParameterSet const & p, 
			 const SimTrackManager* manager) :
  SensitiveTkDetector(name, cpv, clg, p), ftcons(0), name(name),
  hcID(-1), theHC(0), theManager(manager), currentHit(0), theTrack(0), 
  currentPV(0), unitID(0),  previousUnitID(0), preStepPoint(0), 
  postStepPoint(0), eventno(0) {
    
  //Add FastTimer Sentitive Detector Name
  collectionName.insert(name);
    
    
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("FastTimerSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
    
  SetVerboseLevel(verbn);
#ifdef EDM_ML_DEBUG
  std::cout << "*******************************************************\n"
	    << "*                                                     *\n"
	    << "* Constructing a FastTimerSD  with name " << name << "\n"
	    << "*                                                     *\n"
	    << "*******************************************************\n";
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
    
  std::string attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute,name,0)};
  DDFilteredView fv(cpv,filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());
  std::vector<int> temp = dbl_to_int(getDDDArray("Type",sv));
  type_  = temp[0];

  edm::LogInfo("FastTimerSim") << "FastTimerSD: Instantiation completed for "
			       << name << " of type " << type_;
}


FastTimerSD::~FastTimerSD() { 
  if (slave)  delete slave; 
}

double FastTimerSD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void FastTimerSD::Initialize(G4HCofThisEvent * HCE) { 
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimerSD : Initialize called for " << name << std::endl;
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
#ifdef EDM_ML_DEBUG
    std::cout << "FastTimerSD :  number of hits = " << theHC->entries() <<"\n";
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
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimerSD :particleType =  " 
	    << theTrack->GetDefinition()->GetParticleName() << std::endl;
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
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimerSD:unitID = " << std::hex << unitID << std::dec<<"\n";
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
  G4ThreeVector global = aStep->GetPreStepPoint()->GetPosition();
  G4ThreeVector local  = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  int        iz = (global.z() > 0) ? 1 : -1;
  std::pair<int,int> izphi = ((ftcons) ? ((type_ == 1) ? 
					  (ftcons->getZPhi(std::abs(local.z()),local.phi())) : 
					  (ftcons->getEtaPhi(local.perp(),local.phi()))) :
			      (std::pair<int,int>(0,0)));
  uint32_t id = FastTimeDetId(type_,izphi.first,izphi.second,iz).rawId();
#ifdef EDM_ML_DEBUG
  std::cout << "Volume " << touch->GetVolume(0)->GetName() << ":" << global.z()
	    << " Iz(eta)phi " << izphi.first << ":"  << izphi.second << ":" 
	    << iz  << " id " << std::hex << id << std::dec << std::endl;
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

#ifdef EDM_ML_DEBUG
  std::cout << "FastTimerSD CreateNewHit for" << " PV "
	    << currentPV->GetName() << " PVid = " << currentPV->GetCopyNo()
	    << " Unit " << unitID << std::endl;
  std::cout << " primary " << primaryID << " time slice " << tSliceID 
	    << " For Track  " << theTrack->GetTrackID() << " which is a "
	    << theTrack->GetDefinition()->GetParticleName();
	   
  if (theTrack->GetTrackID()==1) {
    std::cout << " of energy "     << theTrack->GetTotalEnergy();
  } else {
    std::cout << " daughter of part. " << theTrack->GetParentID();
  }

  std::cout << " and created by " ;
  if (theTrack->GetCreatorProcess()!=NULL)
    std::cout << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    std::cout << "NO process";
  std::cout << std::endl;
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

#ifdef EDM_ML_DEBUG
    std::cout << "updateHit: add eloss " << Eloss <<std::endl;
    std::cout << "CurrentHit="<< currentHit<< ", PostStepPoint = "
	      << postStepPoint->GetPosition() << std::endl;
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
#ifdef EDM_ML_DEBUG
    std::cout << "hit number " << j << " unit ID = " << std::hex 
	      << aHit->getUnitID() << std::dec << " entry z "
	      << aHit->getEntry().z() << " entry theta "
	      << aHit->getThetaAtEntry() << std::endl;
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
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimerSD: Collection " << theHC->GetName() << std::endl;
#endif
  theHC->PrintAllHits();
} 

void FastTimerSD::fillHits(edm::PSimHitContainer& c, std::string n) {
  if (slave->name() == n) c=slave->hits();
}

void FastTimerSD::update(const BeginOfJob * job) {

  const edm::EventSetup* es = (*job)();
  edm::ESHandle<FastTimeDDDConstants> fdc;
  es->get<IdealGeometryRecord>().get(fdc);
  if (fdc.isValid()) {
    ftcons = &(*fdc);
  } else {
    edm::LogError("FastTimerSim") << "FastTimerSD : Cannot find FastTimeDDDConstants";
    throw cms::Exception("Unknown", "FastTimerSD") << "Cannot find FastTimeDDDConstants\n";
  }
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimerSD::Initialized with FastTimeDDDConstants\n";
#endif
}

void FastTimerSD::update (const BeginOfEvent * i) {
#ifdef EDM_ML_DEBUG
  std::cout << "Dispatched BeginOfEvent for " << GetName() << " !\n" ;
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

std::vector<double> FastTimerSD::getDDDArray(const std::string & str, 
					     const DDsvalues_type & sv) {

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("FastTimerSim") << "FastTimerSD : # of " << str
				    << " bins " << nval << " < 1 ==> illegal";
      throw cms::Exception("DDException") << "FastTimerSD: cannot get array " << str;
    }
    return fvec;
  } else {
    edm::LogError("FastTimerSim") << "FastTimerSD: cannot get array " << str;
    throw cms::Exception("DDException") << "FastTimerSD: cannot get array " << str;
  }
}
