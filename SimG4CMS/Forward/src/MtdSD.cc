#include "SimG4CMS/Forward/interface/MtdSD.h"

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

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4Step.hh"

#include <string>
#include <vector>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
//-------------------------------------------------------------------
MtdSD::MtdSD(const std::string& name, const DDCompactView & cpv,
	     const SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, 
	     const SimTrackManager* manager) :
  SensitiveTkDetector(name, cpv, clg, p), 
  hcID(-1), theHC(nullptr), theManager(manager), currentHit(nullptr), theTrack(nullptr), 
  currentPV(nullptr), unitID(0),  previousUnitID(0), preStepPoint(nullptr), 
  postStepPoint(nullptr), numberingScheme(nullptr) {
    
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MtdSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
    
  SetVerboseLevel(verbn);
    
  slave  = new TrackingSlaveSD(name);
        
  std::string attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute,name,0)};
  DDFilteredView fv(cpv,filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());
  std::vector<int> temp = dbl_to_int(getDDDArray("Type",sv));
  type_  = temp[0];

  MTDNumberingScheme* scheme=nullptr;
  if (name == "FastTimerHitsBarrel") {
    scheme = dynamic_cast<MTDNumberingScheme*>(new BTLNumberingScheme());
    isBTL=true;
  } else if (name == "FastTimerHitsEndcap") { 
    scheme = dynamic_cast<MTDNumberingScheme*>(new ETLNumberingScheme());
    isETL=true;
  } else {
    scheme = nullptr;
    edm::LogWarning("EcalSim") << "MtdSD: ReadoutName not supported";
  }
  if (scheme)  setNumberingScheme(scheme);

  edm::LogInfo("MtdSim") << "MtdSD: Instantiation completed for "
			       << name << " of type " << type_;
}

MtdSD::~MtdSD() { 
  delete numberingScheme;
  delete slave; 
}

void MtdSD::Initialize(G4HCofThisEvent * HCE) { 
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MtdSim") << "MtdSD : Initialize called for " << GetName();
#endif

  theHC = new BscG4HitCollection(GetName(), collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID   = -2;
  primID = -2;
}

bool MtdSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  edeposit = aStep->GetTotalEnergyDeposit();
  if (edeposit>0.f){ 
    GetStepInfo(aStep);
#ifdef EDM_ML_DEBUG
    edm::LogInfo("MtdSim") << "MtdSD :  number of hits = " << theHC->entries() <<"\n";
#endif
    if (!HitExists()){ 
      CreateNewHit();
    }
  }
  return true;
} 

void MtdSD::GetStepInfo(const G4Step* aStep) {
  
  preStepPoint = aStep->GetPreStepPoint(); 
  postStepPoint= aStep->GetPostStepPoint(); 
  theTrack     = aStep->GetTrack();   
  hitPoint     = preStepPoint->GetPosition();	
  currentPV    = preStepPoint->GetPhysicalVolume();
  hitPointExit = postStepPoint->GetPosition();	

  hitPointLocal = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  hitPointLocalExit = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPointExit);

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MtdSim") << "MtdSD :particleType =  " 
	    << theTrack->GetDefinition()->GetParticleName();
#endif
  edeposit /= CLHEP::GeV;
  if ( G4TrackToParticleID::isGammaElectronPositron(theTrack) ) {
    edepositEM  = edeposit; edepositHAD = 0.f;
  } else {
    edepositEM  = 0.f; edepositHAD = edeposit;
  }
  tSlice    = (100*postStepPoint->GetGlobalTime() )/CLHEP::nanosecond;
  tSliceID  = (int) tSlice;
  unitID    = setDetUnitId(aStep);
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MtdSim") << "MtdSD:unitID = " << std::hex << unitID << std::dec<<"\n";
#endif
  primaryID    = theTrack->GetTrackID();
  //  Position     = hitPoint;
  Pabs         = aStep->GetPreStepPoint()->GetMomentum().mag()/CLHEP::GeV;
  Tof          = aStep->GetPostStepPoint()->GetGlobalTime()/CLHEP::nanosecond;
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

uint32_t MtdSD::setDetUnitId(const G4Step * aStep) { 
  if (numberingScheme == nullptr) {
    return MTDDetId();
  } else {
    getBaseNumber(aStep);
    return numberingScheme->getUnitID(theBaseNumber);
  }
}

G4bool MtdSD::HitExists() {
  if (primaryID<1) {
    edm::LogWarning("MtdSim") << "***** MtdSD error: primaryID = " 
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

  for (int j=0; j<theHC->entries(); ++j) {
    BscG4Hit* aPreviousHit = (*theHC)[j];
    if (aPreviousHit->getTrackID()     == primaryID &&
	aPreviousHit->getTimeSliceID() == tSliceID  &&
	aPreviousHit->getUnitID()      == unitID       ) {
      currentHit = aPreviousHit;
      found      = true;
      break;
    }
  }          

  if (found) { UpdateHit(); }
  return found;
}

void MtdSD::ResetForNewPrimary() {
  entrancePoint  = SetToLocal(hitPoint);
  exitPoint      = SetToLocalExit(hitPointExit);
  incidentEnergy = preStepPoint->GetKineticEnergy();
}

void MtdSD::StoreHit(BscG4Hit* hit){

  if (primID<0) return;
  if (hit == nullptr) {
    edm::LogWarning("MtdSim") << "MtdSD: hit to be stored is NULL !!";
  } else {
    theHC->insert( hit );
  }
}

void MtdSD::CreateNewHit() {

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MtdSim") << "MtdSD CreateNewHit for" << " PV "
	    << currentPV->GetName() << " PVid = " << currentPV->GetCopyNo()
	    << " Unit " << unitID ;
  edm::LogInfo("MtdSim") << " primary " << primaryID << " time slice " << tSliceID 
	    << " For Track  " << theTrack->GetTrackID() << " which is a "
	    << theTrack->GetDefinition()->GetParticleName();
	   
  if (theTrack->GetTrackID()==1) {
    edm::LogInfo("MtdSim") << " of energy "     << theTrack->GetTotalEnergy();
  } else {
    edm::LogInfo("MtdSim") << " daughter of part. " << theTrack->GetParentID();
  }

  edm::LogInfo("MtdSim") << " and created by " ;
  if (theTrack->GetCreatorProcess()!=NULL)
    edm::LogInfo("MtdSim") << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    edm::LogInfo("MtdSim") << "NO process";
  edm::LogInfo("MtdSim") ;
#endif          
    
  currentHit = new BscG4Hit;
  currentHit->setTrackID(primaryID);
  currentHit->setTimeSlice(tSlice);
  currentHit->setUnitID(unitID);
  currentHit->setIncidentEnergy(incidentEnergy);

  currentHit->setPabs(Pabs);
  currentHit->setTof(Tof);
  currentHit->setEnergyLoss(edeposit);
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
 
void MtdSD::UpdateHit() {

  currentHit->addEnergyDeposit(edepositEM,edepositHAD);

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MtdSim") << "updateHit: add eloss " << edeposit;
  edm::LogInfo("MtdSim") << "CurrentHit="<< currentHit<< ", PostStepPoint = "
			 << postStepPoint->GetPosition() ;
#endif

  // buffer for next steps:
  tsID           = tSliceID;
  primID         = primaryID;
  previousUnitID = unitID;
}

G4ThreeVector MtdSD::SetToLocal(const G4ThreeVector& global){

  const G4VTouchable* touch= preStepPoint->GetTouchable();
  theEntryPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return theEntryPoint;  
}
     
G4ThreeVector MtdSD::SetToLocalExit(const G4ThreeVector& globalPoint){

  const G4VTouchable* touch= postStepPoint->GetTouchable();
  theExitPoint = touch->GetHistory()->GetTopTransform().TransformPoint(globalPoint);
  return theExitPoint;  
}
     
void MtdSD::EndOfEvent(G4HCofThisEvent* ) {

  // here we loop over transient hits and make them persistent
  for (int j=0; j<theHC->entries(); j++) {
    BscG4Hit* aHit = (*theHC)[j];
#ifdef EDM_ML_DEBUG
    edm::LogInfo("MtdSim") << "hit number " << j << " unit ID = " << std::hex 
	      << aHit->getUnitID() << std::dec << " entry z "
	      << aHit->getEntry().z() << " entry theta "
	      << aHit->getThetaAtEntry() ;
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
}
     
void MtdSD::PrintAll() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MtdSim") << "MtdSD: Collection " << theHC->GetName() ;
#endif
  theHC->PrintAllHits();
} 

void MtdSD::fillHits(edm::PSimHitContainer& cc, const std::string& hname) {
  if (slave->name() == hname) { cc=slave->hits(); }
}

void MtdSD::update (const BeginOfEvent * i) {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MtdSim") << "Dispatched BeginOfEvent for " << GetName() << " !\n" ;
#endif
   clearHits();
}

void MtdSD::clearHits(){
  slave->Initialize();
}

std::vector<double> MtdSD::getDDDArray(const std::string & str, 
					     const DDsvalues_type & sv) {

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("MtdSim") << "MtdSD : # of " << str
				    << " bins " << nval << " < 1 ==> illegal";
      throw cms::Exception("DDException") << "MtdSD: cannot get array " << str;
    }
    return fvec;
  } else {
    edm::LogError("MtdSim") << "MtdSD: cannot get array " << str;
    throw cms::Exception("DDException") << "MtdSD: cannot get array " << str;
  }
}

void MtdSD::setNumberingScheme(MTDNumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogInfo("MtdSim") << "MtdSD: updates numbering scheme for " 
                            << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

void MtdSD::getBaseNumber(const G4Step* aStep) {

  theBaseNumber.reset();
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  int theSize = touch->GetHistoryDepth()+1;
  if ( theBaseNumber.getCapacity() < theSize ) theBaseNumber.setSize(theSize);
  //Get name and copy numbers
  if ( theSize > 1 ) {
    for (int ii = 0; ii < theSize ; ii++) {
      theBaseNumber.addLevel(touch->GetVolume(ii)->GetName(),touch->GetReplicaNumber(ii));
#ifdef EDM_ML_DEBUG
      edm::LogInfo("MtdSim") << "MtdSD::getBaseNumber(): Adding level " << ii
                              << ": " << touch->GetVolume(ii)->GetName() << "["
                              << touch->GetReplicaNumber(ii) << "]";
#endif
    }
  }
}
