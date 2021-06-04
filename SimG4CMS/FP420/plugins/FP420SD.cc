///////////////////////////////////////////////////////////////////////////////
// File: FP420SD.cc
// Date: 02.2006
// Description: Sensitive Detector class for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "SimG4CMS/FP420/interface/FP420SD.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4Step.hh"
#include "G4ParticleTable.hh"

#include "G4SystemOfUnits.hh"

#include <string>
#include <vector>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define debug
//-------------------------------------------------------------------
FP420SD::FP420SD(const std::string& name,
                 const edm::EventSetup& es,
                 const SensitiveDetectorCatalog& clg,
                 edm::ParameterSet const& p,
                 const SimTrackManager* manager)
    : SensitiveTkDetector(name, clg),
      numberingScheme(nullptr),
      hcID(-1),
      theHC(nullptr),
      theManager(manager),
      currentHit(nullptr),
      theTrack(nullptr),
      currentPV(nullptr),
      unitID(0),
      previousUnitID(0),
      preStepPoint(nullptr),
      postStepPoint(nullptr),
      eventno(0) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("FP420SD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
  //int verbn = 1;

  SetVerboseLevel(verbn);

  slave = new TrackingSlaveSD(name);

  if (name == "FP420SI") {
    if (verbn > 0) {
      edm::LogInfo("FP420Sim") << "name = FP420SI and  new FP420NumberingSchem";
    }
    numberingScheme = new FP420NumberingScheme();
  } else {
    edm::LogWarning("FP420Sim") << "FP420SD: ReadoutName not supported\n";
  }
}

FP420SD::~FP420SD() {
  delete slave;
  delete numberingScheme;
}

double FP420SD::getEnergyDeposit(G4Step* aStep) { return aStep->GetTotalEnergyDeposit(); }

void FP420SD::Initialize(G4HCofThisEvent* HCE) {
#ifdef debug
  LogDebug("FP420Sim") << "FP420SD : Initialize called for " << name << std::endl;
#endif

  theHC = new FP420G4HitCollection(GetName(), collectionName[0]);
  if (hcID < 0)
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID = -2;
  //  primID = -2;
  primID = 0;

  ////    slave->Initialize();
}

bool FP420SD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  if (aStep == nullptr) {
    return true;
  } else {
    GetStepInfo(aStep);
    //   LogDebug("FP420Sim") << edeposit <<std::endl;

    //AZ
#ifdef debug
    LogDebug("FP420Sim") << "FP420SD :  number of hits = " << theHC->entries() << std::endl;
#endif

    if (HitExists() == false && edeposit > 0. && theHC->entries() < 200) {
      CreateNewHit();
      return true;
    }
  }
  return true;
}

void FP420SD::GetStepInfo(G4Step* aStep) {
  preStepPoint = aStep->GetPreStepPoint();
  postStepPoint = aStep->GetPostStepPoint();
  theTrack = aStep->GetTrack();
  hitPoint = preStepPoint->GetPosition();
  currentPV = preStepPoint->GetPhysicalVolume();
  hitPointExit = postStepPoint->GetPosition();

  hitPointLocal = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  hitPointLocalExit = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPointExit);

  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  if (particleCode == emPDG || particleCode == epPDG || particleCode == gammaPDG) {
    edepositEM = getEnergyDeposit(aStep);
    edepositHAD = 0.;
  } else {
    edepositEM = 0.;
    edepositHAD = getEnergyDeposit(aStep);
  }
  edeposit = aStep->GetTotalEnergyDeposit();
  tSlice = (postStepPoint->GetGlobalTime()) / nanosecond;
  tSliceID = (int)tSlice;
  unitID = setDetUnitId(aStep);
#ifdef debug
  LogDebug("FP420Sim") << "unitID=" << unitID << std::endl;
#endif
  primaryID = theTrack->GetTrackID();
  //  Position     = hitPoint;
  Pabs = aStep->GetPreStepPoint()->GetMomentum().mag() / GeV;
  //Tof          = 1400. + aStep->GetPostStepPoint()->GetGlobalTime()/nanosecond;
  Tof = aStep->GetPostStepPoint()->GetGlobalTime() / nanosecond;
  Eloss = aStep->GetTotalEnergyDeposit() / GeV;
  ParticleType = theTrack->GetDefinition()->GetPDGEncoding();
  ThetaAtEntry = aStep->GetPreStepPoint()->GetPosition().theta() / deg;
  PhiAtEntry = aStep->GetPreStepPoint()->GetPosition().phi() / deg;

  ParentId = theTrack->GetParentID();
  Vx = theTrack->GetVertexPosition().x();
  Vy = theTrack->GetVertexPosition().y();
  Vz = theTrack->GetVertexPosition().z();
  X = hitPoint.x();
  Y = hitPoint.y();
  Z = hitPoint.z();
}

uint32_t FP420SD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme == nullptr ? 0 : numberingScheme->getUnitID(aStep));
}

G4bool FP420SD::HitExists() {
  if (primaryID < 1) {
    edm::LogWarning("FP420Sim") << "***** FP420SD error: primaryID = " << primaryID << " maybe detector name changed";
  }

  // Update if in the same detector, time-slice and for same track
  //  if (primaryID == primID && tSliceID == tsID && unitID==previousUnitID) {
  if (tSliceID == tsID && unitID == previousUnitID) {
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
  int nhits = theHC->entries();
  for (int j = 0; j < nhits && !found; j++) {
    FP420G4Hit* aPreviousHit = (*theHC)[j];
    if (aPreviousHit->getTrackID() == primaryID && aPreviousHit->getTimeSliceID() == tSliceID &&
        aPreviousHit->getUnitID() == unitID) {
      //AZ:
      currentHit = aPreviousHit;
      found = true;
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
  entrancePoint = SetToLocal(hitPoint);
  exitPoint = SetToLocalExit(hitPointExit);
  incidentEnergy = preStepPoint->GetKineticEnergy();
}

void FP420SD::StoreHit(FP420G4Hit* hit) {
  //  if (primID<0) return;
  if (hit == nullptr) {
    edm::LogWarning("FP420Sim") << "FP420SD: hit to be stored is NULL !!";
    return;
  }

  theHC->insert(hit);
}

void FP420SD::CreateNewHit() {
#ifdef debug
  //       << " MVid = " << currentPV->GetMother()->GetCopyNo()
  LogDebug("FP420Sim") << "FP420SD CreateNewHit for"
                       << " PV " << currentPV->GetName() << " PVid = " << currentPV->GetCopyNo() << " Unit " << unitID
                       << std::endl;
  LogDebug("FP420Sim") << " primary " << primaryID << " time slice " << tSliceID << " For Track  "
                       << theTrack->GetTrackID() << " which is a " << theTrack->GetDefinition()->GetParticleName();

  if (theTrack->GetTrackID() == 1) {
    LogDebug("FP420Sim") << " of energy " << theTrack->GetTotalEnergy();
  } else {
    LogDebug("FP420Sim") << " daughter of part. " << theTrack->GetParentID();
  }

  LogDebug("FP420Sim") << " and created by ";
  if (theTrack->GetCreatorProcess() != NULL)
    LogDebug("FP420Sim") << theTrack->GetCreatorProcess()->GetProcessName();
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
  tsID = tSliceID;
  primID = primaryID;
  previousUnitID = unitID;

  StoreHit(currentHit);
}

void FP420SD::UpdateHit() {
  //
  if (Eloss > 0.) {
    currentHit->addEnergyDeposit(edepositEM, edepositHAD);

#ifdef debug
    LogDebug("FP420Sim") << "updateHit: add eloss " << Eloss << std::endl;
    LogDebug("FP420Sim") << "CurrentHit=" << currentHit << ", PostStepPoint=" << postStepPoint->GetPosition()
                         << std::endl;
#endif
    //AZ
    //   currentHit->setEnergyLoss(Eloss);
    currentHit->addEnergyLoss(Eloss);
  }

  // buffer for next steps:
  tsID = tSliceID;
  primID = primaryID;
  previousUnitID = unitID;
}

G4ThreeVector FP420SD::SetToLocal(const G4ThreeVector& global) {
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  theEntryPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return theEntryPoint;
}

G4ThreeVector FP420SD::SetToLocalExit(const G4ThreeVector& globalPoint) {
  const G4VTouchable* touch = postStepPoint->GetTouchable();
  theExitPoint = touch->GetHistory()->GetTopTransform().TransformPoint(globalPoint);
  return theExitPoint;
}

void FP420SD::EndOfEvent(G4HCofThisEvent*) {
  // here we loop over transient hits and make them persistent

  //  if(theHC->entries() > 100){
  //    LogDebug("FP420Sim") << "FP420SD: warning!!! Number of hits exceed 100 and =" << theHC->entries() << "\n";
  //  }
  //  for (int j=0; j<theHC->entries() && j<100; j++) {
  int nhitsHPS240 = 0;
  int nhitsFP420 = 0;
  int nhits = theHC->entries();
  for (int j = 0; j < nhits; j++) {
    FP420G4Hit* aHit = (*theHC)[j];
    if ((fabs(aHit->getTof()) > 780. && fabs(aHit->getTof()) < 840.))
      ++nhitsHPS240;
    if ((fabs(aHit->getTof()) > 1380. && fabs(aHit->getTof()) < 1450.))
      ++nhitsFP420;
    //    if(fabs(aHit->getTof()) < 1700.) {
    if ((fabs(aHit->getTof()) > 780. && fabs(aHit->getTof()) < 840. && nhitsHPS240 < 200.) ||
        (fabs(aHit->getTof()) > 1380. && fabs(aHit->getTof()) < 1450. && nhitsFP420 < 200.)) {
#ifdef ddebug
      //    LogDebug("FP420SD") << " FP420Hit " << j << " " << *aHit << std::endl;
      LogDebug("FP420Sim") << "hit number" << j << "unit ID = " << aHit->getUnitID() << "\n";
      LogDebug("FP420Sim") << "entry z " << aHit->getEntry().z() << "\n";
      LogDebug("FP420Sim") << "entr theta " << aHit->getThetaAtEntry() << "\n";
#endif

      //    Local3DPoint locExitPoint(0,0,0);
      //  Local3DPoint locEntryPoint(aHit->getEntry().x(),
      //	 aHit->getEntry().y(),
      //	 aHit->getEntry().z());
      Local3DPoint locExitPoint(aHit->getExitLocalP().x(), aHit->getExitLocalP().y(), aHit->getExitLocalP().z());
      Local3DPoint locEntryPoint(aHit->getEntryLocalP().x(), aHit->getEntryLocalP().y(), aHit->getEntryLocalP().z());
      // implicit conversion (slicing) to PSimHit!!!
      slave->processHits(PSimHit(locEntryPoint,
                                 locExitPoint,             //entryPoint(), exitPoint()  Local3DPoint
                                 aHit->getPabs(),          // pabs()  float
                                 aHit->getTof(),           // tof() float
                                 aHit->getEnergyLoss(),    // energyLoss() float
                                 aHit->getParticleType(),  // particleType()   int
                                 aHit->getUnitID(),        // detUnitId() unsigned int
                                 aHit->getTrackID(),       // trackId() unsigned int
                                 aHit->getThetaAtEntry(),  //  thetaAtEntry()   float
                                 aHit->getPhiAtEntry()));  //  phiAtEntry()   float

      //PSimHit( const Local3DPoint& entry, const Local3DPoint& exit,
      //	   float pabs, float tof, float eloss, int particleType,
      //	   unsigned int detId, unsigned int trackId,
      //	   float theta, float phi, unsigned short processType=0) :

      //  LocalVector direction = hit.exitPoint() - hit.entryPoint();
      //hit.energyLoss

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
    }  //if Tof<1600. if nhits<100
  }    // for loop on hits

  Summarize();
}

void FP420SD::Summarize() {}

void FP420SD::clear() {}

void FP420SD::DrawAll() {}

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

void FP420SD::fillHits(edm::PSimHitContainer& cc, const std::string& hname) {
  if (slave->name() == hname) {
    cc = slave->hits();
  }
}

void FP420SD::update(const BeginOfEvent* i) {
  LogDebug("ForwardSim") << " Dispatched BeginOfEvent for " << GetName() << " !";
  clearHits();
  eventno = (*i)()->GetEventID();
}

void FP420SD::update(const BeginOfRun*) {
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  emPDG = theParticleTable->FindParticle(particleName = "e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle(particleName = "e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle(particleName = "gamma")->GetPDGEncoding();
}

void FP420SD::update(const ::EndOfEvent*) {}

void FP420SD::clearHits() {
  //AZ:
  slave->Initialize();
}
