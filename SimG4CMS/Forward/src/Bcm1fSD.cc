#include "SimG4CMS/Forward/interface/Bcm1fSD.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
 
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4VProcess.hh"

#include <string>
#include <vector>
#include <iostream>

#include "CLHEP/Units/GlobalSystemOfUnits.h"

Bcm1fSD::Bcm1fSD(std::string name, 
					   const DDCompactView & cpv,
					   SensitiveDetectorCatalog & clg, 
					   edm::ParameterSet const & p,
					   const SimTrackManager* manager) : 
  SensitiveTkDetector(name, cpv, clg, p), myName(name), mySimHit(0),
  oldVolume(0), lastId(0), lastTrack(0), eventno(0) {
  
  edm::ParameterSet m_TrackerSD = p.getParameter<edm::ParameterSet>("Bcm1fSD");
  energyCut           = m_TrackerSD.getParameter<double>("EnergyThresholdForPersistencyInGeV")*GeV; //default must be 0.5 (?)
  energyHistoryCut    = m_TrackerSD.getParameter<double>("EnergyThresholdForHistoryInGeV")*GeV;//default must be 0.05 (?)

  edm::LogInfo("Bcm1fSD") <<"Criteria for Saving Tracker SimTracks: \n "
				       <<" History: "<<energyHistoryCut<< " MeV ; Persistency: "<< energyCut<<" MeV\n"
				       <<" Constructing a Bcm1fSD with ";

  slave  = new TrackingSlaveSD(name);
  
  // Now attach the right detectors (LogicalVolumes) to me
  std::vector<std::string>  lvNames = clg.logicalNames(name);
  this->Register();
  for (std::vector<std::string>::iterator it = lvNames.begin(); it != lvNames.end(); it++)
  {
     edm::LogInfo("Bcm1fSD")<< name << " attaching LV " << *it;
     this->AssignSD(*it);
  }

  theG4ProcessTypeEnumerator = new G4ProcessTypeEnumerator;
  myG4TrackToParticleID = new G4TrackToParticleID;
}

Bcm1fSD::~Bcm1fSD() { 
  delete slave;
  delete theG4ProcessTypeEnumerator;
  delete myG4TrackToParticleID;
}

bool Bcm1fSD::ProcessHits(G4Step * aStep,  G4TouchableHistory *) {

  LogDebug("Bcm1fSD") << " Entering a new Step " 
		    << aStep->GetTotalEnergyDeposit() << " " 
		    << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName();

  G4Track * gTrack  = aStep->GetTrack(); 
  if ((unsigned int)(gTrack->GetTrackID()) != lastTrack) {
    
    if (gTrack->GetKineticEnergy() > energyCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
      info->storeTrack(true);
    }
    if (gTrack->GetKineticEnergy() > energyHistoryCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
      info->putInHistory();
    }
  }

  if (aStep->GetTotalEnergyDeposit()>0.) {
    if (newHit(aStep) == true) {
      sendHit();
      createHit(aStep);
    } else {
      updateHit(aStep);
    }
    return true;
  }
  return false;
}

uint32_t Bcm1fSD::setDetUnitId(G4Step * aStep ) {
 
  unsigned int detId = 0;

  LogDebug("Bcm1fSD")<< " DetID = "<<detId; 
  
  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch) level = ((touch->GetHistoryDepth())+1);
  
  //Get name and copy numbers
  if ( level > 1 ) {
     
     G4String sensorName   = touch->GetVolume(0)->GetName();
     G4String diamondName  = touch->GetVolume(1)->GetName();
     G4String detectorName = touch->GetVolume(2)->GetName();
     G4String volumeName   = touch->GetVolume(3)->GetName();
     
     if ( sensorName != "BCM1FSensor" )
        std::cout << " Bcm1fSD::setDetUnitId -w- Sensor name not BCM1FSensor " << std::endl;
     if ( detectorName != "BCM1F" )
        std::cout << " Bcm1fSD::setDetUnitId -w- Detector name not BCM1F " << std::endl;
     
     int sensorNo    = touch->GetReplicaNumber(0);
     int diamondNo   = touch->GetReplicaNumber(1);
//     int detectorNo  = touch->GetReplicaNumber(2);
     int volumeNo    = touch->GetReplicaNumber(3);
     
     // Detector ID definition
     // detId = XYYZ
     // X  = volume,  1: +Z, 2: -Z
     // YY = diamond, 01-12, 12: phi = 90 deg, numbering clockwise when looking from the IP
     // Z  = sensor,  1 or 2, clockwise when looking from the IP
     
//     detId = 10*volumeNo + diamondNo;
     detId = 1000*volumeNo + 10*diamondNo + sensorNo;
     
//     for ( int ii = 0; ii < level; ii++ ) {
//       int i      = level - ii - 1;
//       G4String name   = touch->GetVolume(i)->GetName();
//       int copyno = touch->GetReplicaNumber(i);
//       const G4ThreeVector trans = touch->GetVolume(i)->GetTranslation();
//       std::cout << " detId = " << detId << " name = " << name << " " << copyno
//             << ",     translation x, y, z = " << trans.x() << ", " <<trans.y() << ", " <<trans.z()
//             << ", eta = " << trans.eta() << std::endl;
//     }
  }

  return detId;
}

void Bcm1fSD::EndOfEvent(G4HCofThisEvent *) {
  
  LogDebug("Bcm1fSD")<< " Saving the last hit in a ROU " << myName;

  if (mySimHit == 0) return;
  sendHit();
}

void Bcm1fSD::fillHits(edm::PSimHitContainer& c, std::string n){
  if (slave->name() == n)  c=slave->hits();
}

void Bcm1fSD::sendHit() {  
  if (mySimHit == 0) return;
  LogDebug("Bcm1fSD") << " Storing PSimHit: " << pname << " " << mySimHit->detUnitId() 
				   << " " << mySimHit->trackId() << " " << mySimHit->energyLoss() 
				   << " " << mySimHit->entryPoint() << " " << mySimHit->exitPoint();
    
  slave->processHits(*mySimHit); 

  // clean up
  delete mySimHit;
  mySimHit = 0;
  lastTrack = 0;
  lastId = 0;
}

void Bcm1fSD::updateHit(G4Step * aStep) {

  Local3DPoint theExitPoint = SensitiveDetector::FinalStepPosition(aStep,LocalCoordinates); 
  float theEnergyLoss = aStep->GetTotalEnergyDeposit()/GeV;
  mySimHit->setExitPoint(theExitPoint);
  LogDebug("Bcm1fSD")<< " Before : " << mySimHit->energyLoss();
  mySimHit->addEnergyLoss(theEnergyLoss);
  globalExitPoint = SensitiveDetector::FinalStepPosition(aStep,WorldCoordinates);

  LogDebug("Bcm1fSD") << " Updating: new exitpoint " << pname << " " 
				   << theExitPoint << " new energy loss " << theEnergyLoss 
				   << "\n Updated PSimHit: " << mySimHit->detUnitId() 
				   << " " << mySimHit->trackId()
				   << " " << mySimHit->energyLoss() << " " 
				   << mySimHit->entryPoint() << " " << mySimHit->exitPoint();
}

bool Bcm1fSD::newHit(G4Step * aStep) {

  G4Track * theTrack = aStep->GetTrack(); 
  uint32_t theDetUnitId = setDetUnitId(aStep);
  unsigned int theTrackID = theTrack->GetTrackID();

  LogDebug("Bcm1fSD") << " OLD (d,t) = (" << lastId << "," << lastTrack 
				   << "), new = (" << theDetUnitId << "," << theTrackID << ") return "
				   << ((theTrackID == lastTrack) && (lastId == theDetUnitId));
  if ((mySimHit != 0) && (theTrackID == lastTrack) && (lastId == theDetUnitId) && closeHit(aStep))
    return false;
  return true;
}

bool Bcm1fSD::closeHit(G4Step * aStep) {

  if (mySimHit == 0) return false; 
  const float tolerance = 0.05 * mm; // 50 micron are allowed between the exit 
  // point of the current hit and the entry point of the new hit
  Local3DPoint theEntryPoint = SensitiveDetector::InitialStepPosition(aStep,LocalCoordinates);  
  LogDebug("Bcm1fSD")<< " closeHit: distance = " << (mySimHit->exitPoint()-theEntryPoint).mag();

  if ((mySimHit->exitPoint()-theEntryPoint).mag()<tolerance) return true;
  return false;
}

void Bcm1fSD::createHit(G4Step * aStep) {

  if (mySimHit != 0) {
    delete mySimHit;
    mySimHit=0;
  }
    
  G4Track * theTrack  = aStep->GetTrack(); 
  G4VPhysicalVolume * v = aStep->GetPreStepPoint()->GetPhysicalVolume();

  Local3DPoint theEntryPoint = SensitiveDetector::InitialStepPosition(aStep,LocalCoordinates);  
  Local3DPoint theExitPoint  = SensitiveDetector::FinalStepPosition(aStep,LocalCoordinates); 
  
  float thePabs             = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
  float theTof              = aStep->GetPreStepPoint()->GetGlobalTime()/nanosecond;
  float theEnergyLoss       = aStep->GetTotalEnergyDeposit()/GeV;
  int theParticleType       = myG4TrackToParticleID->particleID(theTrack);
  uint32_t theDetUnitId     = setDetUnitId(aStep);
  
  globalEntryPoint = SensitiveDetector::InitialStepPosition(aStep,WorldCoordinates);
  globalExitPoint = SensitiveDetector::FinalStepPosition(aStep,WorldCoordinates);
  pname = theTrack->GetDynamicParticle()->GetDefinition()->GetParticleName();
  
  unsigned int theTrackID = theTrack->GetTrackID();
  
  G4ThreeVector gmd  = aStep->GetPreStepPoint()->GetMomentumDirection();
  // convert it to local frame
  G4ThreeVector lmd = ((G4TouchableHistory *)(aStep->GetPreStepPoint()->GetTouchable()))->GetHistory()->GetTopTransform().TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  float theThetaAtEntry = lnmd.theta();
  float thePhiAtEntry = lnmd.phi();
    
  mySimHit = new UpdatablePSimHit(theEntryPoint,theExitPoint,thePabs,theTof,
				  theEnergyLoss,theParticleType,theDetUnitId,
				  theTrackID,theThetaAtEntry,thePhiAtEntry,
				  theG4ProcessTypeEnumerator->processId(theTrack->GetCreatorProcess()));  

  LogDebug("Bcm1fSD") << " Created PSimHit: " << pname << " " 
				   << mySimHit->detUnitId() << " " << mySimHit->trackId()
				   << " " << mySimHit->energyLoss() << " " 
				   << mySimHit->entryPoint() << " " << mySimHit->exitPoint();
  lastId = theDetUnitId;
  lastTrack = theTrackID;
  oldVolume = v;
}

void Bcm1fSD::update(const BeginOfJob * ) { }

void Bcm1fSD::update(const BeginOfEvent * i) {

  clearHits();
  eventno = (*i)()->GetEventID();
  mySimHit = 0;
}

void Bcm1fSD::update(const BeginOfTrack *bot) {

  const G4Track* gTrack = (*bot)();
  pname = gTrack->GetDynamicParticle()->GetDefinition()->GetParticleName();
}

void Bcm1fSD::clearHits() {
    slave->Initialize();
}

TrackInformation* Bcm1fSD::getOrCreateTrackInformation( const G4Track* gTrack) {
  G4VUserTrackInformation* temp = gTrack->GetUserInformation();
  if (temp == 0){
    edm::LogError("Bcm1fSD") <<" ERROR: no G4VUserTrackInformation available";
    abort();
  }else{
    TrackInformation* info = dynamic_cast<TrackInformation*>(temp);
    if (info == 0){
      edm::LogError("Bcm1fSD") <<" ERROR: TkSimTrackSelection: the UserInformation does not appear to be a TrackInformation";
      abort();
    }
    return info;
  }
}
