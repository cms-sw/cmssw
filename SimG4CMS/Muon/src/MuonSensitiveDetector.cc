#include "SimG4CMS/Muon/interface/MuonSensitiveDetector.h"
#include "SimG4CMS/Muon/interface/MuonSlaveSD.h"
#include "SimG4CMS/Muon//interface/MuonEndcapFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonRPCFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonGEMFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonME0FrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include "SimG4CMS/Muon/interface/SimHitPrinter.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonSimHitNumberingScheme.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

//#define DebugLog

MuonSensitiveDetector::MuonSensitiveDetector(const std::string& name, 
					     const DDCompactView & cpv,
					     const SensitiveDetectorCatalog & clg,
					     edm::ParameterSet const & p,
					     const SimTrackManager* manager) 
  : SensitiveTkDetector(name, cpv, clg, p),
    thePV(nullptr), theHit(nullptr), theDetUnitId(0), newDetUnitId(0), 
    theTrackID(0), theManager(manager)
{
  edm::ParameterSet m_MuonSD = p.getParameter<edm::ParameterSet>("MuonSD");
  ePersistentCutGeV = m_MuonSD.getParameter<double>("EnergyThresholdForPersistency")/CLHEP::GeV;//Default 1. GeV
  allMuonsPersistent = m_MuonSD.getParameter<bool>("AllMuonsPersistent");
  printHits  = m_MuonSD.getParameter<bool>("PrintHits");
  
  //
  // Here simply create 1 MuonSlaveSD for the moment
  //  
  LogDebug("MuonSimDebug") << "create MuonSubDetector "<<name;
  detector = new MuonSubDetector(name);

  //The constants take time to calculate and are needed by many helpers
  MuonDDDConstants constants(cpv);
  G4String sdet = "unknown";
  if (detector->isEndcap()) {
    theRotation=new MuonEndcapFrameRotation();
    sdet = "Endcap";
  } else if (detector->isRPC()) {
    theRotation=new MuonRPCFrameRotation( constants );
    sdet = "RPC";
  } else if (detector->isGEM()) {
    theRotation=new MuonGEMFrameRotation( constants );
    sdet = "GEM";
  } else if (detector->isME0()) {
    theRotation=new MuonME0FrameRotation( constants );
    sdet = "ME0";
  } else {
    theRotation = new MuonFrameRotation();
  }
  slaveMuon  = new MuonSlaveSD(detector,theManager);
  numbering  = new MuonSimHitNumberingScheme(detector, constants);
  g4numbering = new MuonG4Numbering(constants);
  
  if (printHits) {
    thePrinter = new SimHitPrinter("HitPositionOSCAR.dat");
  }

  edm::LogVerbatim("MuonSensitiveDetector") 
    << " of type " << sdet << " <" << GetName() 
    << "> EnergyThresholdForPersistency(GeV) " << ePersistentCutGeV/CLHEP::GeV 
    << " allMuonsPersistent: " << allMuonsPersistent;
    
  theG4ProcessTypeEnumerator = new G4ProcessTypeEnumerator;
}

MuonSensitiveDetector::~MuonSensitiveDetector() { 
  delete g4numbering;
  delete numbering;
  delete slaveMuon;
  delete theRotation;
  delete detector;
  delete theG4ProcessTypeEnumerator;
}

void MuonSensitiveDetector::update(const BeginOfEvent * i){
  clearHits();
  //----- Initialize variables to check if two steps belong to same hit
  thePV = nullptr;
  theDetUnitId = 0;
  theTrackID = 0;
}

void MuonSensitiveDetector::clearHits()
{
  LogDebug("MuonSimDebug") << "MuonSensitiveDetector::clearHits";
  slaveMuon->Initialize();
}

bool MuonSensitiveDetector::ProcessHits(G4Step * aStep, G4TouchableHistory * ROhist)
{
  LogDebug("MuonSimDebug") <<" MuonSensitiveDetector::ProcessHits "
			   <<InitialStepPosition(aStep,WorldCoordinates);

  if (aStep->GetTotalEnergyDeposit()>0.){
    newDetUnitId = setDetUnitId(aStep);

    // do not count neutrals that are killed by User Limits MinEKine
    //---VI: this is incorrect, neutral particle, like neutron may have local
    //       energy deposit, which potentially may make a hit
    if( aStep->GetTrack()->GetDynamicParticle()->GetCharge() != 0 ){
  
      if (newHit(aStep)) {
	saveHit();
	createHit(aStep);
      } else {
	updateHit(aStep);
      } 
    } else {
      thePV = aStep->GetPreStepPoint()->GetPhysicalVolume();
      theTrackID = aStep->GetTrack()->GetTrackID();
      theDetUnitId = newDetUnitId;
    }
  }
  return true;
}

uint32_t MuonSensitiveDetector::setDetUnitId(const G4Step * aStep)
{ 
  MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);

#ifdef DebugLog
  std::stringstream MuonBaseNumber; 
  MuonBaseNumber << "MuonNumbering :: number of levels = "<<num.getLevels()<<std::endl;
  MuonBaseNumber << "Level \t SuperNo \t BaseNo"<<std::endl;
  for (int level=1;level<=num.getLevels();level++) {
    MuonBaseNumber << level << " \t " << num.getSuperNo(level)
	      << " \t " << num.getBaseNo(level) << std::endl;
  }
  std::string MuonBaseNumbr = MuonBaseNumber.str();

  LogDebug("MuonSimDebug") <<"MuonSensitiveDetector::setDetUnitId :: "<<MuonBaseNumbr;
  LogDebug("MuonSimDebug") <<"MuonSensitiveDetector::setDetUnitId :: MuonDetUnitId = "
			   <<(numbering->baseNumberToUnitNumber(num));
#endif
  return numbering->baseNumberToUnitNumber(num);
}

bool MuonSensitiveDetector::newHit(const G4Step * aStep){

  return (!theHit || (aStep->GetTrack()->GetTrackID() != theTrackID) 
	  || (aStep->GetPreStepPoint()->GetPhysicalVolume() != thePV)
	  || newDetUnitId != theDetUnitId);
}

void MuonSensitiveDetector::createHit(const G4Step * aStep){

  Local3DPoint theEntryPoint;
  Local3DPoint theExitPoint;

  if (detector->isBarrel()) {
    // 1 levels up
    theEntryPoint = cmsUnits(theRotation->transformPoint(InitialStepPositionVsParent(aStep,1),aStep));
    theExitPoint  = cmsUnits(theRotation->transformPoint(FinalStepPositionVsParent(aStep,1),aStep));
  } else if (detector->isEndcap()) {
    // save local z at current level
    theEntryPoint = theRotation->transformPoint(InitialStepPosition(aStep,LocalCoordinates),aStep);
    theExitPoint  = theRotation->transformPoint(FinalStepPosition(aStep,LocalCoordinates),aStep);
    float zentry  = theEntryPoint.z();
    float zexit   = theExitPoint.z();
    // 4 levels up
    Local3DPoint tempEntry = theRotation->transformPoint(InitialStepPositionVsParent(aStep,4),aStep);
    Local3DPoint tempExit  = theRotation->transformPoint(FinalStepPositionVsParent(aStep,4),aStep);
    // reset local z from z wrt deep-parent volume to z wrt low-level volume
    theEntryPoint = cmsUnits(Local3DPoint( tempEntry.x(), tempEntry.y(), zentry ));
    theExitPoint  = cmsUnits(Local3DPoint( tempExit.x(),  tempExit.y(),  zexit ));
  } else {
    theEntryPoint = cmsUnits(theRotation->transformPoint(InitialStepPosition(aStep,LocalCoordinates),aStep));
    theExitPoint  = cmsUnits(theRotation->transformPoint(FinalStepPosition(aStep,LocalCoordinates),aStep)); 
  }

  const G4Track* theTrack = aStep->GetTrack(); 
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();

  float thePabs       = preStepPoint->GetMomentum().mag()/CLHEP::GeV;
  float theTof        = preStepPoint->GetGlobalTime()/CLHEP::nanosecond;
  float theEnergyLoss = aStep->GetTotalEnergyDeposit()/CLHEP::GeV;
  int theParticleType = G4TrackToParticleID::particleID(theTrack);

  theDetUnitId = newDetUnitId;
  thePV = preStepPoint->GetPhysicalVolume();
  theTrackID = theTrack->GetTrackID();

  // convert momentum direction it to local frame
  const G4ThreeVector& gmd  = preStepPoint->GetMomentumDirection();
  G4ThreeVector lmd = static_cast<const G4TouchableHistory *>(preStepPoint->GetTouchable())->GetHistory()
      ->GetTopTransform().TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  lnmd = theRotation->transformPoint(lnmd, aStep);
  float theThetaAtEntry = lnmd.theta();
  float thePhiAtEntry = lnmd.phi();

  theHit = new UpdatablePSimHit(theEntryPoint,theExitPoint,thePabs,theTof,
				theEnergyLoss,theParticleType,theDetUnitId,
				theTrackID,theThetaAtEntry,thePhiAtEntry,
				theG4ProcessTypeEnumerator->processId(theTrack->GetCreatorProcess()));

  // Make track persistent
  int thePID = std::abs(theTrack->GetDefinition()->GetPDGEncoding());
  //---VI - in parameters cut in energy is declared but applied to momentum
  if(thePabs > ePersistentCutGeV || ( thePID == 13 && allMuonsPersistent ) ){
    TrackInformation* info = cmsTrackInformation(theTrack);
    info->storeTrack(true);
  }

#ifdef DebugLog
  edm::LogVerbatim("MuonSimDebug") <<"=== NEW Muon hit for "<< GetName() 
				   << " Edep(GeV)= " << theEnergyLoss
				   <<" " <<thePV->GetLogicalVolume()->GetName();
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = preStepPoint->GetProcessDefinedStep();
  G4String sss = "";
  if (p) sss += " POST PROCESS: " + p->GetProcessName();
  if (p2)sss += ";  PRE  PROCESS: " + p2->GetProcessName();
  if("" != sss) edm::LogVerbatim("MuonSimDebug") << sss;
  edm::LogVerbatim("MuonSimDebug") << " theta= " << theThetaAtEntry
				   << " phi= " << thePhiAtEntry
				   << " Pabs(GeV/c)  " << thePabs
				   << " Eloss(GeV)= " << theEnergyLoss
				   << " Tof(ns)=  " << theTof
				   << " trackID= " << theTrackID
				   << " detID= " << theDetUnitId
				   << "\n Local:  entry " << theEntryPoint 
				   << " exit " << theExitPoint
				   << " delta " << (theExitPoint-theEntryPoint)
				   << "\n Global: entry " << preStepPoint->GetPosition() 
				   << " exit " << aStep->GetPostStepPoint()->GetPosition();
#endif
}

void MuonSensitiveDetector::updateHit(const G4Step * aStep){

  Local3DPoint theExitPoint;

  if (detector->isBarrel()) {
    theExitPoint = cmsUnits(theRotation->transformPoint(FinalStepPositionVsParent(aStep,1),aStep));  
  } else if (detector->isEndcap()) {
    // save local z at current level
    theExitPoint = theRotation->transformPoint(FinalStepPosition(aStep,LocalCoordinates),aStep);
    float zexit  = theExitPoint.z();
    Local3DPoint tempExitPoint = theRotation->transformPoint(FinalStepPositionVsParent(aStep,4),aStep);
    theExitPoint = cmsUnits(Local3DPoint( tempExitPoint.x(), tempExitPoint.y(), zexit));
  } else {
    theExitPoint = cmsUnits(theRotation->transformPoint(FinalStepPosition(aStep,LocalCoordinates),aStep)); 
  }

  float theEnergyLoss = aStep->GetTotalEnergyDeposit()/CLHEP::GeV;  

  theHit->updateExitPoint(theExitPoint);
  theHit->addEnergyLoss(theEnergyLoss);

#ifdef DebugLog
  edm::LogVerbatim("MuonSimDebug") <<"=== NEW Update muon hit for "<< GetName() 
				   << " Edep(GeV)= " << theEnergyLoss
				   <<" " <<thePV->GetLogicalVolume()->GetName();
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = preStepPoint->GetProcessDefinedStep();
  G4String sss = "";
  if (p) sss += " POST PROCESS: " + p->GetProcessName();
  if (p2)sss += ";  PRE  PROCESS: " + p2->GetProcessName();
  if("" != sss) edm::LogVerbatim("MuonSimDebug") << sss;
  edm::LogVerbatim("MuonSimDebug") << " delEloss(GeV)= " << theEnergyLoss
				   << " Tof(ns)=  " << theTof
				   << " trackID= " << theTrackID
				   << " detID= " << theDetUnitId
				   << " exit " << theExitPoint;
#endif
}

void MuonSensitiveDetector::saveHit(){
  if (theHit) {
    if (printHits) {
      thePrinter->startNewSimHit(detector->name());
      thePrinter->printId(theHit->detUnitId());
      thePrinter->printLocal(theHit->entryPoint(),theHit->exitPoint());
    }
    // hit is included into hit collection
    slaveMuon->processHits(*theHit);
    delete theHit;
    theHit = nullptr;
  }
}

void MuonSensitiveDetector::EndOfEvent(G4HCofThisEvent*)
{
  saveHit();
}

void MuonSensitiveDetector::fillHits(edm::PSimHitContainer& cc, const std::string& hname){
  if (slaveMuon->name() == hname) { cc=slaveMuon->hits(); }
}

Local3DPoint MuonSensitiveDetector::InitialStepPositionVsParent(const G4Step * currentStep, G4int levelsUp) {
  
  const G4StepPoint * preStepPoint = currentStep->GetPreStepPoint();
  const G4ThreeVector& globalCoordinates = preStepPoint->GetPosition();
  
  const G4TouchableHistory * theTouchable=(const G4TouchableHistory *)
    (preStepPoint->GetTouchable());

  G4int depth = theTouchable->GetHistory()->GetDepth();
  G4ThreeVector localCoordinates = theTouchable->GetHistory()
    ->GetTransform(depth-levelsUp).TransformPoint(globalCoordinates);
  
  return ConvertToLocal3DPoint(localCoordinates); 
}
 
Local3DPoint MuonSensitiveDetector::FinalStepPositionVsParent(const G4Step* currentStep, G4int levelsUp) {
  
  const G4StepPoint * postStepPoint = currentStep->GetPostStepPoint();
  const G4StepPoint * preStepPoint  = currentStep->GetPreStepPoint();
  const G4ThreeVector& globalCoordinates = postStepPoint->GetPosition();
    
  const G4TouchableHistory * theTouchable = (const G4TouchableHistory *)
    (preStepPoint->GetTouchable());

  G4int depth = theTouchable->GetHistory()->GetDepth();
  G4ThreeVector localCoordinates = theTouchable->GetHistory()
    ->GetTransform(depth-levelsUp).TransformPoint(globalCoordinates);

  return ConvertToLocal3DPoint(localCoordinates); 
}
