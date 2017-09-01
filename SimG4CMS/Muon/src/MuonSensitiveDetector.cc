#include "SimG4CMS/Muon/interface/MuonSensitiveDetector.h"
#include "SimG4CMS/Muon/interface/MuonSlaveSD.h"
#include "SimG4CMS/Muon//interface/MuonEndcapFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonRPCFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonGEMFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonME0FrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "SimG4CMS/Muon/interface/SimHitPrinter.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"

#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonSimHitNumberingScheme.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4SystemOfUnits.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

MuonSensitiveDetector::MuonSensitiveDetector(const std::string& iname, 
					     const DDCompactView & cpv,
					     const SensitiveDetectorCatalog & clg,
					     edm::ParameterSet const & p,
					     const SimTrackManager* manager) 
  : SensitiveTkDetector(iname, cpv, clg, p),
    thePV(nullptr), theHit(nullptr), theDetUnitId(0), theTrackID(0), theManager(manager)
{
  edm::ParameterSet m_MuonSD = p.getParameter<edm::ParameterSet>("MuonSD");
  STenergyPersistentCut = m_MuonSD.getParameter<double>("EnergyThresholdForPersistency");//Default 1. GeV
  STallMuonsPersistent = m_MuonSD.getParameter<bool>("AllMuonsPersistent");
  printHits  = m_MuonSD.getParameter<bool>("PrintHits");
  
  //
  // Here simply create 1 MuonSlaveSD for the moment
  //  
  
  LogDebug("MuonSimDebug") << "create MuonSubDetector "<<iname;

  detector = new MuonSubDetector(iname);

  LogDebug("MuonSimDebug") << "create MuonFrameRotation"<<std::endl;

 //The constants take time to calculate and are needed by many helpers
 MuonDDDConstants constants(cpv);
 if (detector->isEndcap()) {
   //    cout << "MuonFrameRotation create MuonEndcapFrameRotation"<<endl;
    theRotation=new MuonEndcapFrameRotation();
  } else if (detector->isRPC()) {
    //    cout << "MuonFrameRotation create MuonRPCFrameRotation"<<endl;
    theRotation=new MuonRPCFrameRotation( constants );
  } else if (detector->isGEM()) {
    //    cout << "MuonFrameRotation create MuonGEMFrameRotation"<<endl;
    theRotation=new MuonGEMFrameRotation( constants );
  } else if (detector->isME0()) {
    //    cout << "MuonFrameRotation create MuonME0FrameRotation"<<endl;
    theRotation=new MuonME0FrameRotation( constants );
  }  else {
    theRotation = nullptr;
  }
  LogDebug("MuonSimDebug") << "create MuonSlaveSD";
  slaveMuon  = new MuonSlaveSD(detector,theManager);
  LogDebug("MuonSimDebug") << "create MuonSimHitNumberingScheme";
  numbering  = new MuonSimHitNumberingScheme(detector, constants);
  g4numbering = new MuonG4Numbering(constants);
  
  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  const std::vector<std::string>& lvNames = clg.logicalNames(iname);
  this->Register();
  for (auto & lvname : lvNames){
    LogDebug("MuonSimDebug") << iname << " MuonSensitiveDetector:: attaching SD to LV " << lvname;
    this->AssignSD(lvname);
  }

  if (printHits) {
    thePrinter = new SimHitPrinter("HitPositionOSCAR.dat");
  }

  LogDebug("MuonSimDebug") << "  EnergyThresholdForPersistency " << STenergyPersistentCut << " AllMuonsPersistent " <<  STallMuonsPersistent << std::endl;
    
  theG4ProcessTypeEnumerator = new G4ProcessTypeEnumerator;
  myG4TrackToParticleID = new G4TrackToParticleID;
}

MuonSensitiveDetector::~MuonSensitiveDetector() { 
  delete g4numbering;
  delete numbering;
  delete slaveMuon;
  delete theRotation;
  delete detector;

  delete theG4ProcessTypeEnumerator;
  
  delete myG4TrackToParticleID;
}

void MuonSensitiveDetector::update(const BeginOfEvent *){
  clearHits();

  //----- Initialize variables to check if two steps belong to same hit
  thePV = nullptr;
  theDetUnitId = 0;
  theTrackID = 0;

}

void MuonSensitiveDetector::clearHits()
{
  LogDebug("MuonSimDebug") << "MuonSensitiveDetector::clearHits"<<std::endl;
  slaveMuon->Initialize();
}

bool MuonSensitiveDetector::ProcessHits(G4Step * aStep, G4TouchableHistory * ROhist)
{
  LogDebug("MuonSimDebug") <<" MuonSensitiveDetector::ProcessHits "<<InitialStepPosition(aStep,WorldCoordinates)<<std::endl;

  bool res = true;
  if (aStep->GetTotalEnergyDeposit()>0.){
    // do not count neutrals that are killed by User Limits MinEKine
    if( aStep->GetTrack()->GetDynamicParticle()->GetCharge() != 0 ){
  
      if (newHit(aStep)) {
	saveHit();
	createHit(aStep);
      } else {
	updateHit(aStep);
      }    
    } else {
      storeVolumeAndTrack(aStep);
      res = false;
    }
  }
  return res;
}

uint32_t MuonSensitiveDetector::setDetUnitId(const G4Step * aStep)
{ 
  MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);

  std::stringstream MuonBaseNumber; 
  // LogDebug :: Print MuonBaseNumber
  MuonBaseNumber << "MuonNumbering :: number of levels = "<<num.getLevels()<<"\n";
  MuonBaseNumber << "Level \t SuperNo \t BaseNo"<<"\n";
  for (int level=1; level<=num.getLevels(); ++level) {
    MuonBaseNumber << level << " \t " << num.getSuperNo(level)
	      << " \t " << num.getBaseNo(level) << "\n";
  }
  std::string MuonBaseNumbr = MuonBaseNumber.str();

  LogDebug("MuonSimDebug") <<"MuonSensitiveDetector::setDetUnitId :: "<<MuonBaseNumbr
			   <<" MuonDetUnitId = "<<(numbering->baseNumberToUnitNumber(num));
  return numbering->baseNumberToUnitNumber(num);
}


Local3DPoint MuonSensitiveDetector::toOrcaRef(const Local3DPoint& in ,const G4Step * step){
  return std::move(theRotation ? theRotation->transformPoint(in,step) : in);
}

Local3DPoint MuonSensitiveDetector::toOrcaUnits(const Local3DPoint& in){
  return std::move(Local3DPoint(in.x()/cm,in.y()/cm,in.z()/cm));
}

Global3DPoint MuonSensitiveDetector::toOrcaUnits(const Global3DPoint& in){
  return std::move(Global3DPoint(in.x()/cm,in.y()/cm,in.z()/cm));
}

void MuonSensitiveDetector::storeVolumeAndTrack(const G4Step * aStep) {
  thePV = aStep->GetPreStepPoint()->GetPhysicalVolume();
  theTrackID = aStep->GetTrack()->GetTrackID();
}

bool MuonSensitiveDetector::newHit(const G4Step * aStep){
  
  const G4VPhysicalVolume* pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  uint32_t currentUnitId=setDetUnitId(aStep);
  LogDebug("MuonSimDebug") <<"MuonSensitiveDetector::newHit :: currentUnitId = "<<currentUnitId;
  unsigned int currentTrackID = aStep->GetTrack()->GetTrackID();

  return ((pv!=thePV) || (currentUnitId!=theDetUnitId) || (currentTrackID!=theTrackID));
}

void MuonSensitiveDetector::createHit(const G4Step * aStep){

  const G4Track * theTrack  = aStep->GetTrack(); 

  Local3DPoint theEntryPoint;
  Local3DPoint theExitPoint;

  if (detector->isBarrel()) {
    theEntryPoint= toOrcaUnits(toOrcaRef(InitialStepPositionVsParent(aStep,1),aStep)); // 1 level up
    theExitPoint= toOrcaUnits(toOrcaRef(FinalStepPositionVsParent(aStep,1),aStep));  
  } else if (detector->isEndcap()) {
    // save local z at current level
    theEntryPoint= toOrcaUnits(toOrcaRef(InitialStepPosition(aStep,LocalCoordinates),aStep));
    theExitPoint= toOrcaUnits(toOrcaRef(FinalStepPosition(aStep,LocalCoordinates),aStep));
    float zentry = theEntryPoint.z();
    float zexit = theExitPoint.z();
    Local3DPoint tempEntryPoint= toOrcaUnits(toOrcaRef(InitialStepPositionVsParent(aStep,4),aStep)); // 4 levels up
    Local3DPoint tempExitPoint= toOrcaUnits(toOrcaRef(FinalStepPositionVsParent(aStep,4),aStep));
    // reset local z from z wrt deep-parent volume to z wrt low-level volume
    theEntryPoint = Local3DPoint( tempEntryPoint.x(), tempEntryPoint.y(), zentry );
    theExitPoint  = Local3DPoint( tempExitPoint.x(), tempExitPoint.y(), zexit );
  } else {
    theEntryPoint= toOrcaUnits(toOrcaRef(InitialStepPosition(aStep,LocalCoordinates),aStep));
    theExitPoint= toOrcaUnits(toOrcaRef(FinalStepPosition(aStep,LocalCoordinates),aStep)); 
  }

  float thePabs             = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
  float theTof              = aStep->GetPreStepPoint()->GetGlobalTime()/nanosecond;
  float theEnergyLoss       = aStep->GetTotalEnergyDeposit()/GeV;
  int theParticleType     = myG4TrackToParticleID->particleID(theTrack);
  G4ThreeVector gmd  = aStep->GetPreStepPoint()->GetMomentumDirection();
  G4ThreeVector lmd = aStep->GetPreStepPoint()->GetTouchable()->GetHistory()
    ->GetTopTransform().TransformAxis(gmd);
  Local3DPoint lnmd = toOrcaRef(ConvertToLocal3DPoint(lmd),aStep);
  float theThetaAtEntry = lnmd.theta();
  float thePhiAtEntry = lnmd.phi();

  storeVolumeAndTrack( aStep );
  theDetUnitId              = setDetUnitId(aStep);

  Global3DPoint theGlobalPos;
  if (printHits) {   
    Local3DPoint theGlobalHelp = InitialStepPosition(aStep,WorldCoordinates);
    theGlobalEntry = toOrcaUnits(Global3DPoint (theGlobalHelp.x(),theGlobalHelp.y(),theGlobalHelp.z()));

    G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
    const G4TouchableHistory * theTouchable=(const G4TouchableHistory *)
                                            (preStepPoint->GetTouchable());
    theGlobalHelp=ConvertToLocal3DPoint(theTouchable->GetTranslation());
    theGlobalPos = toOrcaUnits(Global3DPoint (theGlobalHelp.x(),theGlobalHelp.y(),theGlobalHelp.z()));
    //    const G4RotationMatrix * theGlobalRot = theTouchable->GetRotation();
  }
  
  LogDebug("MuonSimDebug") << "MuonSensitiveDetector::createHit UpdatablePSimHit"<<std::endl;

  theHit = new UpdatablePSimHit(theEntryPoint,theExitPoint,thePabs,theTof,
                  theEnergyLoss,theParticleType,theDetUnitId,
                  theTrackID,theThetaAtEntry,thePhiAtEntry,
                  theG4ProcessTypeEnumerator->processId(theTrack->GetCreatorProcess()));


  LogDebug("MuonSimDebug") <<"=== NEW ==================> ELOSS   = "<<theEnergyLoss<<" "
       <<thePV->GetLogicalVolume()->GetName()<<std::endl;
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = aStep->GetPreStepPoint()->GetProcessDefinedStep();
  if (p)
    LogDebug("MuonSimDebug") <<" POST PROCESS = "<<p->GetProcessName()<<std::endl;
  if (p2)
    LogDebug("MuonSimDebug") <<" PRE  PROCESS = "<<p2->GetProcessName()<<std::endl;
  LogDebug("MuonSimDebug") << "newhit theta " << theThetaAtEntry<<std::endl;
  LogDebug("MuonSimDebug") << "newhit phi   " << thePhiAtEntry<<std::endl;
  LogDebug("MuonSimDebug") << "newhit pabs  " << thePabs<<std::endl;
  LogDebug("MuonSimDebug") << "newhit tof   " << theTof<<std::endl;
  LogDebug("MuonSimDebug") << "newhit track " << theTrackID<<std::endl;
  LogDebug("MuonSimDebug") << "newhit entry " << theEntryPoint<<std::endl;
  LogDebug("MuonSimDebug") << "newhit exit  " << theExitPoint<<std::endl;
  LogDebug("MuonSimDebug") << "newhit eloss " << theEnergyLoss << std::endl;
  LogDebug("MuonSimDebug") << "newhit detid " << theDetUnitId<<std::endl;
  LogDebug("MuonSimDebug") << "newhit delta " << (theExitPoint-theEntryPoint)<<std::endl;
  LogDebug("MuonSimDebug") << "newhit deltu " << (theExitPoint-theEntryPoint).unit();
  LogDebug("MuonSimDebug") << " " << (theExitPoint-theEntryPoint).mag()<<std::endl;
  LogDebug("MuonSimDebug") << "newhit glob  " << theGlobalEntry<<std::endl;
  LogDebug("MuonSimDebug") << "newhit dpos  " << theGlobalPos<<std::endl;
  LogDebug("MuonSimDebug") << "newhit drot  " << std::endl;
  //  theGlobalRot->print(LogDebug("MuonSimDebug"));


  //
  //----- SimTracks: Make it persistent?
  //
  int thePID = theTrack->GetDefinition()->GetPDGEncoding();
  LogDebug("MuonSimDebug") << " checking simtrack " << thePID << " " << thePabs << " STenergyPersistentCut " << STenergyPersistentCut << std::endl;

  if( thePabs*GeV > STenergyPersistentCut 
      || ( abs(thePID) == 13 && STallMuonsPersistent ) ){
    TrackInformation* info = getOrCreateTrackInformation(theTrack);
    LogDebug("MuonSimDebug") <<" track leaving hit in muons made selected for persistency"<<std::endl;

    info->storeTrack(true);
  }
     
}

void MuonSensitiveDetector::updateHit(const G4Step * aStep){

  Local3DPoint theExitPoint;

  if (detector->isBarrel()) {
    theExitPoint= toOrcaUnits(toOrcaRef(FinalStepPositionVsParent(aStep,1),aStep));  
  } else if (detector->isEndcap()) {
    // save local z at current level
    theExitPoint= toOrcaUnits(toOrcaRef(FinalStepPosition(aStep,LocalCoordinates),aStep));
    float zexit = theExitPoint.z();
    Local3DPoint tempExitPoint= toOrcaUnits(toOrcaRef(FinalStepPositionVsParent(aStep,4),aStep));
    theExitPoint  = Local3DPoint( tempExitPoint.x(), tempExitPoint.y(), zexit );
  } else {
    theExitPoint= toOrcaUnits(toOrcaRef(FinalStepPosition(aStep,LocalCoordinates),aStep)); 
  }

  float theEnergyLoss = aStep->GetTotalEnergyDeposit()/GeV;  

  theHit->updateExitPoint(theExitPoint);
  theHit->addEnergyLoss(theEnergyLoss);

  LogDebug("MuonSimDebug") <<"=== UPDATE ===============> ELOSS   = "<<theEnergyLoss<<" "
       <<thePV->GetLogicalVolume()->GetName()<<std::endl;
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = aStep->GetPreStepPoint()->GetProcessDefinedStep();
  if (p)
    LogDebug("MuonSimDebug") <<" POST PROCESS = "<<p->GetProcessName()<<std::endl;
  if (p2)
    LogDebug("MuonSimDebug") <<" PRE  PROCESS = "<<p2->GetProcessName()<<std::endl;
  LogDebug("MuonSimDebug") << "updhit exit  " << theExitPoint<<std::endl;
  LogDebug("MuonSimDebug") << "updhit eloss " << theHit->energyLoss() <<std::endl;
  LogDebug("MuonSimDebug") << "updhit detid " << theDetUnitId<<std::endl;
  LogDebug("MuonSimDebug") << "updhit delta " << (theExitPoint-theHit->entryPoint())<<std::endl;
  LogDebug("MuonSimDebug") << "updhit deltu " << (theExitPoint-theHit->entryPoint()).unit();
  LogDebug("MuonSimDebug") << " " << (theExitPoint-theHit->entryPoint()).mag()<<std::endl; 

}

void MuonSensitiveDetector::saveHit(){

  if (theHit) {
    if (printHits) {
      thePrinter->startNewSimHit(detector->name());
      thePrinter->printId(theHit->detUnitId());
      //      thePrinter->printTrack(theHit->trackId());
      //thePrinter->printPabs(theHit->pabs());
      //thePrinter->printEloss(theHit->energyLoss());
      thePrinter->printLocal(theHit->entryPoint(),theHit->exitPoint());
      thePrinter->printGlobal(theGlobalEntry);
    }
    slaveMuon->processHits(*theHit);
    // seems the hit does not want to be deleted
    // done by the hit collection?
    delete theHit;
    theHit = nullptr; 
  }
}

TrackInformation* MuonSensitiveDetector::getOrCreateTrackInformation( const G4Track* gTrack)
{
  G4VUserTrackInformation* temp = gTrack->GetUserInformation();
  if (temp == nullptr){
    throw SimG4Exception("MuonSensitiveDetector: no G4VUserTrackInformation available");
  }
  TrackInformation* info = dynamic_cast<TrackInformation*>(temp);
  if (info == nullptr){
    throw SimG4Exception("MuonSensitiveDetector: the UserInformation does not appear to be a TrackInformation");
  }
  return info;
}

void MuonSensitiveDetector::EndOfEvent(G4HCofThisEvent*)
{
  saveHit();
}

void MuonSensitiveDetector::fillHits(edm::PSimHitContainer& chit, const std::string& nhit){
  if (slaveMuon->name() == nhit) {chit=slaveMuon->hits(); }
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
 
Local3DPoint MuonSensitiveDetector::FinalStepPositionVsParent(const G4Step * currentStep, G4int levelsUp) {
  
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
