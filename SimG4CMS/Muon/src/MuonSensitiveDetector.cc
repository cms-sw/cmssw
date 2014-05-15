#include "SimG4CMS/Muon/interface/MuonSensitiveDetector.h"
#include "SimG4CMS/Muon/interface/MuonSlaveSD.h"
#include "SimG4CMS/Muon//interface/MuonEndcapFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonRpcFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonGemFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonMe0FrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "SimG4CMS/Muon/interface/SimHitPrinter.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"

#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonSimHitNumberingScheme.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

MuonSensitiveDetector::MuonSensitiveDetector(std::string name, 
					     const DDCompactView & cpv,
					     SensitiveDetectorCatalog & clg,
					     edm::ParameterSet const & p,
					     const SimTrackManager* manager) 
  : SensitiveTkDetector(name, cpv, clg, p),
    thePV(0), theHit(0), theDetUnitId(0), theTrackID(0), theManager(manager)
{
  edm::ParameterSet m_MuonSD = p.getParameter<edm::ParameterSet>("MuonSD");
  STenergyPersistentCut = m_MuonSD.getParameter<double>("EnergyThresholdForPersistency");//Default 1. GeV
  STallMuonsPersistent = m_MuonSD.getParameter<bool>("AllMuonsPersistent");
  printHits  = m_MuonSD.getParameter<bool>("PrintHits");
  
  //
  // Here simply create 1 MuonSlaveSD for the moment
  //  
  
  LogDebug("MuonSimDebug") << "create MuonSubDetector "<<name<<std::endl;

  detector = new MuonSubDetector(name);

  LogDebug("MuonSimDebug") << "create MuonFrameRotation"<<std::endl;

 if (detector->isEndcap()) {
   //    cout << "MuonFrameRotation create MuonEndcapFrameRotation"<<endl;
    theRotation=new MuonEndcapFrameRotation();
  } else if (detector->isRpc()) {
    //    cout << "MuonFrameRotation create MuonRpcFrameRotation"<<endl;
    theRotation=new MuonRpcFrameRotation( cpv );
  } else if (detector->isGem()) {
    //    cout << "MuonFrameRotation create MuonGemFrameRotation"<<endl;
    theRotation=new MuonGemFrameRotation( cpv );
  } else if (detector->isME0()) {
    //    cout << "MuonFrameRotation create MuonMe0FrameRotation"<<endl;
    theRotation=new MuonMe0FrameRotation( cpv );
  }  else {
    theRotation = 0;
  }
  LogDebug("MuonSimDebug") << "create MuonSlaveSD"<<std::endl;
  slaveMuon  = new MuonSlaveSD(detector,theManager);
  LogDebug("MuonSimDebug") << "create MuonSimHitNumberingScheme"<<std::endl;
  numbering  = new MuonSimHitNumberingScheme(detector, cpv);
  g4numbering = new MuonG4Numbering(cpv);
  

  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  std::vector<std::string>  lvNames = clg.logicalNames(name);
  this->Register();
  for (std::vector<std::string>::iterator it = lvNames.begin();  it != lvNames.end(); it++){
    LogDebug("MuonSimDebug") << name << " MuonSensitiveDetector:: attaching SD to LV " << *it << std::endl;
    this->AssignSD(*it);
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

void MuonSensitiveDetector::update(const BeginOfEvent * i){
  clearHits();

  //----- Initialize variables to check if two steps belong to same hit
  thePV = 0;
  theDetUnitId = 0;
  theTrackID = 0;

}

void MuonSensitiveDetector::update(const  ::EndOfEvent * ev)
{
  //slaveMuon->renumbering(theManager);
}


void MuonSensitiveDetector::clearHits()
{
  LogDebug("MuonSimDebug") << "MuonSensitiveDetector::clearHits"<<std::endl;
  slaveMuon->Initialize();
}

bool MuonSensitiveDetector::ProcessHits(G4Step * aStep, G4TouchableHistory * ROhist)
{
  LogDebug("MuonSimDebug") <<" MuonSensitiveDetector::ProcessHits "<<InitialStepPosition(aStep,WorldCoordinates)<<std::endl;

 // TimeMe t1( theHitTimer, false);

  if (aStep->GetTotalEnergyDeposit()>0.){
    // do not count neutrals that are killed by User Limits MinEKine
    if( aStep->GetTrack()->GetDynamicParticle()->GetCharge() != 0 ){
  
      if (newHit(aStep)) {
	saveHit();
	createHit(aStep);
      } else {
	updateHit(aStep);
      }    
      return true;
    } else {
      storeVolumeAndTrack(aStep);
      return false;
    }
  }
  return false;
}

uint32_t MuonSensitiveDetector::setDetUnitId(G4Step * aStep)
{ 
  //  G4VPhysicalVolume * pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);
  return numbering->baseNumberToUnitNumber(num);
}


Local3DPoint MuonSensitiveDetector::toOrcaRef(Local3DPoint in ,G4Step * s){
  if (theRotation !=0 ) {
    return theRotation->transformPoint(in,s);
  }
  return (in);
}

Local3DPoint MuonSensitiveDetector::toOrcaUnits(Local3DPoint in){
  return Local3DPoint(in.x()/cm,in.y()/cm,in.z()/cm);
}

Global3DPoint MuonSensitiveDetector::toOrcaUnits(Global3DPoint in){
  return Global3DPoint(in.x()/cm,in.y()/cm,in.z()/cm);
}

void MuonSensitiveDetector::storeVolumeAndTrack(G4Step * aStep) {
  G4VPhysicalVolume* pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  G4Track * t  = aStep->GetTrack();
  thePV=pv;
  theTrackID=t->GetTrackID();
}

bool MuonSensitiveDetector::newHit(G4Step * aStep){
  
  G4VPhysicalVolume* pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  G4Track * t  = aStep->GetTrack();
  uint32_t currentUnitId=setDetUnitId(aStep);
  unsigned int currentTrackID=t->GetTrackID();
  //unsigned int currentEventID=G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
  bool changed=((pv!=thePV) || 
		(currentUnitId!=theDetUnitId) || 
		(currentTrackID!=theTrackID));
  return changed;
}

void MuonSensitiveDetector::createHit(G4Step * aStep){

  G4Track * theTrack  = aStep->GetTrack(); 

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
  //  int theParticleType     = theTrack->GetDefinition()->GetPDGEncoding();
  int theParticleType     = myG4TrackToParticleID->particleID(theTrack);
  G4ThreeVector gmd  = aStep->GetPreStepPoint()->GetMomentumDirection();
  G4ThreeVector lmd = ((const G4TouchableHistory *)(aStep->GetPreStepPoint()->GetTouchable()))->GetHistory()
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

void MuonSensitiveDetector::updateHit(G4Step * aStep){
  //  float thePabs             = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
  //  Local3DPoint theEntryPoint= InitialStepPosition(aStep,LocalCoordinates);  


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

  if( theHit == 0 ){ 
    std::cerr << "!!ERRROR in MuonSensitiveDetector::updateHit. It is called when there is no hit " << std::endl;
  }

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
    theHit = 0; //set it to 0, because you are checking that is 0
  }

}

TrackInformation* MuonSensitiveDetector::getOrCreateTrackInformation( const G4Track* gTrack)
{
  G4VUserTrackInformation* temp = gTrack->GetUserInformation();
  if (temp == 0){
    std::cerr <<" ERROR: no G4VUserTrackInformation available"<<std::endl;
    abort();
  }else{
    TrackInformation* info = dynamic_cast<TrackInformation*>(temp);
    if (info ==0){
      std::cerr <<" ERROR: TkSimTrackSelection: the UserInformation does not appear to be a TrackInformation"<<std::endl;
      abort();
    }
    return info;
  }
}

void MuonSensitiveDetector::EndOfEvent(G4HCofThisEvent*)
{
//  TimeMe t("MuonSensitiveDetector::EndOfEvent", false);
 // LogDebug("MuonSimDebug") << "MuonSensitiveDetector::EndOfEvent saving last hit en event " << std::endl;
  saveHit();
}


void MuonSensitiveDetector::fillHits(edm::PSimHitContainer& c, std::string n){
  //
  // do it once for low, once for High
  //

  if (slaveMuon->name() == n) c=slaveMuon->hits();

}

std::vector<std::string> MuonSensitiveDetector::getNames(){
  std::vector<std::string> temp;
  temp.push_back(slaveMuon->name());
  return temp;
}

Local3DPoint MuonSensitiveDetector::InitialStepPositionVsParent(G4Step * currentStep, G4int levelsUp) {
  
  G4StepPoint * preStepPoint = currentStep->GetPreStepPoint();
  G4ThreeVector globalCoordinates = preStepPoint->GetPosition();
  
  const G4TouchableHistory * theTouchable=(const G4TouchableHistory *)
    (preStepPoint->GetTouchable());

  G4int depth = theTouchable->GetHistory()->GetDepth();
  G4ThreeVector localCoordinates = theTouchable->GetHistory()
    ->GetTransform(depth-levelsUp).TransformPoint(globalCoordinates);
  
  return ConvertToLocal3DPoint(localCoordinates); 
}
 
Local3DPoint MuonSensitiveDetector::FinalStepPositionVsParent(G4Step * currentStep, G4int levelsUp) {
  
  G4StepPoint * postStepPoint = currentStep->GetPostStepPoint();
  G4StepPoint * preStepPoint  = currentStep->GetPreStepPoint();
  G4ThreeVector globalCoordinates = postStepPoint->GetPosition();
    
  const G4TouchableHistory * theTouchable = (const G4TouchableHistory *)
    (preStepPoint->GetTouchable());

  G4int depth = theTouchable->GetHistory()->GetDepth();
  G4ThreeVector localCoordinates = theTouchable->GetHistory()
    ->GetTransform(depth-levelsUp).TransformPoint(globalCoordinates);

  return ConvertToLocal3DPoint(localCoordinates); 
}
