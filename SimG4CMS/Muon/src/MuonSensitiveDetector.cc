#include "SimG4CMS/Muon/interface/MuonSensitiveDetector.h"
#include "SimG4CMS/Muon/interface/MuonSlaveSD.h"
#include "Geometry/MuonBaseAlgo/interface/MuonSubDetector.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

#include "SimG4CMS/Muon/interface/SimHitPrinter.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"

#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "Geometry/MuonBaseAlgo/interface/MuonBaseNumber.h"
#include "Geometry/MuonBaseAlgo/interface/MuonSimHitNumberingScheme.h"

#include "SimG4Core/Geometry/interface/SDCatalog.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"

#include <iostream>

#define DEBUG
#define DEBUGST

MuonSensitiveDetector::MuonSensitiveDetector(std::string name) 
  : SensitiveTkDetector(name),
    thePV(0), theHit(0), theDetUnitId(0), theTrackID(0) 
{

  //
  // Here simply create 1 MuonSlaveSD for the moment
  //  
  
#ifdef DEBUG 
  std::cout << "create MuonSubDetector "<<name<<std::endl;
#endif
  detector = new MuonSubDetector(name);
#ifdef DEBUG 
  std::cout << "create MuonSlaveSD"<<std::endl;
#endif
  slaveMuon  = new MuonSlaveSD(detector);
#ifdef DEBUG 
  std::cout << "create MuonSimHitNumberingScheme"<<std::endl;
#endif
  numbering  = new MuonSimHitNumberingScheme(detector);
  g4numbering = new MuonG4Numbering;
  

  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  std::vector<std::string>  lvNames= SensitiveDetectorCatalog::instance()->logicalNames(name);
  this->Register();
  for (std::vector<std::string>::iterator it = lvNames.begin();  it != lvNames.end(); it++){
#ifdef DEBUG
    std::cout << name << " MuonSensitiveDetector:: attaching SD to LV " << *it << std::endl;
#endif
    this->AssignSD(*it);
  }

//  static SimpleConfigurable<bool>
//    pNumberingPrintHits(0,"MuonNumbering:PrintHits");
//  thePrintHits = pNumberingPrintHits.value();
  thePrintHits = 0;
  if (thePrintHits) {
    thePrinter = new SimHitPrinter("HitPositionOSCAR.dat");
  }

  /*
// timer initialization
  static bool on = 
    SimpleConfigurable<bool>(false,"MuonSensitiveDetector:DetailedTiming").value();
  if (on) {
    std::string trname("MuonSD:");
    theHitTimer.init( trname + name + ":hits", true);
  }
  else {
    theHitTimer.init( "MuonSensitiveDetector:hits", true);
  }
*/

  //----- Parameters for SimTracks creation and persistent storage
//  STenergyPersistentCut = SimpleConfigurable<float>(1*GeV,"MuonSimTrackSelection:EnergyThresholdForPersistency");
  STenergyPersistentCut = 1*GeV;
// STallMuonsPersistent = SimpleConfigurable<bool>(true,"MuonSimTrackSelection:AllMuonsPersistent");
  STallMuonsPersistent = true;

#ifdef DEBUGST
    std::cout << "  EnergyThresholdForPersistency " << STenergyPersistentCut << " AllMuonsPersistent " <<  STallMuonsPersistent << std::endl;
#endif
    
    theG4ProcessTypeEnumerator = new G4ProcessTypeEnumerator;
    myG4TrackToParticleID = new G4TrackToParticleID;

}


MuonSensitiveDetector::~MuonSensitiveDetector() { 
  //  saveHit();
  delete g4numbering;
  delete numbering;
  delete slaveMuon;
  delete detector;

  delete theG4ProcessTypeEnumerator;
  
  delete myG4TrackToParticleID;
}

void MuonSensitiveDetector::upDate(const BeginOfEvent * i){
  clearHits();

  //----- Initialize variables to check if two steps belong to same hit
  thePV = 0;
  theDetUnitId = 0;
  theTrackID = 0;

}

void MuonSensitiveDetector::clearHits()
{
#ifdef DEBUG 
  std::cout << "MuonSensitiveDetector::clearHits"<<std::endl;
#endif
  slaveMuon->clearHits();
}

bool MuonSensitiveDetector::ProcessHits(G4Step * aStep, G4TouchableHistory * ROhist)
{
#ifdef DEBUG
  std::cout <<" MuonSensitiveDetector::ProcessHits "<<InitialStepPosition(aStep,WorldCoordinates)<<std::endl;
#endif

 // TimeMe t1( theHitTimer, false);

  if (aStep->GetTotalEnergyDeposit()>0.){
    // do not count neutrals that are killed by User Limits MinEKine
    if( aStep->GetTrack()->GetDynamicParticle()->GetCharge() != 0 ){
  
      //----- Discard hits that would be produced where chimneys are
      if( hitInChimney(aStep) ) return false;
    
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

bool MuonSensitiveDetector::hitInChimney(G4Step * aStep)
{
  bool inChimney = false;
  //This should go to DDD.....
  double zlowRPC = 142.*cm;
  double zuppRPC = 185.5*cm;
  double zlowDT = 142.*cm;
  double zuppDT = 188.8*cm;

  G4ThreeVector pos = aStep->GetPreStepPoint()->GetPosition();
  G4double Zabs = fabs( pos.z() );

  if( Zabs < zlowDT && Zabs > zuppDT ) return false;


  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  //----- If step is in RPC sensitive, second mother will be MB*S, if it is in DT sensitive, third mother will be MB*S
  G4String mName;
  G4int chambNo = -1, wheelNo = -1;
  G4bool stepToCheck = false;
  G4double zlow = -1, zupp = -1;

  for( int ii = 0; ii < std::min(4,touch->GetHistoryDepth()); ii++ ){
    mName = touch->GetVolume(ii)->GetName();
    if( mName.length() == 4 ){
      if( mName.substr(0,2) == "MB" && mName.substr(3,1) == "S" ) {
        if(ii == 2 ){
          //--- Hit in RPC
          zlow = zlowRPC; 
          zupp = zuppRPC;
        } else if(ii == 3 ){
          //--- Hit in DT
          zlow = zlowDT; 
          zupp = zuppDT;
        }
        stepToCheck = true;
        chambNo = touch->GetReplicaNumber(ii);
        wheelNo = touch->GetReplicaNumber(ii+1);
      }
    }
  }
  
  //----- If step in MB*S, check it
  if( stepToCheck ) {
    G4ThreeVector pos = aStep->GetPreStepPoint()->GetPosition();

    //--- Check first the Z is within limits
    if( Zabs < zlow || Zabs > zupp ) {
      return false;
    }
    //--- Check if it is inside chambers touched by chimneys
    if( wheelNo%10 == 2 ) { // wheel +1
      G4int chamberRingID = (chambNo/100)%10;
      if( chamberRingID == 1 || chamberRingID == 2 || chamberRingID == 3 ){
	G4int chamberPhiID = chambNo%100;
	if( chamberPhiID == 4 ) inChimney = true;
      } else if( chamberRingID == 5){
	G4int chamberPhiID = chambNo%100;
	if( chamberPhiID == 1 || chamberPhiID == 2 ) inChimney = true;
      }
    } else if( wheelNo%10 == 3 ) {    // wheel -1
      G4int chamberRingID = (chambNo/100)%10;
      if( chamberRingID == 1 || chamberRingID == 2 || chamberRingID == 3 || chamberRingID == 4 ){
	G4int chamberPhiID = chambNo%100;
	if( chamberPhiID == 3 ) inChimney = true;
      }
    }

#ifdef DEBUG
#endif
  }
  return inChimney;
}

int MuonSensitiveDetector::SetDetUnitId(G4Step * aStep)
{ 
  //  G4VPhysicalVolume * pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);
  return numbering->baseNumberToUnitNumber(num);
}

Local3DPoint MuonSensitiveDetector::toOrcaUnits(Local3DPoint in){
  return Local3DPoint(in.x()/cm,in.y()/cm,in.z()/cm);
}

Global3DPoint MuonSensitiveDetector::toOrcaUnits(Global3DPoint in){
  return Global3DPoint(in.x()/cm,in.y()/cm,in.z()/cm);
}

void MuonSensitiveDetector::storeVolumeAndTrack(G4Step * aStep)
{
  G4VPhysicalVolume* pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  G4Track * t  = aStep->GetTrack();
  thePV=pv;
  theTrackID=t->GetTrackID();
}

bool MuonSensitiveDetector::newHit(G4Step * aStep){
  
  G4VPhysicalVolume* pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  G4Track * t  = aStep->GetTrack();
  int currentUnitId=SetDetUnitId(aStep);
  unsigned int currentTrackID=t->GetTrackID();
  //unsigned int currentEventID=G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
  bool changed=((pv!=thePV) || 
		(currentUnitId!=theDetUnitId) || 
		(currentTrackID!=theTrackID));
  return changed;
}

void MuonSensitiveDetector::createHit(G4Step * aStep){

  G4Track * theTrack  = aStep->GetTrack(); 

  Local3DPoint theEntryPoint= toOrcaUnits(InitialStepPosition(aStep,LocalCoordinates));  
  Local3DPoint theExitPoint = toOrcaUnits(FinalStepPosition(aStep,LocalCoordinates)); 
  float thePabs             = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
  float theTof              = aStep->GetPreStepPoint()->GetGlobalTime()/nanosecond;
  float theEnergyLoss       = aStep->GetTotalEnergyDeposit()/GeV;
  //  short theParticleType     = theTrack->GetDefinition()->GetPDGEncoding();
  short theParticleType     = myG4TrackToParticleID->particleID(theTrack);
  G4ThreeVector gmd  = aStep->GetPreStepPoint()->GetMomentumDirection();
  G4ThreeVector lmd = ((G4TouchableHistory *)(aStep->GetPreStepPoint()->GetTouchable()))->GetHistory()
    ->GetTopTransform().TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  float theThetaAtEntry = lnmd.theta();
  float thePhiAtEntry = lnmd.phi();

  storeVolumeAndTrack( aStep );
  theDetUnitId              = SetDetUnitId(aStep);

#ifdef DEBUG
  Global3DPoint theGlobalPos;
  const G4RotationMatrix * theGlobalRot;
#endif
  if (thePrintHits) {   
    Local3DPoint theGlobalHelp = InitialStepPosition(aStep,WorldCoordinates);
    theGlobalEntry = toOrcaUnits(Global3DPoint (theGlobalHelp.x(),theGlobalHelp.y(),theGlobalHelp.z()));
#ifdef DEBUG
    G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
    G4TouchableHistory * theTouchable=(G4TouchableHistory *)
                                      (preStepPoint->GetTouchable());
    theGlobalHelp=ConvertToLocal3DPoint(theTouchable->GetTranslation());
    theGlobalPos = toOrcaUnits(Global3DPoint (theGlobalHelp.x(),theGlobalHelp.y(),theGlobalHelp.z()));
    theGlobalRot = theTouchable->GetRotation();
#endif
  }
  
  
#ifdef DEBUG 
  std::cout << "MuonSensitiveDetector::createHit UpdatablePSimHit"<<std::endl;
#endif
  theHit = new UpdatablePSimHit(theEntryPoint,theExitPoint,thePabs,theTof,
                  theEnergyLoss,theParticleType,theDetUnitId,
                  theTrackID,theThetaAtEntry,thePhiAtEntry,
                  theG4ProcessTypeEnumerator->processId(theTrack->GetCreatorProcess()));

#ifdef DEBUG      
  std::cout <<"=== NEW ==================> ELOSS   = "<<theEnergyLoss<<" "
       <<thePV->GetLogicalVolume()->GetName()<<std::endl;
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = aStep->GetPreStepPoint()->GetProcessDefinedStep();
  if (p)
    std::cout <<" POST PROCESS = "<<p->GetProcessName()<<std::endl;
  if (p2)
    std::cout <<" PRE  PROCESS = "<<p2->GetProcessName()<<std::endl;
  std::cout << "newhit theta " << theThetaAtEntry<<std::endl;
  std::cout << "newhit phi   " << thePhiAtEntry<<std::endl;
  std::cout << "newhit pabs  " << thePabs<<std::endl;
  std::cout << "newhit tof   " << theTof<<std::endl;
  std::cout << "newhit track " << theTrackID<<std::endl;
  std::cout << "newhit entry " << theEntryPoint<<std::endl;
  std::cout << "newhit exit  " << theExitPoint<<std::endl;
  std::cout << "newhit eloss " << theEnergyLoss << std::endl;
  std::cout << "newhit detid " << theDetUnitId<<std::endl;
  std::cout << "newhit delta " << (theExitPoint-theEntryPoint)<<std::endl;
  std::cout << "newhit deltu " << (theExitPoint-theEntryPoint).unit();
  std::cout << " " << (theExitPoint-theEntryPoint).mag()<<std::endl;
  std::cout << "newhit glob  " << theGlobalEntry<<std::endl;
  std::cout << "newhit dpos  " << theGlobalPos<<std::endl;
  std::cout << "newhit drot  " << std::endl;
  //  theGlobalRot->print(std::cout);
#endif

  //
  //----- SimTracks: Make it persistent?
  //
  int thePID = theTrack->GetDefinition()->GetPDGEncoding();
#ifdef DEBUGST
  std::cout << " checking simtrack " << thePID << " " << thePabs << " STenergyPersistentCut " << STenergyPersistentCut << std::endl;
#endif

  if( thePabs*GeV > STenergyPersistentCut 
      || ( abs(thePID) == 13 && STallMuonsPersistent ) ){
    TrackInformation* info = getOrCreateTrackInformation(theTrack);
#ifdef DEBUGST
    std::cout <<" track leaving hit in muons made selected for persistency"<<std::endl;
#endif
    info->storeTrack(true);
  }
     
}

void MuonSensitiveDetector::updateHit(G4Step * aStep){
  //  float thePabs             = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
  //  Local3DPoint theEntryPoint= InitialStepPosition(aStep,LocalCoordinates);  
  Local3DPoint theExitPoint = toOrcaUnits(FinalStepPosition(aStep,LocalCoordinates)); 
  float theEnergyLoss = aStep->GetTotalEnergyDeposit()/GeV;  

  if( theHit == 0 ){ 
    std::cerr << "!!ERRROR in MuonSensitiveDetector::updateHit. It is called when there is no hit " << std::endl;
  }

  theHit->updateExitPoint(theExitPoint);
  theHit->addEnergyLoss(theEnergyLoss);

#ifdef DEBUG      
  std::cout <<"=== UPDATE ===============> ELOSS   = "<<theEnergyLoss<<" "
       <<thePV->GetLogicalVolume()->GetName()<<std::endl;
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = aStep->GetPreStepPoint()->GetProcessDefinedStep();
  if (p)
    std::cout <<" POST PROCESS = "<<p->GetProcessName()<<std::endl;
  if (p2)
    std::cout <<" PRE  PROCESS = "<<p2->GetProcessName()<<std::endl;
  std::cout << "updhit exit  " << theExitPoint<<std::endl;
  std::cout << "updhit eloss " << theHit->energyLoss() <<std::endl;
  std::cout << "updhit detid " << theDetUnitId<<std::endl;
  std::cout << "updhit delta " << (theExitPoint-theHit->entryPoint())<<std::endl;
  std::cout << "updhit deltu " << (theExitPoint-theHit->entryPoint()).unit();
  std::cout << " " << (theExitPoint-theHit->entryPoint()).mag()<<std::endl; 
#endif

}

void MuonSensitiveDetector::saveHit(){

  if (theHit) {
    if (thePrintHits) {
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
    std::cout <<" ERROR: no G4VUserTrackInformation available"<<std::endl;
    abort();
  }else{
    TrackInformation* info = dynamic_cast<TrackInformation*>(temp);
    if (info ==0){
      std::cout <<" ERROR: TkSimTrackSelection: the UserInformation does not appear to be a TrackInformation"<<std::endl;
      abort();
    }
    return info;
  }
}

void MuonSensitiveDetector::EndOfEvent(G4HCofThisEvent*)
{
//  TimeMe t("MuonSensitiveDetector::EndOfEvent", false);
 // std::cout << "MuonSensitiveDetector::EndOfEvent saving last hit en event " << std::endl;
  saveHit();
}


void MuonSensitiveDetector::fillHits(edm::PSimHitContainer& c, std::string n){
  //
  // do it once for low, once for High
  //

  if (slaveMuon->name() == n)c.insertHits(slaveMuon->hits());

}

std::vector<std::string> MuonSensitiveDetector::getNames(){
  std::vector<std::string> temp;
  temp.push_back(slaveMuon->name());
  return temp;
}

