#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Geometry/interface/SDCatalog.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "SimG4CMS/Tracker/interface/TkAccumulatingSensitiveDetector.h"
#include "SimG4CMS/Tracker/interface/TrackingSlaveSDWithRenumbering.h"
#include "SimG4CMS/Tracker/interface/FakeFrameRotation.h"
#include "SimG4CMS/Tracker/interface/StandardFrameRotation.h"
#include "SimG4CMS/Tracker/interface/TrackerFrameRotation.h"
#include "SimG4CMS/Tracker/interface/TkSimHitPrinter.h"
#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

#include <string>
#include <vector>
#include <iostream>

//#define FAKEFRAMEROTATION
//#define DEBUG
//#define DEBUG_ID
//#define DEBUG_TRACK
//#define DUMPPROCESSES

#ifdef DUMPPROCESSES
#include "G4VProcess.hh"
#endif

using std::cout;
using std::endl;
using std::vector;
using std::string;

TkAccumulatingSensitiveDetector::TkAccumulatingSensitiveDetector(string name, 
								 const DDCompactView & cpv,
								 edm::ParameterSet const & p,
								 const SimTrackManager* manager) : 
  SensitiveTkDetector(name, cpv, p), myName(name), myRotation(0),  mySimHit(0),theManager(manager),
  oldVolume(0), lastId(0), lastTrack(0), eventno(0) ,rTracker(1200.*mm),zTracker(3000.*mm){

  edm::ParameterSet m_TrackerSD = p.getParameter<edm::ParameterSet>("TrackerSD");
  allowZeroEnergyLoss = m_TrackerSD.getParameter<bool>("ZeroEnergyLoss");
  neverAccumulate     = m_TrackerSD.getParameter<bool>("NeverAccumulate");
  printHits           = m_TrackerSD.getParameter<bool>("PrintHits");
  theSigma            = m_TrackerSD.getParameter<double>("ElectronicSigmaInNanoSeconds");
  energyCut           = m_TrackerSD.getParameter<double>("EnergyThresholdForPersistencyInGeV")*GeV; //default must be 0.5
  energyHistoryCut    = m_TrackerSD.getParameter<double>("EnergyThresholdForHistoryInGeV")*GeV;//default must be 0.05

  std::cout <<"Criteria for Saving Tracker SimTracks:  ";
  std::cout <<" History: "<<energyHistoryCut<< " MeV ; Persistency: "<< energyCut<<" MeV "<<std::endl;
  std::cout<<" Constructing a TkAccumulatingSensitiveDetector with "<<name<<std::endl;


#ifndef FAKEFRAMEROTATION
  // No Rotation given in input, automagically choose one based upon the name
  if (name.find("TrackerHits") != string::npos) 
    {
      std::cout <<" TkAccumulatingSensitiveDetector: using TrackerFrameRotation for "<<myName<<std::endl;
      myRotation = new TrackerFrameRotation;
    }
  // Just in case (test beam etc)
  if (myRotation == 0) 
    {
      std::cout <<" TkAccumulatingSensitiveDetector: using StandardFrameRotation for "<<myName<<std::endl;
      myRotation = new StandardFrameRotation;
    }
#else
  myRotation = new FakeFrameRotation;
  std::cout << " WARNING - Using FakeFrameRotation in TkAccumulatingSensitiveDetector;" << std::endl 
       << "This is NOT producing a usable DB," 
       << " but should be used for debugging purposes only." << std::endl;
  
#endif

    slaveLowTof  = new TrackingSlaveSD(name+"LowTof");
    slaveHighTof = new TrackingSlaveSD(name+"HighTof");
  
    // Now attach the right detectors (LogicalVolumes) to me
    vector<string>  lvNames= SensitiveDetectorCatalog::instance()->logicalNames(name);
    this->Register();
    for (vector<string>::iterator it = lvNames.begin();  it != lvNames.end(); it++)
    {
	std::cout << name << " attaching LV " << *it << std::endl;
	this->AssignSD(*it);
    }

    theG4ProcessTypeEnumerator = new G4ProcessTypeEnumerator;
    myG4TrackToParticleID = new G4TrackToParticleID;
}

TkAccumulatingSensitiveDetector::~TkAccumulatingSensitiveDetector() 
{ 
  delete slaveLowTof;
  delete slaveHighTof;
  delete theG4ProcessTypeEnumerator;
  delete myG4TrackToParticleID;
}


bool TkAccumulatingSensitiveDetector::ProcessHits(G4Step * aStep, G4TouchableHistory * ROhist)
{
#ifdef DEBUG
    std::cout << " Entering a new Step " << aStep->GetTotalEnergyDeposit() << " " 
	 << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName() << std::endl;
#endif
    //TimeMe t1(theTimer, false);
    if (aStep->GetTotalEnergyDeposit()>0. || allowZeroEnergyLoss == true)
    {
	if (newHit(aStep) == true)
	{
	    sendHit();
	    createHit(aStep);
	}
	else
	{
	    updateHit(aStep);
	}
	return true;
    }
    return false;
}

uint32_t TkAccumulatingSensitiveDetector::setDetUnitId(G4Step * s)
{ 
 unsigned int detId = TrackerG4SimHitNumberingScheme::instance().g4ToNumberingScheme(s->GetPreStepPoint()->GetTouchable());

#ifdef DEBUG_ID
 std::cout << " DetID = "<<detId<<std::endl; 
#endif    

 return detId;
}


int TkAccumulatingSensitiveDetector::tofBin(float tof)
{

    float threshold = 3. * theSigma;
    if (tof < threshold) return 1;
    return 2;
}

Local3DPoint TkAccumulatingSensitiveDetector::toOrcaRef(Local3DPoint in ,G4VPhysicalVolume * v)
{
    if (myRotation !=0) return myRotation->transformPoint(in,v);
    return (in);
}


void TkAccumulatingSensitiveDetector::update(const BeginOfTrack *bot){
  const G4Track* gTrack = (*bot)();
#ifdef DUMPPROCESSES
  std::cout <<" -> process creator pointer "<<gTrack->GetCreatorProcess()<<std::endl;
  if (gTrack->GetCreatorProcess())
    std::cout <<" -> PROCESS CREATOR : "<<gTrack->GetCreatorProcess()->GetProcessName()<<std::endl;

#endif


  //
  //Position
  //
  const G4ThreeVector pos = gTrack->GetPosition();
#ifdef DEBUG_TRACK_DEEP
  std::cout <<" ENERGY MeV "<<gTrack->GetKineticEnergy()<<" Energy Cut" << energyCut<<std::endl;
  std::cout <<" TOTAL ENERGY "<<gTrack->GetTotalEnergy()<<std::endl;
  std::cout <<" WEIGHT "<<gTrack->GetWeight()<<std::endl;
#endif
  //
  // Check if in Tracker Volume
  //
  if (pos.perp() < rTracker &&std::fabs(pos.z()) < zTracker){
    //
    // inside the Tracker
    //
#ifdef DEBUG_TRACK_DEEP
      std::cout <<" INSIDE TRACKER"<<std::endl;
#endif
    if (gTrack->GetKineticEnergy() > energyCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
#ifdef DEBUG_TRACK_DEEP
      std::cout <<" POINTER "<<info<<std::endl;
      std::cout <<" track inside the tracker selected for STORE"<<std::endl;
#endif
#ifdef DEBUG_TRACK
      std::cout <<"Track ID (persistent track) = "<<gTrack->GetTrackID() <<std::endl;
#endif
      info->storeTrack(true);
    }
    //
    // Save History?
    //
    if (gTrack->GetKineticEnergy() > energyHistoryCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
      info->putInHistory();
#ifdef DEBUG_TRACK_DEEP
      std::cout <<" POINTER "<<info<<std::endl;
      std::cout <<" track inside the tracker selected for HISTORY"<<std::endl;
#endif
#ifdef DEBUG_TRACK
      std::cout <<"Track ID (history track) = "<<gTrack->GetTrackID() <<std::endl;
#endif


    }
    
  }
}



void TkAccumulatingSensitiveDetector::sendHit()
{  
    if (mySimHit == 0) return;
#ifdef DEBUG
    std::cout << " Storing PSimHit: " << pname << " " << mySimHit->detUnitId() 
	 << " " << mySimHit->trackId() << " " << mySimHit->energyLoss() 
	 << " " << mySimHit->entryPoint() << " " << mySimHit->exitPoint() << std::endl;
#endif
    if (printHits)
    {
	TkSimHitPrinter thePrinter("TkHitPositionOSCAR.dat");
	thePrinter.startNewSimHit(myName,oldVolume->GetLogicalVolume()->GetName(),
				  mySimHit->detUnitId(),mySimHit->trackId(),eventno);
	thePrinter.printLocal(mySimHit->entryPoint(),mySimHit->exitPoint());
	thePrinter.printGlobal(globalEntryPoint,globalExitPoint);
	thePrinter.printHitData(mySimHit->energyLoss(),mySimHit->timeOfFlight());
	thePrinter.printMomentumOfTrack(mySimHit->pabs(),pname,
					thePrinter.getPropagationSign(globalEntryPoint,globalExitPoint));
	thePrinter.printGlobalMomentum(px,py,pz);
    }
    
    if (tofBin(mySimHit->timeOfFlight()) == 1)
	slaveLowTof->processHits(*mySimHit);  // implicit conversion (slicing) to PSimHit!!!
    else
	slaveHighTof->processHits(*mySimHit); // implicit conversion (slicing) to PSimHit!!!
    //
    // clean up
    delete mySimHit;
    mySimHit = 0;
    lastTrack = 0;
    lastId = 0;
}

void TkAccumulatingSensitiveDetector::createHit(G4Step * aStep)
{
    if (mySimHit != 0) 
    {
	delete mySimHit;
	mySimHit=0;
    }
    
    G4Track * theTrack  = aStep->GetTrack(); 

    G4VPhysicalVolume * v = aStep->GetPreStepPoint()->GetPhysicalVolume();
    Local3DPoint theEntryPoint = toOrcaRef(SensitiveDetector::InitialStepPosition(aStep,LocalCoordinates),v);  
    Local3DPoint theExitPoint  = toOrcaRef(SensitiveDetector::FinalStepPosition(aStep,LocalCoordinates),v); 
    //
    //	This allows to send he skipEvent if it is outside!
    //
    checkExitPoint(theExitPoint);
    float thePabs             = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
    float theTof              = aStep->GetPreStepPoint()->GetGlobalTime()/nanosecond;
    float theEnergyLoss       = aStep->GetTotalEnergyDeposit()/GeV;
    short theParticleType     = myG4TrackToParticleID->particleID(theTrack);
    uint32_t theDetUnitId     = setDetUnitId(aStep);
  
    globalEntryPoint = SensitiveDetector::InitialStepPosition(aStep,WorldCoordinates);
    globalExitPoint = SensitiveDetector::FinalStepPosition(aStep,WorldCoordinates);
    pname = theTrack->GetDynamicParticle()->GetDefinition()->GetParticleName();
  
    if (theDetUnitId == 0)
    {
	std::cout << " Error: theDetUnitId is not valid." << std::endl;
	abort();
    }
    unsigned int theTrackID = theTrack->GetTrackID();
  
    // To whom assign the Hit?
    // First iteration: if the track is to be stored, use the current number;
    // otherwise, get to the mother
    unsigned  int theTrackIDInsideTheSimHit=theTrackID;

    
    G4VUserTrackInformation * info = theTrack->GetUserInformation();
    if (info == 0) std::cout << " Error: no UserInformation available " << std::endl;
    else
      {
	TrackInformation * temp = dynamic_cast<TrackInformation* >(info);
	if (temp ==0) std::cout << " Error:G4VUserTrackInformation is not a TrackInformation." << std::endl;
	if (temp->storeTrack() == false) 
	  {
	    // Go to the mother!
#ifdef DEBUG_TRACK
	    std::cout << " TkAccumulatingSensitiveDetector:createHit(): setting the TrackID from "
		      << theTrackIDInsideTheSimHit;
#endif
	    theTrackIDInsideTheSimHit = theTrack->GetParentID();
#ifdef DEBUG_TRACK
	    std::cout << " to the mother one " << theTrackIDInsideTheSimHit << " " << theEnergyLoss << std::endl;
#endif
	  }
	else
	  {
#ifdef DEBUG_TRACK
	    std::cout << " TkAccumulatingSensitiveDetector:createHit(): leaving the current TrackID " 
		      << theTrackIDInsideTheSimHit << std::endl;
#endif
	  }
      }
    
    
    px  = aStep->GetPreStepPoint()->GetMomentum().x()/GeV;
    py  = aStep->GetPreStepPoint()->GetMomentum().y()/GeV;
    pz  = aStep->GetPreStepPoint()->GetMomentum().z()/GeV;
    
    G4ThreeVector gmd  = aStep->GetPreStepPoint()->GetMomentumDirection();
    // convert it to local frame
    G4ThreeVector lmd = ((G4TouchableHistory *)(aStep->GetPreStepPoint()->GetTouchable()))->GetHistory()
      ->GetTopTransform().TransformAxis(gmd);
    Local3DPoint lnmd = toOrcaRef(ConvertToLocal3DPoint(lmd),v);
    float theThetaAtEntry = lnmd.theta();
    float thePhiAtEntry = lnmd.phi();
    
    mySimHit = new UpdatablePSimHit(theEntryPoint,theExitPoint,thePabs,theTof,
				    theEnergyLoss,theParticleType,theDetUnitId,
				    theTrackIDInsideTheSimHit,theThetaAtEntry,thePhiAtEntry,
				    theG4ProcessTypeEnumerator->processId(theTrack->GetCreatorProcess()));  
#ifdef DEBUG_TRACK
    std::cout << " Created PSimHit: " << pname << " " << mySimHit->detUnitId() << " " << mySimHit->trackId()
	 << " " << mySimHit->energyLoss() << " " << mySimHit->entryPoint() 
	 << " " << mySimHit->exitPoint() << std::endl;
#endif    
    lastId = theDetUnitId;
    lastTrack = theTrackID;
    oldVolume = v;
}

void TkAccumulatingSensitiveDetector::updateHit(G4Step * aStep)
{
    G4VPhysicalVolume * v = aStep->GetPreStepPoint()->GetPhysicalVolume();
    Local3DPoint theExitPoint = toOrcaRef(SensitiveDetector::FinalStepPosition(aStep,LocalCoordinates),v); 
    //
    // This allows to send he skipEvent if it is outside!
    //
    checkExitPoint(theExitPoint);
    float theEnergyLoss = aStep->GetTotalEnergyDeposit()/GeV;
    mySimHit->setExitPoint(theExitPoint);
#ifdef DEBUG
    std::cout << " Before : " << mySimHit->energyLoss() << std::endl;
#endif
    mySimHit->addEnergyLoss(theEnergyLoss);
    globalExitPoint = SensitiveDetector::FinalStepPosition(aStep,WorldCoordinates);
#ifdef DEBUG
    std::cout << " Updating: new exitpoint " << pname << " " << theExitPoint 
	 << " new energy loss " << theEnergyLoss << std::endl;
    std::cout << " Updated PSimHit: " << mySimHit->detUnitId() << " " << mySimHit->trackId()
	 << " " << mySimHit->energyLoss() << " " << mySimHit->entryPoint() 
	 << " " << mySimHit->exitPoint() << std::endl;
#endif
}

bool TkAccumulatingSensitiveDetector::newHit(G4Step * aStep)
{
    if (neverAccumulate == true) return true;
    G4Track * theTrack = aStep->GetTrack(); 
    uint32_t theDetUnitId = setDetUnitId(aStep);
    unsigned int theTrackID = theTrack->GetTrackID();
#ifdef DEBUG
    std::cout << " OLD (d,t) = (" << lastId << "," << lastTrack 
	 << "), new = (" << theDetUnitId << "," << theTrackID << ") return "
	 << ((theTrackID == lastTrack) && (lastId == theDetUnitId)) << std::endl;
#endif
    if ((mySimHit != 0) && (theTrackID == lastTrack) && (lastId == theDetUnitId) && closeHit(aStep))
	return false;
    return true;
}

bool TkAccumulatingSensitiveDetector::closeHit(G4Step * aStep)
{
    if (mySimHit == 0) return false; 
    const float tolerance = 0.05 * mm; // 50 micron are allowed between the exit 
    // point of the current hit and the entry point of the new hit
    G4VPhysicalVolume * v = aStep->GetPreStepPoint()->GetPhysicalVolume();
    Local3DPoint theEntryPoint = toOrcaRef(SensitiveDetector::InitialStepPosition(aStep,LocalCoordinates),v);  
#ifdef DEBUG
    std::cout << " closeHit: distance = " << (mySimHit->exitPoint()-theEntryPoint).mag() << std::endl;
#endif
    if ((mySimHit->exitPoint()-theEntryPoint).mag()<tolerance) return true;
    return false;
}

void TkAccumulatingSensitiveDetector::EndOfEvent(G4HCofThisEvent *)
{
#ifdef DEBUG
    std::cout << " Saving the last hit in a ROU " << myName << std::endl;
#endif
    if (mySimHit == 0) return;
      sendHit();
}

void TkAccumulatingSensitiveDetector::update(const ::EndOfEvent*){
   slaveLowTof->renumbering(theManager);	
   slaveHighTof->renumbering(theManager);
}    

void TkAccumulatingSensitiveDetector::update(const BeginOfEvent * i)
{
    clearHits();
    eventno = (*i)()->GetEventID();
    mySimHit = 0;
}

void TkAccumulatingSensitiveDetector::update(const BeginOfJob * i)
{
    edm::ESHandle<GeometricDet> pDD;
    const edm::EventSetup* es = (*i)();
    es->get<IdealGeometryRecord>().get( pDD );
}

void TkAccumulatingSensitiveDetector::clearHits()
{
    slaveLowTof->Initialize();
    slaveHighTof->Initialize();
}

void TkAccumulatingSensitiveDetector::checkExitPoint(Local3DPoint p)
{
    double z = p.z();
    if (std::abs(z)<0.3*mm) return;
    bool sendExc = false;
    //static SimpleConfigurable<bool> sendExc(false,"TrackerSim:ThrowOnBadHits");
    std::cout << " ************ Hit outside the detector ; Local z " << z 
	 << "; skipping event = " << sendExc << std::endl;
    if (sendExc == true)
    {
	G4EventManager::GetEventManager()->AbortCurrentEvent();
	G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->SetEventAborted();
    }
}

TrackInformation* TkAccumulatingSensitiveDetector::getOrCreateTrackInformation( const G4Track* gTrack){
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

void TkAccumulatingSensitiveDetector::fillHits(edm::PSimHitContainer& c, std::string n){
  //
  // do it once for low, once for High
  //

  if (slaveLowTof->name() == n)  c=slaveLowTof->hits();
  if (slaveHighTof->name() == n) c=slaveHighTof->hits();

}

std::vector<string> TkAccumulatingSensitiveDetector::getNames(){
  std::vector<string> temp;
  temp.push_back(slaveLowTof->name());
  temp.push_back(slaveHighTof->name());
  return temp;
}



