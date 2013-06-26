#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "SimG4CMS/Tracker/interface/TkAccumulatingSensitiveDetector.h"
#include "SimG4CMS/Tracker/interface/FakeFrameRotation.h"
#include "SimG4CMS/Tracker/interface/StandardFrameRotation.h"
#include "SimG4CMS/Tracker/interface/TrackerFrameRotation.h"
#include "SimG4CMS/Tracker/interface/TkSimHitPrinter.h"
#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

#include <string>
#include <vector>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define FAKEFRAMEROTATION
//#define DUMPPROCESSES

#ifdef DUMPPROCESSES
#include "G4VProcess.hh"
#endif

using std::cout;
using std::endl;
using std::vector;
using std::string;

static 
TrackerG4SimHitNumberingScheme&
numberingScheme(const DDCompactView& cpv, const GeometricDet& det)
{
   static TrackerG4SimHitNumberingScheme s_scheme(cpv, det);
   return s_scheme;
}


TkAccumulatingSensitiveDetector::TkAccumulatingSensitiveDetector(string name, 
								 const DDCompactView & cpv,
								 SensitiveDetectorCatalog & clg, 
								 edm::ParameterSet const & p,
								 const SimTrackManager* manager) : 
  SensitiveTkDetector(name, cpv, clg, p), myName(name), myRotation(0),  mySimHit(0),theManager(manager),
   oldVolume(0), lastId(0), lastTrack(0), eventno(0) ,rTracker(1200.*mm),zTracker(3000.*mm),
   numberingScheme_(0)
{
   
  edm::ParameterSet m_TrackerSD = p.getParameter<edm::ParameterSet>("TrackerSD");
  allowZeroEnergyLoss = m_TrackerSD.getParameter<bool>("ZeroEnergyLoss");
  neverAccumulate     = m_TrackerSD.getParameter<bool>("NeverAccumulate");
  printHits           = m_TrackerSD.getParameter<bool>("PrintHits");
  theSigma            = m_TrackerSD.getParameter<double>("ElectronicSigmaInNanoSeconds");
  energyCut           = m_TrackerSD.getParameter<double>("EnergyThresholdForPersistencyInGeV")*GeV; //default must be 0.5
  energyHistoryCut    = m_TrackerSD.getParameter<double>("EnergyThresholdForHistoryInGeV")*GeV;//default must be 0.05

  edm::LogInfo("TrackerSimInfo") <<"Criteria for Saving Tracker SimTracks:  ";
  edm::LogInfo("TrackerSimInfo")<<" History: "<<energyHistoryCut<< " MeV ; Persistency: "<< energyCut<<" MeV ";
  edm::LogInfo("TrackerSimInfo")<<" Constructing a TkAccumulatingSensitiveDetector with ";

#ifndef FAKEFRAMEROTATION
  // No Rotation given in input, automagically choose one based upon the name
  if (name.find("TrackerHits") != string::npos) 
    {
      edm::LogInfo("TrackerSimInfo")<<" TkAccumulatingSensitiveDetector: using TrackerFrameRotation for "<<myName;
      myRotation = new TrackerFrameRotation;
    }
  // Just in case (test beam etc)
  if (myRotation == 0) 
    {
      edm::LogInfo("TrackerSimInfo")<<" TkAccumulatingSensitiveDetector: using StandardFrameRotation for "<<myName;
      myRotation = new StandardFrameRotation;
    }
#else
  myRotation = new FakeFrameRotation;
  edm::LogWarning("TrackerSimInfo")<< " WARNING - Using FakeFrameRotation in TkAccumulatingSensitiveDetector;";  
#endif

    slaveLowTof  = new TrackingSlaveSD(name+"LowTof");
    slaveHighTof = new TrackingSlaveSD(name+"HighTof");
  
    // Now attach the right detectors (LogicalVolumes) to me
    vector<string>  lvNames = clg.logicalNames(name);
    this->Register();
    for (vector<string>::iterator it = lvNames.begin();  it != lvNames.end(); it++)
    {
      edm::LogInfo("TrackerSimInfo")<< name << " attaching LV " << *it;
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

    LogDebug("TrackerSimDebug")<< " Entering a new Step " << aStep->GetTotalEnergyDeposit() << " " 
	 << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName();

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
 unsigned int detId = numberingScheme_->g4ToNumberingScheme(s->GetPreStepPoint()->GetTouchable());

 LogDebug("TrackerSimDebug")<< " DetID = "<<detId; 

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
  edm::LogInfo("TrackerSimInfo") <<" -> process creator pointer "<<gTrack->GetCreatorProcess();
  if (gTrack->GetCreatorProcess())
    edm::LogInfo("TrackerSimInfo")<<" -> PROCESS CREATOR : "<<gTrack->GetCreatorProcess()->GetProcessName();
#endif


  //
  //Position
  //
  const G4ThreeVector pos = gTrack->GetPosition();
  //LogDebug("TrackerSimDebug")<<" ENERGY MeV "<<gTrack->GetKineticEnergy()<<" Energy Cut" << energyCut;
  //LogDebug("TrackerSimDebug")<<" TOTAL ENERGY "<<gTrack->GetTotalEnergy();
  //LogDebug("TrackerSimDebug")<<" WEIGHT "<<gTrack->GetWeight();

  //
  // Check if in Tracker Volume
  //
  if (pos.perp() < rTracker &&std::fabs(pos.z()) < zTracker){
    //
    // inside the Tracker
    //
    LogDebug("TrackerSimDebug")<<" INSIDE TRACKER";

    if (gTrack->GetKineticEnergy() > energyCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
      //LogDebug("TrackerSimDebug")<<" POINTER "<<info;
      //LogDebug("TrackerSimDebug")<<" track inside the tracker selected for STORE";
      //LogDebug("TrackerSimDebug")<<"Track ID (persistent track) = "<<gTrack->GetTrackID();

      info->storeTrack(true);
    }
    //
    // Save History?
    //
    if (gTrack->GetKineticEnergy() > energyHistoryCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
      info->putInHistory();
      //LogDebug("TrackerSimDebug")<<" POINTER "<<info;
      //LogDebug("TrackerSimDebug")<<" track inside the tracker selected for HISTORY";
      //LogDebug("TrackerSimDebug")<<"Track ID (history track) = "<<gTrack->GetTrackID();
    }
    
  }
}



void TkAccumulatingSensitiveDetector::sendHit()
{  
    if (mySimHit == 0) return;
    LogDebug("TrackerSimDebug")<< " Storing PSimHit: " << pname << " " << mySimHit->detUnitId() 
	 << " " << mySimHit->trackId() << " " << mySimHit->energyLoss() 
	 << " " << mySimHit->entryPoint() << " " << mySimHit->exitPoint();
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
    Local3DPoint theEntryPoint;
    Local3DPoint theExitPoint = 
      toOrcaRef(SensitiveDetector::FinalStepPosition(aStep,LocalCoordinates),v); 

    //
    //  Check particle type - for gamma and neutral hadrons energy deposition
    //  should be local (V.I.)
    //
    if(0.0 == theTrack->GetDefinition()->GetPDGCharge()) {
      theEntryPoint = theExitPoint; 
    } else {
      theEntryPoint = toOrcaRef(SensitiveDetector::InitialStepPosition(aStep,LocalCoordinates),v);
    }

    //
    //	This allows to send he skipEvent if it is outside!
    //
    checkExitPoint(theExitPoint);
    float thePabs             = aStep->GetPreStepPoint()->GetMomentum().mag()/GeV;
    float theTof              = aStep->GetPreStepPoint()->GetGlobalTime()/nanosecond;
    float theEnergyLoss       = aStep->GetTotalEnergyDeposit()/GeV;
    int theParticleType       = myG4TrackToParticleID->particleID(theTrack);
    uint32_t theDetUnitId     = setDetUnitId(aStep);

    // 
    // Check on particle charge is not applied because these points are not stored
    // in hits (V.I.)
    //  
    globalEntryPoint = SensitiveDetector::InitialStepPosition(aStep,WorldCoordinates);
    globalExitPoint = SensitiveDetector::FinalStepPosition(aStep,WorldCoordinates);

    pname = theTrack->GetDynamicParticle()->GetDefinition()->GetParticleName();
  
    if (theDetUnitId == 0)
    {
      edm::LogError("TrackerSimInfo") << " Error: theDetUnitId is not valid.";
      abort();
    }
    unsigned int theTrackID = theTrack->GetTrackID();
  
    // To whom assign the Hit?
    // First iteration: if the track is to be stored, use the current number;
    // otherwise, get to the mother
    unsigned  int theTrackIDInsideTheSimHit=theTrackID;

    
    G4VUserTrackInformation * info = theTrack->GetUserInformation();
    if (info == 0) edm::LogError("TrackerSimInfo")<< " Error: no UserInformation available ";
    else
      {
	TrackInformation * temp = dynamic_cast<TrackInformation* >(info);
	if (temp ==0) edm::LogError("TrackerSimInfo")<< " Error:G4VUserTrackInformation is not a TrackInformation.";
	if (temp->storeTrack() == false) 
	  {
	    // Go to the mother!
	    LogDebug("TrackerSimDebug")<< " TkAccumulatingSensitiveDetector:createHit(): setting the TrackID from "
		      << theTrackIDInsideTheSimHit;
	    theTrackIDInsideTheSimHit = theTrack->GetParentID();
	    LogDebug("TrackerSimDebug")<< " to the mother one " << theTrackIDInsideTheSimHit << " " << theEnergyLoss;
	  }
	else
	  {
	    LogDebug("TrackerSimDebug")<< " TkAccumulatingSensitiveDetector:createHit(): leaving the current TrackID " 
		      << theTrackIDInsideTheSimHit;
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
    LogDebug("TrackerSimDebug")<< " Created PSimHit: " << pname << " " << mySimHit->detUnitId() << " " << mySimHit->trackId()
			       << " " << mySimHit->energyLoss() << " " << mySimHit->entryPoint() 
			       << " " << mySimHit->exitPoint();
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
    LogDebug("TrackerSimDebug")<< " Before : " << mySimHit->energyLoss();
    mySimHit->addEnergyLoss(theEnergyLoss);
    globalExitPoint = SensitiveDetector::FinalStepPosition(aStep,WorldCoordinates);

    LogDebug("TrackerSimDebug")<< " Updating: new exitpoint " << pname << " " << theExitPoint 
			       << " new energy loss " << theEnergyLoss;
    LogDebug("TrackerSimDebug")<< " Updated PSimHit: " << mySimHit->detUnitId() << " " << mySimHit->trackId()
			       << " " << mySimHit->energyLoss() << " " << mySimHit->entryPoint() 
			       << " " << mySimHit->exitPoint();
}

bool TkAccumulatingSensitiveDetector::newHit(G4Step * aStep)
{
    if (neverAccumulate == true) return true;
    G4Track * theTrack = aStep->GetTrack(); 

    // for neutral particles do not merge hits (V.I.) 
    if(0.0 == theTrack->GetDefinition()->GetPDGCharge()) return true;

    uint32_t theDetUnitId = setDetUnitId(aStep);
    unsigned int theTrackID = theTrack->GetTrackID();

    LogDebug("TrackerSimDebug")<< " OLD (d,t) = (" << lastId << "," << lastTrack 
			       << "), new = (" << theDetUnitId << "," << theTrackID << ") return "
			       << ((theTrackID == lastTrack) && (lastId == theDetUnitId));
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
    LogDebug("TrackerSimDebug")<< " closeHit: distance = " << (mySimHit->exitPoint()-theEntryPoint).mag();

    if ((mySimHit->exitPoint()-theEntryPoint).mag()<tolerance) return true;
    return false;
}

void TkAccumulatingSensitiveDetector::EndOfEvent(G4HCofThisEvent *)
{

  LogDebug("TrackerSimDebug")<< " Saving the last hit in a ROU " << myName;

    if (mySimHit == 0) return;
      sendHit();
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

    edm::ESTransientHandle<DDCompactView> pView;
    es->get<IdealGeometryRecord>().get(pView);

    numberingScheme_=&(numberingScheme(*pView,*pDD));
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
    edm::LogWarning("TrackerSimInfo")<< " ************ Hit outside the detector ; Local z " << z 
				     << "; skipping event = " << sendExc;
    if (sendExc == true)
    {
	G4EventManager::GetEventManager()->AbortCurrentEvent();
	G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->SetEventAborted();
    }
}

TrackInformation* TkAccumulatingSensitiveDetector::getOrCreateTrackInformation( const G4Track* gTrack){
  G4VUserTrackInformation* temp = gTrack->GetUserInformation();
  if (temp == 0){
    edm::LogError("TrackerSimInfo") <<" ERROR: no G4VUserTrackInformation available";
    abort();
  }else{
    TrackInformation* info = dynamic_cast<TrackInformation*>(temp);
    if (info ==0){
      edm::LogError("TrackerSimInfo")<<" ERROR: TkSimTrackSelection: the UserInformation does not appear to be a TrackInformation";
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
