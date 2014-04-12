#include "SimG4CMS/Forward/interface/PltSD.h"

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

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <string>
#include <vector>
#include <iostream>


PltSD::PltSD(std::string name,
             const DDCompactView & cpv,
             SensitiveDetectorCatalog & clg,
             edm::ParameterSet const & p,
             const SimTrackManager* manager) :
SensitiveTkDetector(name, cpv, clg, p), myName(name), mySimHit(0),
oldVolume(0), lastId(0), lastTrack(0), eventno(0) {
    
    edm::ParameterSet m_TrackerSD = p.getParameter<edm::ParameterSet>("PltSD");
    energyCut           = m_TrackerSD.getParameter<double>("EnergyThresholdForPersistencyInGeV")*GeV; //default must be 0.5 (?)
    energyHistoryCut    = m_TrackerSD.getParameter<double>("EnergyThresholdForHistoryInGeV")*GeV;//default must be 0.05 (?)
    
    edm::LogInfo("PltSD") <<"Criteria for Saving Tracker SimTracks: \n "
    <<" History: "<<energyHistoryCut<< " MeV ; Persistency: "<< energyCut<<" MeV\n"
    <<" Constructing a PltSD with ";
    
    slave  = new TrackingSlaveSD(name);
    
    // Now attach the right detectors (LogicalVolumes) to me
    std::vector<std::string>  lvNames = clg.logicalNames(name);
    this->Register();
    for (std::vector<std::string>::iterator it = lvNames.begin();
         it != lvNames.end(); it++)  {
        edm::LogInfo("PltSD")<< name << " attaching LV " << *it;
        this->AssignSD(*it);
    }
    
    theG4ProcessTypeEnumerator = new G4ProcessTypeEnumerator;
    myG4TrackToParticleID = new G4TrackToParticleID;
}

PltSD::~PltSD() {
    delete slave;
    delete theG4ProcessTypeEnumerator;
    delete myG4TrackToParticleID;
}

bool PltSD::ProcessHits(G4Step * aStep,  G4TouchableHistory *) {
    
    LogDebug("PltSD") << " Entering a new Step "
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

uint32_t PltSD::setDetUnitId(G4Step * aStep ) {
    
    unsigned int detId = 0;
    
    LogDebug("PltSD")<< " DetID = "<<detId;
    
    //Find number of levels
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    int level = 0;
    if (touch) level = ((touch->GetHistoryDepth())+1);
    
    //Get name and copy numbers
    if ( level > 1 ) {
        //some debugging with the names
        G4String sensorName   = touch->GetVolume(2)->GetName();
        G4String telName      = touch->GetVolume(3)->GetName();
        G4String volumeName   = touch->GetVolume(4)->GetName();
        if ( sensorName != "PLTSensorPlane" )
        std::cout << " PltSD::setDetUnitId -w- Sensor name not PLTSensorPlane " << std::endl;
        if ( telName != "Telescope" )
        std::cout << " PltSD::setDetUnitId -w- Telescope name not Telescope " << std::endl;
        if ( volumeName != "PLT" )
        std::cout << " PltSD::setDetUnitId -w- Volume name not PLT " << std::endl;
        
        //Get the information about which telescope, plane, row/column was hit
        int columnNum = touch->GetReplicaNumber(0);
        int rowNum = touch->GetReplicaNumber(1);
        int sensorNum = touch->GetReplicaNumber(2);
        int telNum  = touch->GetReplicaNumber(3);
        //temp stores the PLTBCM volume the hit occured in (i.e. was the hit on the + or -z side?)
        int temp = touch->GetReplicaNumber(5);
        //convert to the PLT hit id standard
        int pltNum;
        if (temp == 2) pltNum = 0;
        else pltNum = 1;

        //correct the telescope numbers on the -z side to have the same naming convention in phi as the +z side
        if (pltNum == 0){
            if (telNum == 0){
                telNum = 7;
            }
            else if (telNum == 1){
                telNum = 6;
            }
            else if (telNum == 2){
                telNum = 5;
            }
            else if (telNum == 3){
                telNum = 4;
            }
            else if (telNum == 4){
                telNum = 3;
            }
            else if (telNum == 5){
                telNum = 2;
            }
            else if (telNum == 6){
                telNum = 1;
            }
            else if (telNum == 7){
                telNum = 0;
            }
        }
        //the PLT is divided into sets of telescopes on the + and -x sides
        int halfCarriageNum = -1;

        //If the telescope is on the -x side of the carriage, halfCarriageNum=0.  If on the +x side, it is = 1.
        if(telNum == 0 || telNum == 1 || telNum == 2 || telNum == 3)
            halfCarriageNum = 0;
        else
            halfCarriageNum = 1;
        //correct the telescope numbers of the +x half-carriage to range from 0 to 3
        if(halfCarriageNum == 1){
            if(telNum == 4){
                telNum = 0;
            }
            else if (telNum == 5){
                telNum = 1;
            }
            else if (telNum == 6){
                telNum = 2;
            }
            else if (telNum == 7){
                telNum = 3;
            }
        }
        //Define unique detId for each pixel.  See https://twiki.cern.ch/twiki/bin/viewauth/CMS/PLTSimulationGuide for more information
        detId = 10000000*pltNum+1000000*halfCarriageNum+100000*telNum+10000*sensorNum+100*rowNum+columnNum;
        //std::cout <<  "Hit Recorded at " << "plt:" << pltNum << " hc:" << halfCarriageNum << " tel:" << telNum << " plane:" << sensorNum << std::endl;
    }
    
    return detId;
}

void PltSD::EndOfEvent(G4HCofThisEvent *) {
    
    LogDebug("PltSD")<< " Saving the last hit in a ROU " << myName;
    
    if (mySimHit == 0) return;
    sendHit();
}

void PltSD::fillHits(edm::PSimHitContainer& c, std::string n){
    if (slave->name() == n)  c=slave->hits();
}

void PltSD::sendHit() {
    if (mySimHit == 0) return;
    LogDebug("PltSD") << " Storing PSimHit: " << pname << " " << mySimHit->detUnitId()
    << " " << mySimHit->trackId() << " " << mySimHit->energyLoss()
    << " " << mySimHit->entryPoint() << " " << mySimHit->exitPoint();
    
    slave->processHits(*mySimHit);
    
    // clean up
    delete mySimHit;
    mySimHit = 0;
    lastTrack = 0;
    lastId = 0;
}

void PltSD::updateHit(G4Step * aStep) {
    
    Local3DPoint theExitPoint = SensitiveDetector::FinalStepPosition(aStep,LocalCoordinates);
    float theEnergyLoss = aStep->GetTotalEnergyDeposit()/GeV;
    mySimHit->setExitPoint(theExitPoint);
    LogDebug("PltSD")<< " Before : " << mySimHit->energyLoss();
    mySimHit->addEnergyLoss(theEnergyLoss);
    globalExitPoint = SensitiveDetector::FinalStepPosition(aStep,WorldCoordinates);
    
    LogDebug("PltSD") << " Updating: new exitpoint " << pname << " "
    << theExitPoint << " new energy loss " << theEnergyLoss
    << "\n Updated PSimHit: " << mySimHit->detUnitId()
    << " " << mySimHit->trackId()
    << " " << mySimHit->energyLoss() << " "
    << mySimHit->entryPoint() << " " << mySimHit->exitPoint();
}

bool PltSD::newHit(G4Step * aStep) {
    
    G4Track * theTrack = aStep->GetTrack();
    uint32_t theDetUnitId = setDetUnitId(aStep);
    unsigned int theTrackID = theTrack->GetTrackID();
    
    LogDebug("PltSD") << " OLD (d,t) = (" << lastId << "," << lastTrack
    << "), new = (" << theDetUnitId << "," << theTrackID << ") return "
    << ((theTrackID == lastTrack) && (lastId == theDetUnitId));
    if ((mySimHit != 0) && (theTrackID == lastTrack) && (lastId == theDetUnitId) && closeHit(aStep))
    return false;
    return true;
}

bool PltSD::closeHit(G4Step * aStep) {
    
    if (mySimHit == 0) return false;
    const float tolerance = 0.05 * mm; // 50 micron are allowed between the exit
    // point of the current hit and the entry point of the new hit
    Local3DPoint theEntryPoint = SensitiveDetector::InitialStepPosition(aStep,LocalCoordinates);
    LogDebug("PltSD")<< " closeHit: distance = " << (mySimHit->exitPoint()-theEntryPoint).mag();
    
    if ((mySimHit->exitPoint()-theEntryPoint).mag()<tolerance) return true;
    return false;
}

void PltSD::createHit(G4Step * aStep) {
    
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
    
    LogDebug("PltSD") << " Created PSimHit: " << pname << " "
    << mySimHit->detUnitId() << " " << mySimHit->trackId()
    << " " << mySimHit->energyLoss() << " "
    << mySimHit->entryPoint() << " " << mySimHit->exitPoint();
    lastId = theDetUnitId;
    lastTrack = theTrackID;
    oldVolume = v;
}

void PltSD::update(const BeginOfJob * ) { }

void PltSD::update(const BeginOfEvent * i) {
    
    clearHits();
    eventno = (*i)()->GetEventID();
    mySimHit = 0;
}

void PltSD::update(const BeginOfTrack *bot) {
    
    const G4Track* gTrack = (*bot)();
    pname = gTrack->GetDynamicParticle()->GetDefinition()->GetParticleName();
}

void PltSD::clearHits() {
    slave->Initialize();
}

TrackInformation* PltSD::getOrCreateTrackInformation( const G4Track* gTrack) {
    G4VUserTrackInformation* temp = gTrack->GetUserInformation();
    if (temp == 0){
        edm::LogError("PltSD") <<" ERROR: no G4VUserTrackInformation available";
        abort();
    }else{
        TrackInformation* info = dynamic_cast<TrackInformation*>(temp);
        if (info == 0){
            edm::LogError("PltSD") <<" ERROR: TkSimTrackSelection: the UserInformation does not appear to be a TrackInformation";
            abort();
        }
        return info;
    }
}
