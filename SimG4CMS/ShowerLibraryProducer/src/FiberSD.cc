#include "SimG4CMS/ShowerLibraryProducer/interface/FiberSD.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4VSolid.hh"
#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4SDManager.hh"
#include "G4ios.hh"

FiberSD::FiberSD(std::string name, const DDCompactView & cpv, 
		 const SensitiveDetectorCatalog & clg, edm::ParameterSet const & p,
		 const SimTrackManager* manager) :
  SensitiveCaloDetector(name, cpv, clg, p), theName(name),
  m_trackManager(manager), theHCID(-1), theHC(0) {

  collectionName.insert(name);
  LogDebug("FiberSim") << "***************************************************"
		       << "\n"
		       << "*                                                 *"
		       << "\n"
		       << "* Constructing a FiberSD  with name " << GetName()
		       << "\n"
		       << "*                                                 *"
		       << "\n"
		       << "***************************************************";
  theShower = new HFShower(name, cpv, p, 1);

  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  const std::vector<std::string>& lvNames = clg.logicalNames(name);
  this->Register();
  for (std::vector<std::string>::const_iterator it=lvNames.begin();
       it !=lvNames.end(); it++){
    this->AssignSD(*it);
    LogDebug("FiberSim") << "FiberSD : Assigns SD to LV " << (*it);
  }
}

FiberSD::~FiberSD() {
 
  if (theShower) delete theShower;
  if (theHC)     delete theHC;
}

void FiberSD::Initialize(G4HCofThisEvent * HCE) {

  LogDebug("FiberSim") << "FiberSD : Initialize called for " << GetName();
  theHC = new FiberG4HitsCollection(GetName(), collectionName[0]);
  if (theHCID<0)
    theHCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(theHCID, theHC);
  
}

G4bool FiberSD::ProcessHits(G4Step * aStep, G4TouchableHistory*) {

  //std::vector<HFShower::Hit> hits = theShower->getHits(aStep);
  double zoffset = 1000;
  std::vector<HFShower::Hit> hits = theShower->getHits(aStep,true,zoffset);

  
  if (hits.size() > 0) {
    std::vector<HFShowerPhoton> thePE;
    for (unsigned int i=0; i<hits.size(); i++) {
      //std::cout<<"hit position z "<<hits[i].position.z()<<std::endl;
      HFShowerPhoton pe = HFShowerPhoton(hits[i].position.x(), 
					 hits[i].position.y(), 
					 hits[i].position.z(), 
					 hits[i].wavelength, hits[i].time);
      thePE.push_back(pe);
    }
    int trackID = aStep->GetTrack()->GetTrackID();
    G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
    const G4VTouchable* touch = preStepPoint->GetTouchable();
    G4LogicalVolume* lv = touch->GetVolume(0)->GetLogicalVolume();
    int depth = (touch->GetReplicaNumber(0))%10;
    int detID = setDetUnitId(aStep);
    math::XYZPoint theHitPos(preStepPoint->GetPosition().x(),
			     preStepPoint->GetPosition().y(),
			     preStepPoint->GetPosition().z());
    //std::cout<<"presteppoint position z "<<preStepPoint->GetPosition().z()<<std::endl;
    
    FiberG4Hit *aHit = new FiberG4Hit(lv, detID, depth, trackID);
    std::cout<<"hit size "<<hits.size()<<"  npe"<<aHit->npe()<<std::endl;
    std::cout<<"pre hit position "<<aHit->hitPos()<<std::endl;
    aHit->setNpe(hits.size());
    aHit->setPos(theHitPos);
    aHit->setTime(preStepPoint->GetGlobalTime());
    aHit->setPhoton(thePE);
    std::cout<<"ShowerPhoton position "<<thePE[0].x()<<" "<<thePE[0].y()<<" "<<thePE[0].z()<<std::endl;

    LogDebug("FiberSim") << "FiberSD: Hit created at " << lv->GetName()
			 << " DetID: " << aHit->towerId() << " Depth: " 
			 << aHit->depth() << " Track ID: " << aHit->trackId() 
			 << " Nb. of Cerenkov Photons: " << aHit->npe()
			 << " Time: " << aHit->time() << " at " 
			 << aHit->hitPos();
    for (unsigned int i=0; i<thePE.size(); i++) 
      LogDebug("FiberSim") << "FiberSD: PE[" << i << "] " << thePE[i];

    theHC->insert(aHit);
  }
  return true;
}

void FiberSD::EndOfEvent(G4HCofThisEvent * HCE) {
 
  LogDebug("FiberSim") << "FiberSD: Sees" << theHC->entries() << " hits";
  clear();
  std::cout<<"theHC entries = "<<theHC->entries()<<std::endl;
}

void FiberSD::clear() {}

void FiberSD::DrawAll()  {}

void FiberSD::PrintAll() {}

void FiberSD::update(const BeginOfJob * job) {

  const edm::EventSetup* es = (*job)();
  edm::ESHandle<HcalDDDSimConstants>    hdc;
  es->get<HcalSimNumberingRecord>().get(hdc);
  if (hdc.isValid()) {
    HcalDDDSimConstants *hcalConstants = (HcalDDDSimConstants*)(&(*hdc));
    theShower->initRun(0, hcalConstants);
  } else {
    edm::LogError("HcalSim") << "HCalSD : Cannot find HcalDDDSimConstant";
    throw cms::Exception("Unknown", "HCalSD") << "Cannot find HcalDDDSimConstant" << "\n";
  }

}

void FiberSD::update(const BeginOfRun *) {}

void FiberSD::update(const BeginOfEvent *) {}

void FiberSD::update(const ::EndOfEvent *) {}

void FiberSD::clearHits() {}

uint32_t FiberSD::setDetUnitId(G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int fibre = (touch->GetReplicaNumber(1))%10;
  int cell  = (touch->GetReplicaNumber(2));
  int tower = (touch->GetReplicaNumber(3));
  return ((tower*1000+cell)*10+fibre);
}

void FiberSD::fillHits(edm::PCaloHitContainer&, std::string) {}
