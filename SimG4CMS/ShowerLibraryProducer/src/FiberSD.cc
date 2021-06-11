#include "SimG4CMS/ShowerLibraryProducer/interface/FiberSD.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
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

FiberSD::FiberSD(const std::string& iname,
                 const edm::EventSetup& es,
                 const SensitiveDetectorCatalog& clg,
                 edm::ParameterSet const& p,
                 const SimTrackManager* manager)
    : SensitiveCaloDetector(iname, clg), m_trackManager(manager), theShower(nullptr), theHCID(-1), theHC(nullptr) {
  edm::LogVerbatim("FiberSim") << "FiberSD : Instantiating for " << iname;
  // Get pointer to HcalDDDConstants and HcalSimulationConstants
  edm::ESHandle<HcalSimulationConstants> hdsc;
  es.get<HcalSimNumberingRecord>().get(hdsc);
  if (!hdsc.isValid()) {
    edm::LogError("FiberSim") << "FiberSD : Cannot find HcalSimulationConstants";
    throw cms::Exception("Unknown", "FiberSD") << "Cannot find HcalSimulationConstants\n";
  }
  const HcalSimulationConstants* hsps = hdsc.product();
  edm::ESHandle<HcalDDDSimConstants> hdc;
  es.get<HcalSimNumberingRecord>().get(hdc);
  if (hdc.isValid()) {
    const HcalDDDSimConstants* hcalConstants = hdc.product();
    theShower = new HFShower(iname, hcalConstants, hsps->hcalsimpar(), p, 1);
  } else {
    edm::LogError("FiberSim") << "FiberSD : Cannot find HcalDDDSimConstant";
    throw cms::Exception("Unknown", "FiberSD") << "Cannot find HcalDDDSimConstant\n";
  }
}

FiberSD::~FiberSD() {
  delete theShower;
  delete theHC;
}

void FiberSD::Initialize(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "FiberSD : Initialize called for " << GetName() << " in collection " << HCE;
  theHC = new FiberG4HitsCollection(GetName(), collectionName[0]);
  if (theHCID < 0)
    theHCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(theHCID, theHC);
  edm::LogVerbatim("FiberSim") << "FiberSD : Add hit collectrion for " << collectionName[0] << ":" << theHCID << ":"
                               << theHC;
}

G4bool FiberSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  //std::vector<HFShower::Hit> hits = theShower->getHits(aStep);
  double zoffset = 1000;
  std::vector<HFShower::Hit> hits = theShower->getHits(aStep, true, zoffset);

  if (!hits.empty()) {
    std::vector<HFShowerPhoton> thePE;
    for (unsigned int i = 0; i < hits.size(); i++) {
      //edm::LogVerbatim("FiberSim") << "FiberSD :hit position z " << hits[i].position.z();
      HFShowerPhoton pe = HFShowerPhoton(
          hits[i].position.x(), hits[i].position.y(), hits[i].position.z(), hits[i].wavelength, hits[i].time);
      thePE.push_back(pe);
    }
    int trackID = aStep->GetTrack()->GetTrackID();
    G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
    const G4VTouchable* touch = preStepPoint->GetTouchable();
    G4LogicalVolume* lv = touch->GetVolume(0)->GetLogicalVolume();
    int depth = (touch->GetReplicaNumber(0)) % 10;
    int detID = setDetUnitId(aStep);
    math::XYZPoint theHitPos(
        preStepPoint->GetPosition().x(), preStepPoint->GetPosition().y(), preStepPoint->GetPosition().z());
    //edm::LogVerbatim("FiberSim") << "FiberSD :presteppoint position z " << preStepPoint->GetPosition().z();

    FiberG4Hit* aHit = new FiberG4Hit(lv, detID, depth, trackID);
    edm::LogVerbatim("FiberSim") << "FiberSD :hit size " << hits.size() << "  npe" << aHit->npe();
    edm::LogVerbatim("FiberSim") << "FiberSD :pre hit position " << aHit->hitPos();
    aHit->setNpe(hits.size());
    aHit->setPos(theHitPos);
    aHit->setTime(preStepPoint->GetGlobalTime());
    aHit->setPhoton(thePE);
    edm::LogVerbatim("FiberSim") << "FiberSD :ShowerPhoton position " << thePE[0].x() << " " << thePE[0].y() << " "
                                 << thePE[0].z();

    edm::LogVerbatim("FiberSim") << "FiberSD: Hit created at " << lv->GetName() << " DetID: " << aHit->towerId()
                                 << " Depth: " << aHit->depth() << " Track ID: " << aHit->trackId()
                                 << " Nb. of Cerenkov Photons: " << aHit->npe() << " Time: " << aHit->time() << " at "
                                 << aHit->hitPos();
    for (unsigned int i = 0; i < thePE.size(); i++)
      edm::LogVerbatim("FiberSim") << "FiberSD: PE[" << i << "] " << thePE[i];

    theHC->insert(aHit);
  }
  return true;
}

void FiberSD::EndOfEvent(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "FiberSD: finds " << theHC->entries() << " hits";
  clear();
  edm::LogVerbatim("FiberSim") << "theHC entries = " << theHC->entries();
}

void FiberSD::clear() {}

void FiberSD::DrawAll() {}

void FiberSD::PrintAll() {}

void FiberSD::update(const BeginOfJob* job) {}

void FiberSD::update(const BeginOfRun*) {}

void FiberSD::update(const BeginOfEvent*) {}

void FiberSD::update(const ::EndOfEvent*) {}

void FiberSD::clearHits() {}

uint32_t FiberSD::setDetUnitId(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int fibre = (touch->GetReplicaNumber(1)) % 10;
  int cell = (touch->GetReplicaNumber(2));
  int tower = (touch->GetReplicaNumber(3));
  return ((tower * 1000 + cell) * 10 + fibre);
}

void FiberSD::fillHits(edm::PCaloHitContainer&, const std::string&) {}
