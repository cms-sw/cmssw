#include "SimG4CMS/ShowerLibraryProducer/interface/FiberSensitiveDetector.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"

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

//#define EDM_ML_DEBUG

FiberSensitiveDetector::FiberSensitiveDetector(const std::string& iname,
                                               const HcalSimulationConstants* hsps,
                                               const HcalDDDSimConstants* hdc,
                                               const SensitiveDetectorCatalog& clg,
                                               edm::ParameterSet const& p,
                                               const SimTrackManager* manager)
    : SensitiveCaloDetector(iname, clg), theShower(nullptr), theHCID(-1), theHC(nullptr) {
  edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector : Instantiating for " << iname;
  // Get pointer to HcalDDDConstants and HcalSimulationConstants
  theShower = new HFShower(iname, hdc, hsps->hcalsimpar(), p, 1);
}

FiberSensitiveDetector::~FiberSensitiveDetector() {
  delete theShower;
  delete theHC;
}

void FiberSensitiveDetector::Initialize(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector : Initialize called for " << GetName() << " in collection "
                               << HCE;
  theHC = new FiberG4HitsCollection(GetName(), collectionName[0]);
  if (theHCID < 0)
    theHCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(theHCID, theHC);
  edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector : Add hit collectrion for " << collectionName[0] << ":"
                               << theHCID << ":" << theHC;
}

G4bool FiberSensitiveDetector::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  //std::vector<HFShower::Hit> hits = theShower->getHits(aStep);
  double zoffset = 1000;
  std::vector<HFShower::Hit> hits = theShower->getHits(aStep, true, zoffset);

  if (!hits.empty()) {
    std::vector<HFShowerPhoton> thePE;
    for (unsigned int i = 0; i < hits.size(); i++) {
      //edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector :hit position z " << hits[i].position.z();
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
    //edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector :presteppoint position z " << preStepPoint->GetPosition().z();

    FiberG4Hit* aHit = new FiberG4Hit(lv, detID, depth, trackID);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector :hit size " << hits.size() << "  npe" << aHit->npe();
    edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector :pre hit position " << aHit->hitPos();
#endif
    aHit->setNpe(hits.size());
    aHit->setPos(theHitPos);
    aHit->setTime(preStepPoint->GetGlobalTime());
    aHit->setPhoton(thePE);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector :ShowerPhoton position " << thePE[0].x() << " "
                                 << thePE[0].y() << " " << thePE[0].z();

    edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector: Hit created at " << lv->GetName()
                                 << " DetID: " << aHit->towerId() << " Depth: " << aHit->depth()
                                 << " Track ID: " << aHit->trackId() << " Nb. of Cerenkov Photons: " << aHit->npe()
                                 << " Time: " << aHit->time() << " at " << aHit->hitPos();
    for (unsigned int i = 0; i < thePE.size(); i++)
      edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector: PE[" << i << "] " << thePE[i];
#endif
    theHC->insert(aHit);
  }
  return true;
}

void FiberSensitiveDetector::EndOfEvent(G4HCofThisEvent* HCE) {
  edm::LogVerbatim("FiberSim") << "FiberSensitiveDetector: finds " << theHC->entries() << " hits";
  clear();
  edm::LogVerbatim("FiberSim") << "theHC entries = " << theHC->entries();
}

void FiberSensitiveDetector::clear() {}

void FiberSensitiveDetector::DrawAll() {}

void FiberSensitiveDetector::PrintAll() {}

void FiberSensitiveDetector::update(const BeginOfJob* job) {}

void FiberSensitiveDetector::update(const BeginOfRun*) {}

void FiberSensitiveDetector::update(const BeginOfEvent*) {}

void FiberSensitiveDetector::update(const ::EndOfEvent*) {}

void FiberSensitiveDetector::clearHits() {}

uint32_t FiberSensitiveDetector::setDetUnitId(const G4Step* aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int fibre = (touch->GetReplicaNumber(1)) % 10;
  int cell = (touch->GetReplicaNumber(2));
  int tower = (touch->GetReplicaNumber(3));
  return ((tower * 1000 + cell) * 10 + fibre);
}

void FiberSensitiveDetector::fillHits(edm::PCaloHitContainer&, const std::string&) {}
