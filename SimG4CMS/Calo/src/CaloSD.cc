///////////////////////////////////////////////////////////////////////////////
// File: CaloSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"
#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4EventManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4GFlashSpot.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"

#include <CLHEP/Units/SystemOfUnits.h>

#include <fstream>
#include <memory>
#include <sstream>

//#define EDM_ML_DEBUG

CaloSD::CaloSD(const std::string& name,
               const SensitiveDetectorCatalog& clg,
               edm::ParameterSet const& p,
               const SimTrackManager* manager,
               float timeSliceUnit,
               bool ignoreTkID,
               const std::string& newcollName)
    : SensitiveCaloDetector(name, clg, newcollName),
      G4VGFlashSensitiveDetector(),
      eminHit(0.),
      m_trackManager(manager),
      ignoreTrackID(ignoreTkID),
      timeSlice(timeSliceUnit),
      eminHitD(0.) {
  //Parameters
  nHC_ = (newcollName.empty()) ? 1 : 2;
  for (int k = 0; k < 2; ++k) {
    currentHit[k] = nullptr;
    theHC[k] = nullptr;
    hcID[k] = -1;
  }
  bool dd4hep = p.getParameter<bool>("g4GeometryDD4hepSource");
  int addlevel = dd4hep ? 1 : 0;
  edm::ParameterSet m_CaloSD = p.getParameter<edm::ParameterSet>("CaloSD");
  energyCut = m_CaloSD.getParameter<double>("EminTrack") * CLHEP::GeV;
  tmaxHit = m_CaloSD.getParameter<double>("TmaxHit") * CLHEP::ns;
  std::vector<double> eminHits = m_CaloSD.getParameter<std::vector<double>>("EminHits");
  std::vector<double> tmaxHits = m_CaloSD.getParameter<std::vector<double>>("TmaxHits");
  hcn_ = m_CaloSD.getParameter<std::vector<std::string>>("HCNames");
  useResMap_ = m_CaloSD.getParameter<std::vector<int>>("UseResponseTables");
  std::vector<double> eminHitX = m_CaloSD.getParameter<std::vector<double>>("EminHitsDepth");
  suppressHeavy = m_CaloSD.getParameter<bool>("SuppressHeavy");
  kmaxIon = m_CaloSD.getParameter<double>("IonThreshold") * CLHEP::MeV;
  kmaxProton = m_CaloSD.getParameter<double>("ProtonThreshold") * CLHEP::MeV;
  kmaxNeutron = m_CaloSD.getParameter<double>("NeutronThreshold") * CLHEP::MeV;
  nCheckedHits[0] = m_CaloSD.getUntrackedParameter<int>("CheckHits", 25);
  nCheckedHits[1] = nCheckedHits[0];
  useMap = m_CaloSD.getUntrackedParameter<bool>("UseMap", true);
  int verbn = m_CaloSD.getUntrackedParameter<int>("Verbosity", 0);
  corrTOFBeam = m_CaloSD.getParameter<bool>("CorrectTOFBeam");
  double beamZ = m_CaloSD.getParameter<double>("BeamPosition") * CLHEP::cm;
  correctT = beamZ / CLHEP::c_light / CLHEP::nanosecond;
  doFineCalo_ = m_CaloSD.getParameter<bool>("DoFineCalo");
  eMinFine_ = m_CaloSD.getParameter<double>("EminFineTrack") * CLHEP::MeV;
  std::vector<std::string> fineNames = m_CaloSD.getParameter<std::vector<std::string>>("FineCaloNames");
  std::vector<int> fineLevels = m_CaloSD.getParameter<std::vector<int>>("FineCaloLevels");
  std::vector<int> useFines = m_CaloSD.getParameter<std::vector<int>>("UseFineCalo");
  for (auto& level : fineLevels)
    level += addlevel;

  SetVerboseLevel(verbn);
  for (int k = 0; k < 2; ++k)
    meanResponse[k].reset(nullptr);
  for (unsigned int k = 0; k < hcn_.size(); ++k) {
    if (name == hcn_[k]) {
      if (k < eminHits.size())
        eminHit = eminHits[k] * CLHEP::MeV;
      if (k < eminHitX.size())
        eminHitD = eminHitX[k] * CLHEP::MeV;
      if (k < tmaxHits.size())
        tmaxHit = tmaxHits[k] * CLHEP::ns;
      if (k < useResMap_.size() && useResMap_[k] > 0) {
        meanResponse[0] = std::make_unique<CaloMeanResponse>(p);
        break;
      }
    }
  }
  detName_ = name;
  collName_[0] = name;
  collName_[1] = newcollName;
  slave[0] = std::make_unique<CaloSlaveSD>(name);
  slave[1].reset(nullptr);

  for (int k = 0; k < 2; ++k) {
    currentID[k] = CaloHitID(timeSlice, ignoreTrackID);
    previousID[k] = CaloHitID(timeSlice, ignoreTrackID);
  }
  isParameterized = false;

  entrancePoint.set(0., 0., 0.);
  entranceLocal.set(0., 0., 0.);
  posGlobal.set(0., 0., 0.);
  incidentEnergy = edepositEM = edepositHAD = 0.f;

  primAncestor = 0;
  forceSave = false;
  for (int k = 0; k < 2; ++k) {
    cleanIndex[k] = primIDSaved[k] = totalHits[k] = 0;
  }

  edm::LogVerbatim("CaloSim") << "CaloSD: Minimum energy of track for saving it " << energyCut / CLHEP::GeV
                              << " GeV\n        Use of HitID Map " << useMap << "\n        Check last "
                              << nCheckedHits[0] << " before saving the hit\n        Correct TOF globally by "
                              << correctT << " ns (Flag =" << corrTOFBeam << ")\n        Save hits recorded before "
                              << tmaxHit << " ns and if energy is above " << eminHit / CLHEP::MeV
                              << " MeV (for depth 0) or " << eminHitD / CLHEP::MeV
                              << " MeV (for nonzero depths);\n        Time Slice Unit " << timeSlice
                              << "\nIgnore TrackID Flag " << ignoreTrackID << " doFineCalo flag " << doFineCalo_
                              << "\nBeam Position " << beamZ / CLHEP::cm << " cm";
  if (doFineCalo_)
    edm::LogVerbatim("DoFineCalo") << "Using finecalo v2";

  // Treat fine calorimeters
  edm::LogVerbatim("CaloSim") << "CaloSD: Have a possibility of " << fineNames.size() << " fine calorimeters of which "
                              << useFines.size() << " are selected";
  for (unsigned int k = 0; k < fineNames.size(); ++k)
    edm::LogVerbatim("CaloSim") << "[" << k << "] " << fineNames[k] << " at " << fineLevels[k];
  std::ostringstream st1;
  for (unsigned int k = 0; k < useFines.size(); ++k)
    st1 << " [" << k << "] " << useFines[k] << ":" << fineNames[useFines[k]];
  edm::LogVerbatim("CaloSim") << "CaloSD used calorimeters" << st1.str();
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume*>::const_iterator lvcite;
  for (unsigned int i = 0; i < useFines.size(); i++) {
    G4LogicalVolume* lv = nullptr;
    G4String name = static_cast<G4String>(fineNames[useFines[i]]);
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
      G4String namx(static_cast<G4String>(DD4hep2DDDName::nameMatterLV((*lvcite)->GetName(), dd4hep)));
      if (namx == name) {
        lv = (*lvcite);
        break;
      }
    }
    if (lv != nullptr) {
      CaloSD::Detector detector;
      detector.name = name;
      detector.lv = lv;
      detector.level = fineLevels[useFines[i]];
      fineDetectors_.emplace_back(detector);
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::Loads information for " << fineDetectors_.size() << " fine detectors";
  unsigned int k(0);
  for (const auto& detector : fineDetectors_) {
    edm::LogVerbatim("CaloSim") << "Detector[" << k << "] " << detector.name << " at level " << detector.level
                                << " pointer to LV: " << detector.lv;
  }
#endif
}

CaloSD::~CaloSD() {}

void CaloSD::newCollection(const std::string& name, edm::ParameterSet const& p) {
  for (unsigned int k = 0; k < hcn_.size(); ++k) {
    if (name == hcn_[k]) {
      if (k < useResMap_.size() && useResMap_[k] > 0) {
        meanResponse[1] = std::make_unique<CaloMeanResponse>(p);
        break;
      }
    }
  }
  slave[1] = std::make_unique<CaloSlaveSD>(name);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD:: Initialise a second collection for " << name;
  for (int k = 0; k < nHC_; ++k)
    edm::LogVerbatim("CaloSim") << "Detector[" << k << "]" << detName_ << " collection " << collName_[k];

#endif
}

G4bool CaloSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  NaNTrap(aStep);
  ignoreReject = false;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::" << GetName() << " ID= " << aStep->GetTrack()->GetTrackID()
                              << " prID= " << aStep->GetTrack()->GetParentID()
                              << " Eprestep= " << aStep->GetPreStepPoint()->GetKineticEnergy()
                              << " step= " << aStep->GetStepLength() << " Edep= " << aStep->GetTotalEnergyDeposit();
#endif

  // Class variable to determine whether finecalo rules should apply for this step
  doFineCaloThisStep_ = (doFineCalo_ && isItFineCalo(aStep->GetPreStepPoint()->GetTouchable()));

  // apply shower library or parameterisation
  // independent on energy deposition at a step
  if (isParameterized) {
    if (getFromLibrary(aStep)) {
      // for parameterized showers the primary track should be killed
      // secondary tracks should be killed if they are in the same volume
      (aStep->GetTrack())->SetTrackStatus(fStopAndKill);
      if (0 < aStep->GetNumberOfSecondariesInCurrentStep()) {
        auto tv = aStep->GetSecondaryInCurrentStep();
        auto vol = aStep->GetPreStepPoint()->GetPhysicalVolume();
        for (auto& tk : *tv) {
          if (tk->GetVolume() == vol) {
            const_cast<G4Track*>(tk)->SetTrackStatus(fStopAndKill);
          }
        }
      }
      return true;
    }
  }

  // ignore steps without energy deposit
  edepositEM = edepositHAD = 0.f;
  if (aStep->GetTotalEnergyDeposit() <= 0.0) {
    return false;
  }

  // check unitID
  unsigned int unitID = setDetUnitId(aStep);
  auto const theTrack = aStep->GetTrack();
  uint16_t depth = getDepth(aStep);

  double time = theTrack->GetGlobalTime() / CLHEP::nanosecond;
  int primaryID = getTrackID(theTrack);
  if (unitID > 0) {
    currentID[0].setID(unitID, time, primaryID, depth);
  } else {
    if (!ignoreReject) {
      const G4TouchableHistory* touch = static_cast<const G4TouchableHistory*>(theTrack->GetTouchable());
      edm::LogVerbatim("CaloSim") << "CaloSD::ProcessHits: unitID= " << unitID
                                  << " currUnit=   " << currentID[0].unitID() << " Detector: " << GetName()
                                  << " trackID= " << theTrack->GetTrackID() << " "
                                  << theTrack->GetDefinition()->GetParticleName()
                                  << "\n Edep= " << aStep->GetTotalEnergyDeposit()
                                  << " PV: " << touch->GetVolume(0)->GetName()
                                  << " PVid= " << touch->GetReplicaNumber(0) << " MVid= " << touch->GetReplicaNumber(1);
    }
    return false;
  }
  double energy = getEnergyDeposit(aStep);
  if (energy <= 0.0) {
    return false;
  }

  if (doFineCaloThisStep_) {
    currentID[0].setID(unitID, time, findBoundaryCrossingParent(theTrack), depth);
    currentID[0].markAsFinecaloTrackID();
  }

  if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
    edepositEM = energy;
  } else {
    edepositHAD = energy;
  }
#ifdef EDM_ML_DEBUG
  const G4TouchableHistory* touch = static_cast<const G4TouchableHistory*>(theTrack->GetTouchable());
  edm::LogVerbatim("CaloSim") << "CaloSD::" << GetName() << " PV:" << touch->GetVolume(0)->GetName()
                              << " PVid=" << touch->GetReplicaNumber(0) << " MVid=" << touch->GetReplicaNumber(1)
                              << " Unit:" << std::hex << unitID << std::dec << " Edep=" << edepositEM << " "
                              << edepositHAD << " ID=" << theTrack->GetTrackID() << " pID=" << theTrack->GetParentID()
                              << " E=" << theTrack->GetKineticEnergy() << " S=" << aStep->GetStepLength() << "\n "
                              << theTrack->GetDefinition()->GetParticleName() << " primaryID= " << primaryID
                              << " currentID= (" << currentID[0] << ") previousID= (" << previousID[0] << ")";
#endif
  if (!hitExists(aStep, 0)) {
    currentHit[0] = createNewHit(aStep, theTrack, 0);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "Not creating new hit, only updating " << shortreprID(currentHit[0]);
#endif
  }

  processSecondHit(aStep, theTrack);

  return true;
}

bool CaloSD::ProcessHits(G4GFlashSpot* aSpot, G4TouchableHistory*) {
  edepositEM = edepositHAD = 0.f;
  const G4Track* track = aSpot->GetOriginatorTrack()->GetPrimaryTrack();

  double edep = aSpot->GetEnergySpot()->GetEnergy();
  if (edep <= 0.0) {
    return false;
  }

  G4Step fFakeStep;
  G4StepPoint* fFakePreStepPoint = fFakeStep.GetPreStepPoint();
  G4StepPoint* fFakePostStepPoint = fFakeStep.GetPostStepPoint();
  fFakePreStepPoint->SetPosition(aSpot->GetPosition());
  fFakePostStepPoint->SetPosition(aSpot->GetPosition());

  G4TouchableHandle fTouchableHandle = aSpot->GetTouchableHandle();
  fFakePreStepPoint->SetTouchableHandle(fTouchableHandle);
  fFakeStep.SetTotalEnergyDeposit(edep);
  edep = EnergyCorrected(fFakeStep, track);

  // zero edep means hit outside the calorimeter
  if (edep <= 0.0) {
    return false;
  }

  if (G4TrackToParticleID::isGammaElectronPositron(track)) {
    edepositEM = edep;
  } else {
    edepositHAD = edep;
  }

  unsigned int unitID = setDetUnitId(&fFakeStep);

  if (unitID > 0) {
    // time of initial track
    double time = track->GetGlobalTime() / CLHEP::nanosecond;
    int primaryID = getTrackID(track);
    uint16_t depth = getDepth(&fFakeStep);
    currentID[0].setID(unitID, time, primaryID, depth);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD:: GetSpotInfo for Unit 0x" << std::hex << currentID[0].unitID() << std::dec
                                << " Edeposit = " << edepositEM << " " << edepositHAD;
#endif
    // Update if in the same detector, time-slice and for same track
    if (currentID[0] == previousID[0]) {
      updateHit(currentHit[0], 0);
    } else {
      posGlobal = aSpot->GetEnergySpot()->GetPosition();
      // Reset entry point for new primary
      if (currentID[0].trackID() != previousID[0].trackID()) {
        entrancePoint = aSpot->GetPosition();
        entranceLocal = aSpot->GetTouchableHandle()->GetHistory()->GetTopTransform().TransformPoint(entrancePoint);
        incidentEnergy = track->GetKineticEnergy();
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("CaloSim") << "CaloSD: Incident energy " << incidentEnergy / CLHEP::GeV << " GeV and"
                                    << " entrance point " << entrancePoint << " (Global) " << entranceLocal
                                    << " (Local)";
#endif
      }
      if (!checkHit()) {
        currentHit[0] = createNewHit(&fFakeStep, track, 0);
      }
    }
    return true;
  }
  return false;
}

double CaloSD::getEnergyDeposit(const G4Step* aStep) { return aStep->GetTotalEnergyDeposit(); }

double CaloSD::EnergyCorrected(const G4Step& aStep, const G4Track*) { return aStep.GetTotalEnergyDeposit(); }

bool CaloSD::getFromLibrary(const G4Step*) { return false; }

bool CaloSD::isItFineCalo(const G4VTouchable* touch) {
  bool ok(false);
  int level = ((touch->GetHistoryDepth()) + 1);
  for (const auto& detector : fineDetectors_) {
    if (level > 0 && level >= detector.level) {
      int ii = level - detector.level;
      G4LogicalVolume* lv = touch->GetVolume(ii)->GetLogicalVolume();
      ok = (lv == detector.lv);
#ifdef EDM_ML_DEBUG
      std::string name1 = (lv == nullptr) ? "Unknown" : lv->GetName();
      edm::LogVerbatim("CaloSim") << "CaloSD: volume " << name1 << ":" << detector.name << " at Level "
                                  << detector.level << " Flag " << ok;
#endif
      if (ok)
        break;
    }
  }
  return ok;
}

void CaloSD::Initialize(G4HCofThisEvent* HCE) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD : Initialize called for " << detName_;
#endif

  for (int k = 0; k < nHC_; ++k) {
    totalHits[k] = 0;

    //This initialization is performed at the beginning of an event
    //------------------------------------------------------------
    theHC[k] = new CaloG4HitCollection(detName_, collName_[k]);

    if (hcID[k] < 0) {
      hcID[k] = G4SDManager::GetSDMpointer()->GetCollectionID(collName_[k]);
    }
    //theHC ownership is transfered here to HCE
    HCE->AddHitsCollection(hcID[k], theHC[k]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD : Add hits collection for " << hcID[k] << " for " << k << ":" << detName_
                                << ":" << collName_[k];
#endif
  }
}

void CaloSD::EndOfEvent(G4HCofThisEvent*) {
  // clean the hits for the last tracks

  cleanHitCollection();

#ifdef EDM_ML_DEBUG
  for (int k = 0; k < nHC_; ++k) {
    if (theHC[k] == nullptr)
      edm::LogVerbatim("CaloSim") << "CaloSD: EndofEvent entered for container " << k << " with no entries";
    else
      edm::LogVerbatim("CaloSim") << "CaloSD: EndofEvent entered for container " << k << " with " << theHC[k]->entries()
                                  << " entries";
  }
#endif
}

void CaloSD::clear() {}

void CaloSD::DrawAll() {}

void CaloSD::PrintAll() {
  for (int k = 0; k < nHC_; ++k) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: Collection " << theHC[k]->GetName();
#endif
    theHC[k]->PrintAllHits();
  }
}

void CaloSD::fillHits(edm::PCaloHitContainer& cc, const std::string& hname) {
  for (int k = 0; k < nHC_; ++k) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: Tries to transfer " << slave[k].get()->hits().size() << " hits for "
                                << slave[k].get()->name() << "   " << hname;
#endif
    if (slave[k].get()->name() == hname) {
      cc = slave[k].get()->hits();
      slave[k].get()->Clean();
    }
  }
}

G4ThreeVector CaloSD::setToLocal(const G4ThreeVector& global, const G4VTouchable* touch) const {
  return touch->GetHistory()->GetTopTransform().TransformPoint(global);
}

G4ThreeVector CaloSD::setToGlobal(const G4ThreeVector& local, const G4VTouchable* touch) const {
  return touch->GetHistory()->GetTopTransform().Inverse().TransformPoint(local);
}

bool CaloSD::hitExists(const G4Step* aStep, int k) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: hitExists for " << k;
#endif
  // Update if in the same detector, time-slice and for same track
  if (currentID[k] == previousID[k]) {
    updateHit(currentHit[k], k);
    return true;
  }

  // Note T. Klijnsma:
  // This is a rather strange place to set these class variables.
  // The code would be much more readable if all logic for determining
  // whether to update a hit or create a new hit is done in one place,
  // and only then perform the actual updating or creating of the hit.

  // Reset entry point for new primary (only for the primary hit)
  if (k == 0) {
    posGlobal = aStep->GetPreStepPoint()->GetPosition();
    if (currentID[k].trackID() != previousID[k].trackID()) {
      resetForNewPrimary(aStep);
    }
  }
  return checkHit(k);
}

bool CaloSD::checkHit(int k) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: checkHit for " << k << " for map " << useMap << ":" << &hitMap[k] << " Nhits "
                              << nCheckedHits[k] << " HC " << theHC[k] << " ID " << currentID[k];
#endif
  //look in the HitContainer whether a hit with the same ID already exists:
  bool found = false;
  if (useMap) {
    std::map<CaloHitID, CaloG4Hit*>::const_iterator it = hitMap[k].find(currentID[k]);
    if (it != hitMap[k].end()) {
      currentHit[k] = it->second;
      found = true;
    }
  } else if (nCheckedHits[k] > 0) {
    int nhits = theHC[k]->entries();
    int minhit = std::max(nhits - nCheckedHits[k], 0);
    int maxhit = nhits - 1;

    for (int j = maxhit; j > minhit; --j) {
      if ((*theHC[k])[j]->getID() == currentID[k]) {
        currentHit[k] = (*theHC[k])[j];
        found = true;
        break;
      }
    }
  }

  if (found) {
    updateHit(currentHit[k], k);
  }
  return found;
}

int CaloSD::getNumberOfHits(int k) { return theHC[k]->entries(); }

/*
Takes a vector of ints (representing trackIDs), and returns a formatted string
for debugging purposes
*/
std::string CaloSD::printableDecayChain(const std::vector<unsigned int>& decayChain) {
  std::stringstream ss;
  for (long unsigned int i = 0; i < decayChain.size(); i++) {
    if (i > 0)
      ss << " <- ";
    ss << decayChain[i];
  }
  return ss.str();
}

/* Very short representation of a CaloHitID */
std::string CaloSD::shortreprID(const CaloHitID& ID) {
  std::stringstream ss;
  ss << GetName() << "/" << ID.unitID() << "/trk" << ID.trackID() << "/d" << ID.depth() << "/time" << ID.timeSliceID();
  if (ID.isFinecaloTrackID())
    ss << "/FC";
  return ss.str();
}

/* As above, but with a hit as input */
std::string CaloSD::shortreprID(const CaloG4Hit* hit) { return shortreprID(hit->getID()); }

/*
Finds the boundary-crossing parent of a track, and stores it in the CaloSD's map
*/
unsigned int CaloSD::findBoundaryCrossingParent(const G4Track* track, bool markAsSaveable) {
  TrackInformation* trkInfo = cmsTrackInformation(track);
  unsigned int id = track->GetTrackID();
  // First see if this track is already in the map
  auto it = boundaryCrossingParentMap_.find(id);
  if (it != boundaryCrossingParentMap_.end()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "Track " << id << " parent already cached: " << it->second;
#endif
    return it->second;
  }
  // Then see if the track itself crosses the boundary
  else if (trkInfo->crossedBoundary()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "Track " << id << " crosses boundary itself";
#endif
    boundaryCrossingParentMap_[id] = id;
    trkInfo->setStoreTrack();
    return id;
  }
  // Else, traverse the history of the track
  std::vector<unsigned int> decayChain{id};
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DoFineCalo") << "Track " << id << ": Traversing history to find boundary-crossing parent";
#endif
  unsigned int parentID = track->GetParentID();
  while (true) {
    if (parentID == 0)
      throw cms::Exception("Unknown", "CaloSD")
          << "Hit end of parentage for track " << id << " without finding a boundary-crossing parent";
    // First check if this ancestor is already in the map
    auto it = boundaryCrossingParentMap_.find(parentID);
    if (it != boundaryCrossingParentMap_.end()) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DoFineCalo") << "  Track " << parentID
                                     << " boundary-crossing parent already cached: " << it->second;
#endif
      // Store this parent also for the rest of the traversed decay chain
      for (auto ancestorID : decayChain)
        boundaryCrossingParentMap_[ancestorID] = it->second;
#ifdef EDM_ML_DEBUG
      // In debug mode, still build the rest of the decay chain for debugging
      decayChain.push_back(parentID);
      while (parentID != it->second) {
        parentID = m_trackManager->getTrackByID(parentID, true)->parentID();
        decayChain.push_back(parentID);
      }
      edm::LogVerbatim("DoFineCalo") << "  Full decay chain: " << printableDecayChain(decayChain);
#endif
      return it->second;
    }
    // If not, get this parent from the track manager (expensive)
    TrackWithHistory* parentTrack = m_trackManager->getTrackByID(parentID, true);
    if (parentTrack->crossedBoundary()) {
      if (markAsSaveable)
        parentTrack->setToBeSaved();
      decayChain.push_back(parentID);
      // Record this boundary crossing parent for all traversed ancestors
      for (auto ancestorID : decayChain)
        boundaryCrossingParentMap_[ancestorID] = parentID;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DoFineCalo") << "  Found boundary-crossing ancestor " << parentID << " for track " << id
                                     << "; decay chain: " << printableDecayChain(decayChain);
#endif
      return parentID;
    }
    // Next iteration
    decayChain.push_back(parentID);
    parentID = parentTrack->parentID();
  }
}

CaloG4Hit* CaloSD::createNewHit(const G4Step* aStep, const G4Track* theTrack, int k) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::CreateNewHit " << getNumberOfHits() << " for " << GetName()
                              << " Unit:" << currentID[k].unitID() << " " << currentID[k].depth()
                              << " Edep= " << edepositEM << " " << edepositHAD
                              << " primaryID= " << currentID[k].trackID()
                              << " timeSlice= " << currentID[k].timeSliceID() << " ID= " << theTrack->GetTrackID()
                              << " " << theTrack->GetDefinition()->GetParticleName()
                              << " E(GeV)= " << theTrack->GetKineticEnergy() / CLHEP::GeV
                              << " parentID= " << theTrack->GetParentID() << "\n Ein= " << incidentEnergy
                              << " entranceGlobal: " << entrancePoint << " entranceLocal: " << entranceLocal
                              << " posGlobal: " << posGlobal;
#endif

  CaloG4Hit* aHit;
  if (!reusehit[k].empty()) {
    aHit = reusehit[k].back().release();
    aHit->setEM(0.f);
    aHit->setHadr(0.f);
    reusehit[k].pop_back();
  } else {
    aHit = new CaloG4Hit;
  }

  aHit->setID(currentID[k]);
  aHit->setEntry(entrancePoint.x(), entrancePoint.y(), entrancePoint.z());
  aHit->setEntryLocal(entranceLocal.x(), entranceLocal.y(), entranceLocal.z());
  aHit->setPosition(posGlobal.x(), posGlobal.y(), posGlobal.z());
  aHit->setIncidentEnergy(incidentEnergy);
  updateHit(aHit, k);

  storeHit(aHit, k);
  TrackInformation* trkInfo = cmsTrackInformation(theTrack);

#ifdef EDM_ML_DEBUG
  if (doFineCaloThisStep_)
    edm::LogVerbatim("DoFineCalo") << "New hit " << shortreprID(aHit) << " using finecalo;"
                                   << " isItFineCalo(post)=" << isItFineCalo(aStep->GetPostStepPoint()->GetTouchable())
                                   << " isItFineCalo(pre)=" << isItFineCalo(aStep->GetPreStepPoint()->GetTouchable());
#endif

  // 'Traditional', non-fine history bookkeeping
  if (!doFineCaloThisStep_) {
    double etrack = 0;
    if (currentID[k].trackID() == primIDSaved[k]) {  // The track is saved; nothing to be done
    } else if (currentID[k].trackID() == theTrack->GetTrackID()) {
      etrack = theTrack->GetKineticEnergy();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CaloSim") << "CaloSD: set save the track " << currentID[k].trackID() << " etrack " << etrack
                                  << " eCut " << energyCut << " force: " << forceSave
                                  << " save: " << (etrack >= energyCut || forceSave);
#endif
      if (etrack >= energyCut || forceSave) {
        trkInfo->setStoreTrack();
      }
    } else {
      TrackWithHistory* trkh = tkMap[currentID[k].trackID()];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CaloSim") << "CaloSD : TrackWithHistory pointer for " << currentID[k].trackID() << " is "
                                  << trkh;
#endif
      if (trkh != nullptr) {
        etrack = sqrt(trkh->momentum().Mag2());
        if (etrack >= energyCut) {
          trkh->setToBeSaved();
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("CaloSim") << "CaloSD: set save the track " << currentID[k].trackID() << " with Hit";
#endif
        }
      }
    }
    primIDSaved[k] = currentID[k].trackID();
  }

  if (useMap)
    ++totalHits[k];
  return aHit;
}

void CaloSD::updateHit(CaloG4Hit* aHit, int k) {
  aHit->addEnergyDeposit(edepositEM, edepositHAD);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD:" << GetName() << " Add energy deposit in " << currentID[k]
                              << " Edep_em(MeV)= " << edepositEM << " Edep_had(MeV)= " << edepositHAD;
#endif

  // buffer for next steps:
  previousID[k] = currentID[k];
}

void CaloSD::resetForNewPrimary(const G4Step* aStep) {
  auto const preStepPoint = aStep->GetPreStepPoint();
  entrancePoint = preStepPoint->GetPosition();
  entranceLocal = setToLocal(entrancePoint, preStepPoint->GetTouchable());
  incidentEnergy = preStepPoint->GetKineticEnergy();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::resetForNewPrimary for " << GetName()
                              << " ID= " << aStep->GetTrack()->GetTrackID() << " Ein= " << incidentEnergy / CLHEP::GeV
                              << " GeV and"
                              << " entrance point global: " << entrancePoint << " local: " << entranceLocal;
#endif
}

double CaloSD::getAttenuation(const G4Step* aStep, double birk1, double birk2, double birk3) const {
  double weight = 1.;
  double charge = aStep->GetPreStepPoint()->GetCharge();
  double length = aStep->GetStepLength();

  if (charge != 0. && length > 0.) {
    double density = aStep->GetPreStepPoint()->GetMaterial()->GetDensity();
    double dedx = aStep->GetTotalEnergyDeposit() / length;
    double rkb = birk1 / density;
    double c = birk2 * rkb * rkb;
    if (std::abs(charge) >= 2.)
      rkb /= birk3;  // based on alpha particle data
    weight = 1. / (1. + rkb * dedx + c * dedx * dedx);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD::getAttenuation in " << aStep->GetPreStepPoint()->GetMaterial()->GetName()
                                << " Charge " << charge << " dE/dx " << dedx << " Birk Const " << rkb << ", " << c
                                << " Weight = " << weight << " dE " << aStep->GetTotalEnergyDeposit();
#endif
  }
  return weight;
}

void CaloSD::update(const BeginOfRun*) { initRun(); }

void CaloSD::update(const BeginOfEvent* g4Event) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: Dispatched BeginOfEvent for " << GetName() << " !";
#endif
  clearHits();
  initEvent(g4Event);
}

void CaloSD::update(const EndOfTrack* trk) {
  int id = (*trk)()->GetTrackID();
  TrackInformation* trkI = cmsTrackInformation((*trk)());
  int lastTrackID = -1;
  if (trkI)
    lastTrackID = trkI->getIDonCaloSurface();
  if (id == lastTrackID) {
    auto trksForThisEvent = m_trackManager->trackContainer();
    if (!trksForThisEvent->empty()) {
      TrackWithHistory* trkH = trksForThisEvent->back();
      if (trkH->trackID() == id) {
        tkMap[id] = trkH;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("CaloSim") << "CaloSD: get track " << id << " from Container of size "
                                    << trksForThisEvent->size() << " with ID " << trkH->trackID();
#endif
      }
    }
  }
}

void CaloSD::update(const ::EndOfEvent*) {
  endEvent();
  for (int k = 0; k < nHC_; ++k) {
    slave[k].get()->ReserveMemory(theHC[k]->entries());

    int count(0);
    double eEM(0.0);
    double eHAD(0.0);
    double eEM2(0.0);
    double eHAD2(0.0);
#ifdef EDM_ML_DEBUG
    int wrong(0);
    double tt(0.0);
    double zloc(0.0);
    double zglob(0.0);
    double ee(0.0);
#endif
    int hc_entries = theHC[k]->entries();
    for (int i = 0; i < hc_entries; ++i) {
#ifdef EDM_ML_DEBUG
      if (!saveHit((*theHC[k])[i], k)) {
        ++wrong;
      }
#else
      saveHit((*theHC[k])[i], k);
#endif

      ++count;
      double x = (*theHC[k])[i]->getEM();
      eEM += x;
      eEM2 += x * x;
      x = (*theHC[k])[i]->getHadr();
      eHAD += x;
      eHAD2 += x * x;
#ifdef EDM_ML_DEBUG
      tt += (*theHC[k])[i]->getTimeSlice();
      ee += (*theHC[k])[i]->getIncidentEnergy();
      zglob += std::abs((*theHC[k])[i]->getEntry().z());
      zloc += std::abs((*theHC[k])[i]->getEntryLocal().z());
#endif
    }

    double norm = (count > 0) ? 1.0 / count : 0.0;
    eEM *= norm;
    eEM2 *= norm;
    eHAD *= norm;
    eHAD2 *= norm;
    eEM2 = std::sqrt(eEM2 - eEM * eEM);
    eHAD2 = std::sqrt(eHAD2 - eHAD * eHAD);
#ifdef EDM_ML_DEBUG
    tt *= norm;
    ee *= norm;
    zglob *= norm;
    zloc *= norm;
    edm::LogVerbatim("CaloSim") << "CaloSD: " << GetName() << " store " << count << " hits; " << wrong
                                << " track IDs not given properly and " << totalHits - count
                                << " hits not passing cuts\n EmeanEM= " << eEM << " ErmsEM= " << eEM2
                                << "\n EmeanHAD= " << eHAD << " ErmsHAD= " << eHAD2 << " TimeMean= " << tt
                                << " E0mean= " << ee << " Zglob= " << zglob << " Zloc= " << zloc << " ";
#endif
    std::vector<std::unique_ptr<CaloG4Hit>>().swap(reusehit[k]);
    if (useMap)
      hitMap[k].erase(hitMap[k].begin(), hitMap[k].end());
  }
  tkMap.erase(tkMap.begin(), tkMap.end());
  boundaryCrossingParentMap_.clear();
}

void CaloSD::clearHits() {
  for (int k = 0; k < nHC_; ++k) {
    previousID[k].reset();
    primIDSaved[k] = -99;
    cleanIndex[k] = 0;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: Clears hit vector for " << GetName()
                                << " and initialise slave: " << slave[k].get()->name();
#endif
    slave[k].get()->Initialize();
  }
}

void CaloSD::reset() {
  if (fpCaloG4HitAllocator) {
    fpCaloG4HitAllocator->ResetStorage();
  }
}

void CaloSD::initRun() {}

void CaloSD::initEvent(const BeginOfEvent*) {}

void CaloSD::endEvent() {}

int CaloSD::getTrackID(const G4Track* aTrack) {
  int primaryID = 0;
  TrackInformation* trkInfo = cmsTrackInformation(aTrack);
  if (trkInfo) {
    primaryID = trkInfo->getIDonCaloSurface();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "Track ID: " << trkInfo->getIDonCaloSurface() << ":" << aTrack->GetTrackID() << ":"
                                << primaryID;
#endif
  } else {
    primaryID = aTrack->GetTrackID();
#ifdef EDM_ML_DEBUG
    edm::LogWarning("CaloSim") << "CaloSD: Problem with primaryID **** set by force to TkID **** " << primaryID;
#endif
  }
  return primaryID;
}

int CaloSD::setTrackID(const G4Step* aStep) {
  auto const theTrack = aStep->GetTrack();
  TrackInformation* trkInfo = cmsTrackInformation(theTrack);
  int primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID <= 0) {
    primaryID = theTrack->GetTrackID();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "Track ID: " << trkInfo->getIDonCaloSurface() << ":" << theTrack->GetTrackID() << ":"
                              << primaryID;
#endif

  if (primaryID != previousID[0].trackID()) {
    resetForNewPrimary(aStep);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::setTrackID for " << GetName()
                              << " trackID= " << aStep->GetTrack()->GetTrackID() << " primaryID= " << primaryID;
#endif
  return primaryID;
}

uint16_t CaloSD::getDepth(const G4Step*) { return 0; }

void CaloSD::processSecondHit(const G4Step*, const G4Track*) {}

bool CaloSD::filterHit(CaloG4Hit* hit, double time) {
  double emin(eminHit);
  if (hit->getDepth() > 0)
    emin = eminHitD;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::filterHit(..) Depth " << hit->getDepth() << " Emin = " << emin << " ("
                              << eminHit << ", " << eminHitD << ")";
#endif
  return ((time <= tmaxHit) && (hit->getEnergyDeposit() > emin));
}

double CaloSD::getResponseWt(const G4Track* aTrack, int k) {
  double wt = 1.0;
  if (meanResponse[k].get()) {
    TrackInformation* trkInfo = cmsTrackInformation(aTrack);
    wt = meanResponse[k].get()->getWeight(trkInfo->genParticlePID(), trkInfo->genParticleP());
  }
  return wt;
}

void CaloSD::storeHit(CaloG4Hit* hit, int k) {
  if (hit == nullptr || previousID[k].trackID() < 0) {
    edm::LogWarning("CaloSim") << "CaloSD: hit to be stored is nullptr !!"
                               << " previousID.trackID()= " << previousID[k].trackID();
    return;
  }

  theHC[k]->insert(hit);
  if (useMap)
    hitMap[k].insert(std::pair<CaloHitID, CaloG4Hit*>(previousID[k], hit));
}

bool CaloSD::saveHit(CaloG4Hit* aHit, int k) {
  int tkID;
  bool ok = true;

  double time = aHit->getTimeSlice();
  if (corrTOFBeam)
    time += correctT;

  // More strict bookkeeping for finecalo
  if (doFineCalo_ && aHit->isFinecaloTrackID()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "Saving hit " << shortreprID(aHit);
#endif
    if (!m_trackManager)
      throw cms::Exception("Unknown", "CaloSD") << "m_trackManager not set, needed for finecalo!";
    if (!m_trackManager->trackExists(aHit->getTrackID()))
      throw cms::Exception("Unknown", "CaloSD")
          << "Error on hit " << shortreprID(aHit) << ": Parent track not in track manager";
    slave[k].get()->processHits(aHit->getUnitID(),
                                aHit->getEM() / CLHEP::GeV,
                                aHit->getHadr() / CLHEP::GeV,
                                time,
                                aHit->getTrackID(),
                                aHit->getDepth());
  }
  // Regular, not-fine way:
  else {
    if (m_trackManager) {
      tkID = m_trackManager->giveMotherNeeded(aHit->getTrackID());
      if (tkID == 0) {
        if (m_trackManager->trackExists(aHit->getTrackID()))
          tkID = (aHit->getTrackID());
        else {
          ok = false;
        }
      }
    } else {
      tkID = aHit->getTrackID();
      ok = false;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "Saving hit " << shortreprID(aHit) << " with trackID=" << tkID;
#endif

    slave[k].get()->processHits(
        aHit->getUnitID(), aHit->getEM() / CLHEP::GeV, aHit->getHadr() / CLHEP::GeV, time, tkID, aHit->getDepth());
  }

#ifdef EDM_ML_DEBUG
  if (!ok)
    edm::LogWarning("CaloSim") << "CaloSD:Cannot find track ID for " << aHit->getTrackID();
  edm::LogVerbatim("CaloSim") << "CalosD: Track ID " << aHit->getTrackID() << " changed to " << tkID
                              << " by SimTrackManager Status " << ok;
#endif

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: Store Hit at " << std::hex << aHit->getUnitID() << std::dec << " "
                              << aHit->getDepth() << " due to " << tkID << " in time " << time << " of energy "
                              << aHit->getEM() / CLHEP::GeV << " GeV (EM) and " << aHit->getHadr() / CLHEP::GeV
                              << " GeV (Hadr)";
#endif
  return ok;
}

void CaloSD::update(const BeginOfTrack* trk) {
  int primary = -1;
  TrackInformation* trkInfo = cmsTrackInformation((*trk)());
  if (trkInfo->isPrimary())
    primary = (*trk)()->GetTrackID();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "New track: isPrimary " << trkInfo->isPrimary() << " primary ID = " << primary
                              << " primary ancestor ID " << primAncestor;
#endif

  // update the information if a different primary track ID

  if (primary > 0 && primary != primAncestor) {
    primAncestor = primary;

    // clean the hits information

    bool clean(false);
    for (int k = 0; k < nHC_; ++k)
      if (theHC[k]->entries() > 0)
        clean = true;
    if (clean)
      cleanHitCollection();
  }
}

void CaloSD::cleanHitCollection() {
  for (int k = 0; k < nHC_; ++k) {
    std::vector<CaloG4Hit*>* theCollection = theHC[k]->GetVector();

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: collection before merging, size = " << theHC[k]->entries();
#endif
    if (reusehit[k].empty())
      reusehit[k].reserve(theHC[k]->entries() - cleanIndex[k]);

    // if no map used, merge before hits to have the save situation as a map
    if (!useMap) {
      std::vector<CaloG4Hit*> hitvec;

      hitvec.swap(*theCollection);
      sort((hitvec.begin() + cleanIndex[k]), hitvec.end(), CaloG4HitLess());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CaloSim") << "CaloSD::cleanHitCollection: sort hits in buffer starting from "
                                  << "element = " << cleanIndex[k];
      for (unsigned int i = 0; i < hitvec.size(); ++i) {
        if (hitvec[i] == nullptr)
          edm::LogVerbatim("CaloSim") << i << " has a null pointer";
        else
          edm::LogVerbatim("CaloSim") << i << " " << *hitvec[i];
      }
#endif
      CaloG4HitEqual equal;
      for (unsigned int i = cleanIndex[k]; i < hitvec.size(); ++i) {
        int jump = 0;
        for (unsigned int j = i + 1; j < hitvec.size() && equal(hitvec[i], hitvec[j]); ++j) {
          ++jump;
          // merge j to i
          (*hitvec[i]).addEnergyDeposit(*hitvec[j]);
          (*hitvec[j]).setEM(0.);
          (*hitvec[j]).setHadr(0.);
          reusehit[k].emplace_back(hitvec[j]);
          hitvec[j] = nullptr;
        }
        i += jump;
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CaloSim") << "CaloSD: cleanHitCollection merge the hits in buffer ";
      for (unsigned int i = 0; i < hitvec.size(); ++i) {
        if (hitvec[i] == nullptr)
          edm::LogVerbatim("CaloSim") << i << " has a null pointer";
        else
          edm::LogVerbatim("CaloSim") << i << " " << *hitvec[i];
      }
#endif
      //move all nullptr to end of list and then remove them
      hitvec.erase(std::stable_partition(
                       hitvec.begin() + cleanIndex[k], hitvec.end(), [](CaloG4Hit* p) { return p != nullptr; }),
                   hitvec.end());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CaloSim") << "CaloSD::cleanHitCollection: remove the merged hits in buffer, new size = "
                                  << hitvec.size();
#endif
      hitvec.swap(*theCollection);
      totalHits[k] = theHC[k]->entries();
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: collection after merging, size= " << theHC[k]->entries()
                                << " Size of reusehit= " << reusehit[k].size()
                                << "\n      starting hit selection from index = " << cleanIndex[k];
#endif

    int addhit = 0;
    for (unsigned int i = cleanIndex[k]; i < theCollection->size(); ++i) {
      CaloG4Hit* aHit((*theCollection)[i]);

      // selection

      double time = aHit->getTimeSlice();
      if (corrTOFBeam)
        time += correctT;
      if (!filterHit(aHit, time)) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("CaloSim") << "CaloSD: dropped CaloG4Hit "
                                    << " " << *aHit;
#endif

        // create the list of hits to be reused

        reusehit[k].emplace_back((*theCollection)[i]);
        (*theCollection)[i] = nullptr;
        ++addhit;
      }
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: Size of reusehit after selection = " << reusehit[k].size()
                                << " Number of added hit = " << addhit;
#endif
    if (useMap) {
      if (addhit > 0) {
        int offset = reusehit[k].size() - addhit;
        for (int ii = addhit - 1; ii >= 0; --ii) {
          CaloHitID theID = reusehit[k][offset + ii]->getID();
          hitMap[k].erase(theID);
        }
      }
    }

    //move all nullptr to end of list and then remove them
    theCollection->erase(
        std::stable_partition(
            theCollection->begin() + cleanIndex[k], theCollection->end(), [](CaloG4Hit* p) { return p != nullptr; }),
        theCollection->end());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: hit collection after selection, size = " << theHC[k]->entries();
    theHC[k]->PrintAllHits();
#endif

    cleanIndex[k] = theHC[k]->entries();
  }
}

void CaloSD::printDetectorLevels(const G4VTouchable* touch) const {
  //Print name and copy numbers
  int level = ((touch->GetHistoryDepth()) + 1);
  std::ostringstream st1;
  st1 << level << " Levels:";
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      std::string name = (pv != nullptr) ? pv->GetName() : "Unknown";
      st1 << " " << name << ":" << touch->GetReplicaNumber(i);
    }
  }
  edm::LogVerbatim("CaloSim") << st1.str();
}
