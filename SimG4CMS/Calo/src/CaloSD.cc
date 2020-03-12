///////////////////////////////////////////////////////////////////////////////
// File: CaloSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include "G4EventManager.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4GFlashSpot.hh"
#include "G4ParticleTable.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include <fstream>

//#define EDM_ML_DEBUG

CaloSD::CaloSD(const std::string& name,
               const edm::EventSetup& es,
               const SensitiveDetectorCatalog& clg,
               edm::ParameterSet const& p,
               const SimTrackManager* manager,
               float timeSliceUnit,
               bool ignoreTkID)
    : SensitiveCaloDetector(name, es, clg, p),
      G4VGFlashSensitiveDetector(),
      eminHit(0.),
      currentHit(nullptr),
      m_trackManager(manager),
      theHC(nullptr),
      ignoreTrackID(ignoreTkID),
      hcID(-1),
      timeSlice(timeSliceUnit),
      eminHitD(0.) {
  //Parameters
  edm::ParameterSet m_CaloSD = p.getParameter<edm::ParameterSet>("CaloSD");
  energyCut = m_CaloSD.getParameter<double>("EminTrack") * GeV;
  tmaxHit = m_CaloSD.getParameter<double>("TmaxHit") * ns;
  std::vector<double> eminHits = m_CaloSD.getParameter<std::vector<double>>("EminHits");
  std::vector<double> tmaxHits = m_CaloSD.getParameter<std::vector<double>>("TmaxHits");
  std::vector<std::string> hcn = m_CaloSD.getParameter<std::vector<std::string>>("HCNames");
  std::vector<int> useResMap = m_CaloSD.getParameter<std::vector<int>>("UseResponseTables");
  std::vector<double> eminHitX = m_CaloSD.getParameter<std::vector<double>>("EminHitsDepth");
  suppressHeavy = m_CaloSD.getParameter<bool>("SuppressHeavy");
  kmaxIon = m_CaloSD.getParameter<double>("IonThreshold") * MeV;
  kmaxProton = m_CaloSD.getParameter<double>("ProtonThreshold") * MeV;
  kmaxNeutron = m_CaloSD.getParameter<double>("NeutronThreshold") * MeV;
  nCheckedHits = m_CaloSD.getUntrackedParameter<int>("CheckHits", 25);
  useMap = m_CaloSD.getUntrackedParameter<bool>("UseMap", true);
  int verbn = m_CaloSD.getUntrackedParameter<int>("Verbosity", 0);
  corrTOFBeam = m_CaloSD.getParameter<bool>("CorrectTOFBeam");
  double beamZ = m_CaloSD.getParameter<double>("BeamPosition") * cm;
  correctT = beamZ / c_light / nanosecond;
  useFineCaloID_ = m_CaloSD.getParameter<bool>("UseFineCaloID");

  SetVerboseLevel(verbn);
  meanResponse.reset(nullptr);
  for (unsigned int k = 0; k < hcn.size(); ++k) {
    if (name == hcn[k]) {
      if (k < eminHits.size())
        eminHit = eminHits[k] * MeV;
      if (k < eminHitX.size())
        eminHitD = eminHitX[k] * MeV;
      if (k < tmaxHits.size())
        tmaxHit = tmaxHits[k] * ns;
      if (k < useResMap.size() && useResMap[k] > 0) {
        meanResponse.reset(new CaloMeanResponse(p));
        break;
      }
    }
  }
  slave.reset(new CaloSlaveSD(name));

  currentID = CaloHitID(timeSlice, ignoreTrackID);
  previousID = CaloHitID(timeSlice, ignoreTrackID);
  isParameterized = false;

  entrancePoint.set(0., 0., 0.);
  entranceLocal.set(0., 0., 0.);
  posGlobal.set(0., 0., 0.);
  incidentEnergy = edepositEM = edepositHAD = 0.f;

  primAncestor = cleanIndex = totalHits = primIDSaved = 0;
  forceSave = false;

  edm::LogVerbatim("CaloSim") << "CaloSD: Minimum energy of track for saving it " << energyCut / GeV << " GeV"
                              << "\n"
                              << "        Use of HitID Map " << useMap << "\n"
                              << "        Check last " << nCheckedHits << " before saving the hit\n"
                              << "        Correct TOF globally by " << correctT << " ns (Flag =" << corrTOFBeam << ")\n"
                              << "        Save hits recorded before " << tmaxHit << " ns and if energy is above "
                              << eminHit / MeV << " MeV (for depth 0) or " << eminHitD / MeV
                              << " MeV (for nonzero depths);\n"
                              << "        Time Slice Unit " << timeSlice << "\nIgnore TrackID Flag " << ignoreTrackID
                              << " UseFineCaloID flag " << useFineCaloID_;
}

CaloSD::~CaloSD() {}

G4bool CaloSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  NaNTrap(aStep);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::" << GetName() << " ID= " << aStep->GetTrack()->GetTrackID()
                              << " prID= " << aStep->GetTrack()->GetParentID()
                              << " Eprestep= " << aStep->GetPreStepPoint()->GetKineticEnergy()
                              << " step= " << aStep->GetStepLength() << " Edep= " << aStep->GetTotalEnergyDeposit();
#endif
  // apply shower library or parameterisation
  if (isParameterized) {
    if (getFromLibrary(aStep)) {
      // for parameterized showers the primary track should be killed
      aStep->GetTrack()->SetTrackStatus(fStopAndKill);
      auto tv = aStep->GetSecondary();
      auto vol = aStep->GetPreStepPoint()->GetPhysicalVolume();
      for (auto& tk : *tv) {
        if (tk->GetVolume() == vol) {
          tk->SetTrackStatus(fStopAndKill);
        }
      }
      return true;
    }
  }

  // ignore steps without energy deposit
  edepositEM = edepositHAD = 0.f;
  unsigned int unitID = setDetUnitId(aStep);
  auto const theTrack = aStep->GetTrack();
  uint16_t depth = getDepth(aStep);

  double time = theTrack->GetGlobalTime() / nanosecond;
  int primaryID = getTrackID(theTrack);
  if (unitID > 0) {
    currentID.setID(unitID, time, primaryID, depth);
  } else {
    if (aStep->GetTotalEnergyDeposit() > 0.0) {
      const G4TouchableHistory* touch = static_cast<const G4TouchableHistory*>(theTrack->GetTouchable());
      edm::LogVerbatim("CaloSim") << "CaloSD::ProcessHits: unitID= " << unitID << " currUnit=   " << currentID.unitID()
                                  << " Detector: " << GetName() << " trackID= " << theTrack->GetTrackID() << " "
                                  << theTrack->GetDefinition()->GetParticleName()
                                  << "\n Edep= " << aStep->GetTotalEnergyDeposit()
                                  << " PV: " << touch->GetVolume(0)->GetName()
                                  << " PVid= " << touch->GetReplicaNumber(0) << " MVid= " << touch->GetReplicaNumber(1);
    }
    return false;
  }

  if (aStep->GetTotalEnergyDeposit() == 0.0) {
    return false;
  }

  double energy = getEnergyDeposit(aStep);
  if (energy > 0.0) {
    if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
      edepositEM = energy;
    } else {
      edepositHAD = energy;
    }
#ifdef EDM_ML_DEBUG
    G4TouchableHistory* touch = (G4TouchableHistory*)(theTrack->GetTouchable());
    edm::LogVerbatim("CaloSim") << "CaloSD::" << GetName() << " PV:" << touch->GetVolume(0)->GetName()
                                << " PVid=" << touch->GetReplicaNumber(0) << " MVid=" << touch->GetReplicaNumber(1)
                                << " Unit:" << std::hex << unitID << std::dec << " Edep=" << edepositEM << " "
                                << edepositHAD << " ID=" << theTrack->GetTrackID() << " pID=" << theTrack->GetParentID()
                                << " E=" << theTrack->GetKineticEnergy() << " S=" << aStep->GetStepLength() << "\n "
                                << theTrack->GetDefinition()->GetParticleName() << " primaryID= " << primaryID
                                << " currentID= (" << currentID << ") previousID= (" << previousID << ")";
#endif
    if (!hitExists(aStep)) {
      currentHit = createNewHit(aStep, aStep->GetTrack());
    }
    return true;
  }
  return false;
}

bool CaloSD::ProcessHits(G4GFlashSpot* aSpot, G4TouchableHistory*) {
  edepositEM = edepositHAD = 0.f;
  const G4Track* track = aSpot->GetOriginatorTrack()->GetPrimaryTrack();
  if (!G4TrackToParticleID::isGammaElectronPositron(track)) {
    return false;
  }
  double edep = aSpot->GetEnergySpot()->GetEnergy();
  if (edep <= 0.0) {
    return false;
  }
  edepositEM = edep;
  G4Step fFakeStep;
  G4StepPoint* fFakePreStepPoint = fFakeStep.GetPreStepPoint();
  G4StepPoint* fFakePostStepPoint = fFakeStep.GetPostStepPoint();
  fFakePreStepPoint->SetPosition(aSpot->GetPosition());
  fFakePostStepPoint->SetPosition(aSpot->GetPosition());

  G4TouchableHandle fTouchableHandle = aSpot->GetTouchableHandle();
  fFakePreStepPoint->SetTouchableHandle(fTouchableHandle);
  fFakeStep.SetTotalEnergyDeposit(edep);

  unsigned int unitID = setDetUnitId(&fFakeStep);

  if (unitID > 0) {
    double time = 0;
    int primaryID = getTrackID(track);
    uint16_t depth = getDepth(&fFakeStep);
    currentID.setID(unitID, time, primaryID, depth);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD:: GetSpotInfo for Unit 0x" << std::hex << currentID.unitID() << std::dec
                                << " Edeposit = " << edepositEM << " " << edepositHAD;
#endif
    // Update if in the same detector, time-slice and for same track
    if (currentID == previousID) {
      updateHit(currentHit);
    } else {
      posGlobal = aSpot->GetEnergySpot()->GetPosition();
      // Reset entry point for new primary
      if (currentID.trackID() != previousID.trackID()) {
        entrancePoint = aSpot->GetPosition();
        entranceLocal = aSpot->GetTouchableHandle()->GetHistory()->GetTopTransform().TransformPoint(entrancePoint);
        incidentEnergy = track->GetKineticEnergy();
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("CaloSim") << "CaloSD: Incident energy " << incidentEnergy / GeV << " GeV and"
                                    << " entrance point " << entrancePoint << " (Global) " << entranceLocal
                                    << " (Local)";
#endif
      }
      if (!checkHit()) {
        currentHit = createNewHit(&fFakeStep, track);
      }
    }
    return true;
  }
  return false;
}

double CaloSD::getEnergyDeposit(const G4Step* aStep) { return aStep->GetTotalEnergyDeposit(); }

bool CaloSD::getFromLibrary(const G4Step*) { return false; }

void CaloSD::Initialize(G4HCofThisEvent* HCE) {
  totalHits = 0;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD : Initialize called for " << GetName();
#endif

  //This initialization is performed at the beginning of an event
  //------------------------------------------------------------
  theHC = new CaloG4HitCollection(GetName(), collectionName[0]);

  if (hcID < 0) {
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  }
  //theHC ownership is transfered here to HCE
  HCE->AddHitsCollection(hcID, theHC);
}

void CaloSD::EndOfEvent(G4HCofThisEvent*) {
  // clean the hits for the last tracks

  cleanHitCollection();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: EndofEvent entered with " << theHC->entries() << " entries";
#endif
}

void CaloSD::clear() {}

void CaloSD::DrawAll() {}

void CaloSD::PrintAll() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: Collection " << theHC->GetName();
#endif
  theHC->PrintAllHits();
}

void CaloSD::fillHits(edm::PCaloHitContainer& cc, const std::string& hname) {
  if (slave.get()->name() == hname) {
    cc = slave.get()->hits();
  }
  slave.get()->Clean();
}

G4ThreeVector CaloSD::setToLocal(const G4ThreeVector& global, const G4VTouchable* touch) const {
  return touch->GetHistory()->GetTopTransform().TransformPoint(global);
}

G4ThreeVector CaloSD::setToGlobal(const G4ThreeVector& local, const G4VTouchable* touch) const {
  return touch->GetHistory()->GetTopTransform().Inverse().TransformPoint(local);
}

bool CaloSD::hitExists(const G4Step* aStep) {
  // Update if in the same detector, time-slice and for same track
  if (currentID == previousID) {
    updateHit(currentHit);
    return true;
  }

  // Reset entry point for new primary
  posGlobal = aStep->GetPreStepPoint()->GetPosition();
  if (currentID.trackID() != previousID.trackID()) {
    resetForNewPrimary(aStep);
  }
  return checkHit();
}

bool CaloSD::checkHit() {
  //look in the HitContainer whether a hit with the same ID already exists:
  bool found = false;
  if (useMap) {
    std::map<CaloHitID, CaloG4Hit*>::const_iterator it = hitMap.find(currentID);
    if (it != hitMap.end()) {
      currentHit = it->second;
      found = true;
    }
  } else if (nCheckedHits > 0) {
    int nhits = theHC->entries();
    int minhit = std::max(nhits - nCheckedHits, 0);
    int maxhit = nhits - 1;

    for (int j = maxhit; j > minhit; --j) {
      if ((*theHC)[j]->getID() == currentID) {
        currentHit = (*theHC)[j];
        found = true;
        break;
      }
    }
  }

  if (found) {
    updateHit(currentHit);
  }
  return found;
}

int CaloSD::getNumberOfHits() { return theHC->entries(); }

CaloG4Hit* CaloSD::createNewHit(const G4Step* aStep, const G4Track* theTrack) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::CreateNewHit " << getNumberOfHits() << " for " << GetName()
                              << " Unit:" << currentID.unitID() << " " << currentID.depth() << " Edep= " << edepositEM
                              << " " << edepositHAD << " primaryID= " << currentID.trackID()
                              << " timeSlice= " << currentID.timeSliceID() << " ID= " << theTrack->GetTrackID() << " "
                              << theTrack->GetDefinition()->GetParticleName()
                              << " E(GeV)= " << theTrack->GetKineticEnergy() / GeV
                              << " parentID= " << theTrack->GetParentID() << "\n Ein= " << incidentEnergy
                              << " entranceGlobal: " << entrancePoint << " entranceLocal: " << entranceLocal
                              << " posGlobal: " << posGlobal;
#endif

  CaloG4Hit* aHit;
  if (!reusehit.empty()) {
    aHit = reusehit.back().release();
    aHit->setEM(0.f);
    aHit->setHadr(0.f);
    reusehit.pop_back();
  } else {
    aHit = new CaloG4Hit;
  }

  aHit->setID(currentID);
  aHit->setEntry(entrancePoint.x(), entrancePoint.y(), entrancePoint.z());
  aHit->setEntryLocal(entranceLocal.x(), entranceLocal.y(), entranceLocal.z());
  aHit->setPosition(posGlobal.x(), posGlobal.y(), posGlobal.z());
  aHit->setIncidentEnergy(incidentEnergy);
  updateHit(aHit);

  storeHit(aHit);
  double etrack = 0;
  if (currentID.trackID() == primIDSaved) {  // The track is saved; nothing to be done
  } else if (currentID.trackID() == theTrack->GetTrackID()) {
    etrack = theTrack->GetKineticEnergy();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: set save the track " << currentID.trackID() << " etrack " << etrack
                                << " eCut " << energyCut << " force: " << forceSave
                                << " save: " << (etrack >= energyCut || forceSave);
#endif
    if (etrack >= energyCut || forceSave) {
      TrackInformation* trkInfo = cmsTrackInformation(theTrack);
      trkInfo->storeTrack(true);
      trkInfo->putInHistory();
    }
  } else {
    TrackWithHistory* trkh = tkMap[currentID.trackID()];
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD : TrackwithHistory pointer for " << currentID.trackID() << " is " << trkh;
#endif
    if (trkh != nullptr) {
      etrack = sqrt(trkh->momentum().Mag2());
      if (etrack >= energyCut) {
        trkh->save();
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("CaloSim") << "CaloSD: set save the track " << currentID.trackID() << " with Hit";
#endif
      }
    }
  }
  primIDSaved = currentID.trackID();
  if (useMap)
    ++totalHits;
  return aHit;
}

void CaloSD::updateHit(CaloG4Hit* aHit) {
  aHit->addEnergyDeposit(edepositEM, edepositHAD);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD:" << GetName() << " Add energy deposit in " << currentID
                              << " Edep_em(MeV)= " << edepositEM << " Edep_had(MeV)= " << edepositHAD;
#endif

  // buffer for next steps:
  previousID = currentID;
}

void CaloSD::resetForNewPrimary(const G4Step* aStep) {
  auto const preStepPoint = aStep->GetPreStepPoint();
  entrancePoint = preStepPoint->GetPosition();
  entranceLocal = setToLocal(entrancePoint, preStepPoint->GetTouchable());
  incidentEnergy = preStepPoint->GetKineticEnergy();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::resetForNewPrimary for " << GetName()
                              << " ID= " << aStep->GetTrack()->GetTrackID() << " Ein= " << incidentEnergy / GeV
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
    const TrackContainer* trksForThisEvent = m_trackManager->trackContainer();
    if (trksForThisEvent != nullptr) {
      int it = (int)(trksForThisEvent->size()) - 1;
      if (it >= 0) {
        TrackWithHistory* trkH = (*trksForThisEvent)[it];
        if (trkH->trackID() == (unsigned int)(id))
          tkMap[id] = trkH;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("CaloSim") << "CaloSD: get track " << it << " from Container of size "
                                    << trksForThisEvent->size() << " with ID " << trkH->trackID();
      } else {
        edm::LogVerbatim("CaloSim") << "CaloSD: get track " << it << " from Container of size "
                                    << trksForThisEvent->size() << " with no ID";
#endif
      }
    }
  }
}

void CaloSD::update(const ::EndOfEvent*) {
  endEvent();
  slave.get()->ReserveMemory(theHC->entries());

  int count(0);
  int wrong(0);
  double eEM(0.0);
  double eHAD(0.0);
  double eEM2(0.0);
  double eHAD2(0.0);
  double tt(0.0);
  double zloc(0.0);
  double zglob(0.0);
  double ee(0.0);
  int hc_entries = theHC->entries();
  for (int i = 0; i < hc_entries; ++i) {
    if (!saveHit((*theHC)[i])) {
      ++wrong;
    }
    ++count;
    double x = (*theHC)[i]->getEM();
    eEM += x;
    eEM2 += x * x;
    x = (*theHC)[i]->getHadr();
    eHAD += x;
    eHAD2 += x * x;
    tt += (*theHC)[i]->getTimeSlice();
    ee += (*theHC)[i]->getIncidentEnergy();
    zglob += std::abs((*theHC)[i]->getEntry().z());
    zloc += std::abs((*theHC)[i]->getEntryLocal().z());
  }

  double norm = (count > 0) ? 1.0 / count : 0.0;
  eEM *= norm;
  eEM2 *= norm;
  eHAD *= norm;
  eHAD2 *= norm;
  eEM2 = std::sqrt(eEM2 - eEM * eEM);
  eHAD2 = std::sqrt(eHAD2 - eHAD * eHAD);
  tt *= norm;
  ee *= norm;
  zglob *= norm;
  zloc *= norm;

  edm::LogVerbatim("CaloSim") << "CaloSD: " << GetName() << " store " << count << " hits; " << wrong
                              << " track IDs not given properly and " << totalHits - count
                              << " hits not passing cuts\n EmeanEM= " << eEM << " ErmsEM= " << eEM2
                              << "\n EmeanHAD= " << eHAD << " ErmsHAD= " << eHAD2 << " TimeMean= " << tt
                              << " E0mean= " << ee << " Zglob= " << zglob << " Zloc= " << zloc << " ";

  tkMap.erase(tkMap.begin(), tkMap.end());
  std::vector<std::unique_ptr<CaloG4Hit>>().swap(reusehit);
  if (useMap)
    hitMap.erase(hitMap.begin(), hitMap.end());
}

void CaloSD::clearHits() {
  cleanIndex = 0;
  previousID.reset();
  primIDSaved = -99;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: Clears hit vector for " << GetName()
                              << " and initialise slave: " << slave.get()->name();
#endif
  slave.get()->Initialize();
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
  forceSave = false;
  TrackInformation* trkInfo = cmsTrackInformation(aTrack);
  if (trkInfo) {
    primaryID = (useFineCaloID_) ? trkInfo->getIDfineCalo() : trkInfo->getIDonCaloSurface();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "Track ID: " << trkInfo->getIDfineCalo() << ":" << trkInfo->getIDonCaloSurface()
                                << ":" << aTrack->GetTrackID() << ":" << primaryID;
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
  int primaryID = (useFineCaloID_) ? trkInfo->getIDfineCalo() : trkInfo->getIDonCaloSurface();
  if (primaryID <= 0) {
    primaryID = theTrack->GetTrackID();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "Track ID: " << trkInfo->getIDfineCalo() << ":" << trkInfo->getIDonCaloSurface() << ":"
                              << theTrack->GetTrackID() << ":" << primaryID;
#endif

  if (primaryID != previousID.trackID()) {
    resetForNewPrimary(aStep);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD::setTrackID for " << GetName()
                              << " trackID= " << aStep->GetTrack()->GetTrackID() << " primaryID= " << primaryID;
#endif
  return primaryID;
}

uint16_t CaloSD::getDepth(const G4Step*) { return 0; }

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

double CaloSD::getResponseWt(const G4Track* aTrack) {
  double wt = 1.0;
  if (meanResponse.get()) {
    TrackInformation* trkInfo = cmsTrackInformation(aTrack);
    wt = meanResponse.get()->getWeight(trkInfo->genParticlePID(), trkInfo->genParticleP());
  }
  return wt;
}

void CaloSD::storeHit(CaloG4Hit* hit) {
  if (hit == nullptr || previousID.trackID() < 0) {
    edm::LogWarning("CaloSim") << "CaloSD: hit to be stored is nullptr !!"
                               << " previousID.trackID()= " << previousID.trackID();
    return;
  }

  theHC->insert(hit);
  if (useMap)
    hitMap.insert(std::pair<CaloHitID, CaloG4Hit*>(previousID, hit));
}

bool CaloSD::saveHit(CaloG4Hit* aHit) {
  int tkID;
  bool ok = true;
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
  edm::LogVerbatim("CaloSim") << "CalosD: Track ID " << aHit->getTrackID() << " changed to " << tkID
                              << " by SimTrackManager Status " << ok;
#endif
  double time = aHit->getTimeSlice();
  if (corrTOFBeam)
    time += correctT;
  slave.get()->processHits(aHit->getUnitID(), aHit->getEM() / GeV, aHit->getHadr() / GeV, time, tkID, aHit->getDepth());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: Store Hit at " << std::hex << aHit->getUnitID() << std::dec << " "
                              << aHit->getDepth() << " due to " << tkID << " in time " << time << " of energy "
                              << aHit->getEM() / GeV << " GeV (EM) and " << aHit->getHadr() / GeV << " GeV (Hadr)";
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

    if (theHC->entries() > 0)
      cleanHitCollection();
  }
}

void CaloSD::cleanHitCollection() {
  std::vector<CaloG4Hit*>* theCollection = theHC->GetVector();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: collection before merging, size = " << theHC->entries();
#endif
  if (reusehit.empty())
    reusehit.reserve(theHC->entries() - cleanIndex);

  // if no map used, merge before hits to have the save situation as a map
  if (!useMap) {
    std::vector<CaloG4Hit*> hitvec;

    hitvec.swap(*theCollection);
    sort((hitvec.begin() + cleanIndex), hitvec.end(), CaloG4HitLess());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD::cleanHitCollection: sort hits in buffer starting from "
                                << "element = " << cleanIndex;
    for (unsigned int i = 0; i < hitvec.size(); ++i)
      edm::LogVerbatim("CaloSim") << i << " " << *hitvec[i];
#endif
    CaloG4HitEqual equal;
    for (unsigned int i = cleanIndex; i < hitvec.size(); ++i) {
      int jump = 0;
      for (unsigned int j = i + 1; j < hitvec.size() && equal(hitvec[i], hitvec[j]); ++j) {
        ++jump;
        // merge j to i
        (*hitvec[i]).addEnergyDeposit(*hitvec[j]);
        (*hitvec[j]).setEM(0.);
        (*hitvec[j]).setHadr(0.);
        reusehit.emplace_back(hitvec[j]);
        hitvec[j] = nullptr;
      }
      i += jump;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD: cleanHitCollection merge the hits in buffer ";
    for (unsigned int i = 0; i < hitvec.size(); ++i)
      edm::LogVerbatim("CaloSim") << i << " " << *hitvec[i];
#endif
    //move all nullptr to end of list and then remove them
    hitvec.erase(
        std::stable_partition(hitvec.begin() + cleanIndex, hitvec.end(), [](CaloG4Hit* p) { return p != nullptr; }),
        hitvec.end());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloSD::cleanHitCollection: remove the merged hits in buffer,"
                                << " new size = " << hitvec.size();
    for (unsigned int i = 0; i < hitvec.size(); ++i)
      edm::LogVerbatim("CaloSim") << i << " " << *hitvec[i];
#endif
    hitvec.swap(*theCollection);
    totalHits = theHC->entries();
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: collection after merging, size= " << theHC->entries()
                              << " Size of reusehit= " << reusehit.size()
                              << "\n      starting hit selection from index = " << cleanIndex;
#endif

  int addhit = 0;
  for (unsigned int i = cleanIndex; i < theCollection->size(); ++i) {
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

      reusehit.emplace_back((*theCollection)[i]);
      (*theCollection)[i] = nullptr;
      ++addhit;
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: Size of reusehit after selection = " << reusehit.size()
                              << " Number of added hit = " << addhit;
#endif
  if (useMap) {
    if (addhit > 0) {
      int offset = reusehit.size() - addhit;
      for (int ii = addhit - 1; ii >= 0; --ii) {
        CaloHitID theID = reusehit[offset + ii]->getID();
        hitMap.erase(theID);
      }
    }
  }

  //move all nullptr to end of list and then remove them
  theCollection->erase(
      std::stable_partition(
          theCollection->begin() + cleanIndex, theCollection->end(), [](CaloG4Hit* p) { return p != nullptr; }),
      theCollection->end());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSim") << "CaloSD: hit collection after selection, size = " << theHC->entries();
  theHC->PrintAllHits();
#endif

  cleanIndex = theHC->entries();
}
