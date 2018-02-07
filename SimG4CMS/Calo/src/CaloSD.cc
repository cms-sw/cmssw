///////////////////////////////////////////////////////////////////////////////
// File: CaloSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Application/interface/EventAction.h"

#include "G4EventManager.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4GFlashSpot.hh"
#include "G4ParticleTable.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

//#define DebugLog

CaloSD::CaloSD(const std::string& name, const DDCompactView & cpv,
        const SensitiveDetectorCatalog & clg,
        edm::ParameterSet const & p, const SimTrackManager* manager,
        float timeSliceUnit, bool ignoreTkID) : 
  SensitiveCaloDetector(name, cpv, clg, p),
  G4VGFlashSensitiveDetector(), eminHit(0.),currentHit(nullptr), 
  m_trackManager(manager), theHC(nullptr), ignoreTrackID(ignoreTkID), hcID(-1), 
  timeSlice(timeSliceUnit), eminHitD(0.) {

  //Parameters
  edm::ParameterSet m_CaloSD = p.getParameter<edm::ParameterSet>("CaloSD");
  energyCut    = m_CaloSD.getParameter<double>("EminTrack")*GeV;
  tmaxHit      = m_CaloSD.getParameter<double>("TmaxHit")*ns;
  std::vector<double> eminHits = m_CaloSD.getParameter<std::vector<double> >("EminHits");
  std::vector<double> tmaxHits = m_CaloSD.getParameter<std::vector<double> >("TmaxHits");
  std::vector<std::string> hcn = m_CaloSD.getParameter<std::vector<std::string> >("HCNames");
  std::vector<int>   useResMap = m_CaloSD.getParameter<std::vector<int> >("UseResponseTables");
  std::vector<double> eminHitX = m_CaloSD.getParameter<std::vector<double> >("EminHitsDepth");
  suppressHeavy= m_CaloSD.getParameter<bool>("SuppressHeavy");
  kmaxIon      = m_CaloSD.getParameter<double>("IonThreshold")*MeV;
  kmaxProton   = m_CaloSD.getParameter<double>("ProtonThreshold")*MeV;
  kmaxNeutron  = m_CaloSD.getParameter<double>("NeutronThreshold")*MeV;
  checkHits    = m_CaloSD.getUntrackedParameter<int>("CheckHits", 25);
  useMap       = m_CaloSD.getUntrackedParameter<bool>("UseMap", true);
  int verbn    = m_CaloSD.getUntrackedParameter<int>("Verbosity", 0);
  corrTOFBeam  = m_CaloSD.getParameter<bool>("CorrectTOFBeam");
  double beamZ = m_CaloSD.getParameter<double>("BeamPosition")*cm;
  correctT     = beamZ/c_light/nanosecond;

  SetVerboseLevel(verbn);
  meanResponse.reset(nullptr);
  for (unsigned int k=0; k<hcn.size(); ++k) {
    if (name == hcn[k]) {
      if (k < eminHits.size()) eminHit = eminHits[k]*MeV;
      if (k < eminHitX.size()) eminHitD= eminHitX[k]*MeV;
      if (k < tmaxHits.size()) tmaxHit = tmaxHits[k]*ns;
      if (k < useResMap.size() && useResMap[k] > 0) {
        meanResponse.reset(new CaloMeanResponse(p));
        break;
      }
    }
  }
  slave.reset(new CaloSlaveSD(name));

  currentID  = CaloHitID(timeSlice, ignoreTrackID);
  previousID = CaloHitID(timeSlice, ignoreTrackID);
  isParameterized = false;
  
  primAncestor = cleanIndex = totalHits = primIDSaved = 0;
  forceSave = false;
  
  edm::LogInfo("CaloSim") << "CaloSD: Minimum energy of track for saving it " 
                          << energyCut/GeV  << " GeV" << "\n"
                          << "        Use of HitID Map " << useMap << "\n"
                          << "        Check last " << checkHits 
                          << " before saving the hit\n" 
                          << "        Correct TOF globally by " << correctT
                          << " ns (Flag =" << corrTOFBeam << ")\n"
                          << "        Save hits recorded before " << tmaxHit
                          << " ns and if energy is above " << eminHit/MeV
                          << " MeV (for depth 0) or " << eminHitD/MeV
                          << " MeV (for nonzero depths);\n"
                          << "        Time Slice Unit " 
                          << timeSlice << " Ignore TrackID Flag " << ignoreTrackID;
}

CaloSD::~CaloSD()
{
  delete theHC;
}

G4bool CaloSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {
  
  NaNTrap( aStep );

#ifdef DebugLog
  if(aStep->GetTrack()->GetTrackID() == 82946)
  edm::LogInfo("CaloSim") << "CaloSD::" << GetName()
                          << " ID= " << aStep->GetTrack()->GetTrackID() 
                          << " prID= " << aStep->GetTrack()->GetParentID() 
                          << " Eprestep= " << aStep->GetPreStepPoint()->GetKineticEnergy()
                          << " step= " << aStep->GetStepLength() << " Edep= " << aStep->GetTotalEnergyDeposit(); 
#endif
  // apply shower library or parameterisation
  if(isParameterized) { 
    if(getFromLibrary(aStep)) {

      // for parameterized showers the primary track should be killed
      aStep->GetTrack()->SetTrackStatus(fStopAndKill); 
      auto tv  = aStep->GetSecondary();
      auto vol = aStep->GetPreStepPoint()->GetPhysicalVolume();
      for(auto & tk : *tv) {
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
    
  double   time  = theTrack->GetGlobalTime()/nanosecond;
  int      primaryID = getTrackID(theTrack);
  currentID.setID(unitID, time, primaryID, depth);

  if(aStep->GetTotalEnergyDeposit() == 0.0) { 
    //--- This code is for backward compatibility and should be removed 
    hitExists(aStep);
    //--- 
    return false; 
  }

  double energy = getEnergyDeposit(aStep);
  if(energy > 0.0) {
    if(G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
      edepositEM  = energy;
    } else {
      edepositHAD = energy;
    }
#ifdef DebugLog
    G4TouchableHistory* touch =(G4TouchableHistory*)(theTrack->GetTouchable());
    if(theTrack->GetTrackID() == 82946) 
    edm::LogInfo("CaloSim") << "CaloSD::" << GetName()
                            << " PV:"   << touch->GetVolume(0)->GetName()
                            << " PVid=" << touch->GetReplicaNumber(0)
                            << " MVid=" << touch->GetReplicaNumber(1)
                            << " Unit:" << std::hex << unitID << std::dec 
                            << " Edep=" << edepositEM << " " << edepositHAD
                            << " ID=" << theTrack->GetTrackID()
                            << " pID=" << theTrack->GetParentID()
                            << " E=" << theTrack->GetKineticEnergy()
                            << " S=" << aStep->GetStepLength()
                            << " " << theTrack->GetDefinition()->GetParticleName()
                            << " currentID= " << currentID 
                            << " previousID= " << previousID;
#endif
    if(!hitExists(aStep)) {
      currentHit = createNewHit(aStep);
    }
    return true;
  }
  return false;
} 

bool CaloSD::ProcessHits(G4GFlashSpot* aSpot, G4TouchableHistory * ) { 

  edepositEM = edepositHAD = 0.f;
  const G4Track* track = aSpot->GetOriginatorTrack()->GetPrimaryTrack();
  if(!G4TrackToParticleID::isGammaElectronPositron(track)) { return false; }
  double edep = aSpot->GetEnergySpot()->GetEnergy();
  if (edep <= 0.0) { return false; }
  edepositEM = edep;
  G4Step fFakeStep;
  G4StepPoint * fFakePreStepPoint  = fFakeStep.GetPreStepPoint();
  G4StepPoint * fFakePostStepPoint = fFakeStep.GetPostStepPoint();
  fFakePreStepPoint->SetPosition(aSpot->GetPosition());
  fFakePostStepPoint->SetPosition(aSpot->GetPosition());
      
  G4TouchableHandle fTouchableHandle = aSpot->GetTouchableHandle();
  fFakePreStepPoint->SetTouchableHandle(fTouchableHandle);
  fFakeStep.SetTotalEnergyDeposit(edep);
      
  unsigned int unitID = setDetUnitId(&fFakeStep);

  if (unitID > 0) {
    double time   = 0;
    int primaryID = getTrackID(track);
    uint16_t depth = getDepth(&fFakeStep);
    currentID.setID(unitID, time, primaryID, depth);
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD:: GetSpotInfo for"
                            << " Unit 0x" << std::hex << currentID.unitID() 
                            << std::dec << " Edeposit = " << edepositEM << " " 
                            << edepositHAD;
#endif
    // Update if in the same detector, time-slice and for same track   
    if (currentID == previousID) {
      updateHit(currentHit);
    } else {
      posGlobal = aSpot->GetEnergySpot()->GetPosition();
      // Reset entry point for new primary
      if (currentID.trackID() != previousID.trackID()) {
        entrancePoint = aSpot->GetPosition();
        entranceLocal = aSpot->GetTouchableHandle()->GetHistory()->
          GetTopTransform().TransformPoint(entrancePoint);
        incidentEnergy = track->GetKineticEnergy();
#ifdef DebugLog
        LogDebug("CaloSim") << "CaloSD: Incident energy " 
                            << incidentEnergy/GeV << " GeV and" 
                            << " entrance point " << entrancePoint 
                            << " (Global) " << entranceLocal << " (Local)";
#endif
      }
      if (!checkHit()) { currentHit = createNewHit(&fFakeStep); }
    }
    return true;
  }
  return false;
}                                   

double CaloSD::getEnergyDeposit(const G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

bool CaloSD::getFromLibrary(const G4Step*) {
  return false;
}

void CaloSD::Initialize(G4HCofThisEvent * HCE) { 
  totalHits = 0;
  
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD : Initialize called for " << GetName(); 
#endif
  
  //This initialization is performed at the beginning of an event
  //------------------------------------------------------------
  theHC = new CaloG4HitCollection(GetName(), collectionName[0]);
  
  if (hcID<0) { 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]); 
  }
  HCE->AddHitsCollection(hcID, theHC);
}

void CaloSD::EndOfEvent(G4HCofThisEvent* ) {
  // clean the hits for the last tracks
  
  cleanHitCollection();
  
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: EndofEvent entered with " << theHC->entries()
                          << " entries";
#endif
}

void CaloSD::clear() {} 

void CaloSD::DrawAll() {} 

void CaloSD::PrintAll() {
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Collection " << theHC->GetName();
#endif
  theHC->PrintAllHits();
} 

void CaloSD::fillHits(edm::PCaloHitContainer& cc, const std::string& hname) {
  if (slave.get()->name() == hname) { cc=slave.get()->hits(); }
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
  if (currentID.trackID() != previousID.trackID()) { 
    resetForNewPrimary(aStep);
  }
  return checkHit();
}

bool CaloSD::checkHit() {  
  //look in the HitContainer whether a hit with the same ID already exists:
  bool       found = false;
  if (useMap) {
    std::map<CaloHitID,CaloG4Hit*>::const_iterator it = hitMap.find(currentID);
    if (it != hitMap.end()) {
      currentHit = it->second;
      found      = true;
    }
  } else {
    if (checkHits <= 0) return false;
    int  minhit= (theHC->entries()>checkHits ? theHC->entries()-checkHits : 0);
    int  maxhit= theHC->entries()-1;
    
    for (int j=maxhit; j>minhit; --j) {
      if ((*theHC)[j]->getID() == currentID) {
        currentHit = (*theHC)[j];
        found      = true;
        break;
      }
    }          
  }
  
  if (found) {
    updateHit(currentHit);
    return true;
  } else {
    return false;
  }
}

int CaloSD::getNumberOfHits() { return theHC->entries(); }

CaloG4Hit* CaloSD::createNewHit(const G4Step* aStep) {

  auto const theTrack = aStep->GetTrack();
#ifdef DebugLog
  if(GetName() == "CastorFI")
  edm::LogInfo("CaloSim") << "CaloSD::CreateNewHit " << getNumberOfHits()
                          << " for " << GetName()
                          << " Unit:" << currentID.unitID() 
                          << " " << currentID.depth()
                          << " Edep= " << edepositEM << " " << edepositHAD
                          << " primaryID= "    << currentID.trackID()
                          << " timeSlice= " << currentID.timeSliceID()
                          << " ID= " << theTrack->GetTrackID()
                          << " " <<theTrack->GetDefinition()->GetParticleName()
                          << " E(GeV)= "  << theTrack->GetKineticEnergy()/GeV
                          << " parentID= " << theTrack->GetParentID();
#endif  
  
  CaloG4Hit* aHit;
  if (!reusehit.empty()) {
    aHit = reusehit[0];
    aHit->setEM(0.f);
    aHit->setHadr(0.f);
    reusehit.erase(reusehit.begin());
  } else {
    aHit = new CaloG4Hit;
  }
  
  aHit->setID(currentID);
  aHit->setEntry(entrancePoint.x(),entrancePoint.y(),entrancePoint.z());
  aHit->setEntryLocal(entranceLocal.x(),entranceLocal.y(),entranceLocal.z());
  aHit->setPosition(posGlobal.x(),posGlobal.y(),posGlobal.z());
  aHit->setIncidentEnergy(incidentEnergy);
  updateHit(aHit);
  
  storeHit(aHit);
  double etrack = 0;
  if (currentID.trackID() == primIDSaved) { // The track is saved; nothing to be done
  } else if (currentID.trackID() == theTrack->GetTrackID()) {
    etrack= theTrack->GetKineticEnergy();
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD: set save the track " << currentID.trackID()
                            << " etrack " << etrack << " eCut " << energyCut 
                            << " force: " << forceSave 
                            << " save: " << (etrack >= energyCut || forceSave);
#endif
    if (etrack >= energyCut || forceSave) {
      TrackInformation* trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
      trkInfo->storeTrack(true);
      trkInfo->putInHistory();
    }
  } else {
    TrackWithHistory * trkh = tkMap[currentID.trackID()];
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD : TrackwithHistory pointer for " 
                            << currentID.trackID() << " is " << trkh;
#endif
    if (trkh != nullptr) {
      etrack = sqrt(trkh->momentum().Mag2());
      if (etrack >= energyCut) {
        trkh->save();
#ifdef DebugLog
        edm::LogInfo("CaloSim") << "CaloSD: set save the track " 
                                << currentID.trackID() << " with Hit";
#endif
      }
    }
  }
  primIDSaved = currentID.trackID();
  if (useMap) ++totalHits;
  return aHit;
}  

void CaloSD::updateHit(CaloG4Hit* aHit) {

  aHit->addEnergyDeposit(edepositEM,edepositHAD);
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD:" << GetName() << " Add energy deposit in " 
                          << currentID << " Edep_em(MeV)= " 
                          << edepositEM << " Edep_had(MeV)= " << edepositHAD; 
#endif

  // buffer for next steps:
  previousID = currentID;
}

void CaloSD::resetForNewPrimary(const G4Step* aStep) { 
  auto const preStepPoint = aStep->GetPreStepPoint();
  entrancePoint  = preStepPoint->GetPosition();
  entranceLocal  = setToLocal(entrancePoint, preStepPoint->GetTouchable());
  incidentEnergy = preStepPoint->GetKineticEnergy();
#ifdef DebugLog
  if(aStep->GetTrack()->GetTrackID() == 82946)
  edm::LogInfo("CaloSim") << "CaloSD::resetForNewPrimary Ein= " << incidentEnergy/GeV 
                          << " GeV and" << " entrance point global: " << entrancePoint 
                          << " local: " << entranceLocal;
#endif
}

double CaloSD::getAttenuation(const G4Step* aStep, double birk1, double birk2, double birk3) const {
  double weight = 1.;
  double charge = aStep->GetPreStepPoint()->GetCharge();
  double length = aStep->GetStepLength();

  if (charge != 0. && length > 0.) {
    double density = aStep->GetPreStepPoint()->GetMaterial()->GetDensity();
    double dedx    = aStep->GetTotalEnergyDeposit()/length;
    double rkb     = birk1/density;
    double c       = birk2*rkb*rkb;
    if (std::abs(charge) >= 2.) rkb /= birk3; // based on alpha particle data
    weight = 1./(1.+rkb*dedx+c*dedx*dedx);
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD::getAttenuation in " << mat->GetName() 
                            << " Charge " << charge << " dE/dx " << dedx 
                            << " Birk Const " << rkb << ", " << c << " Weight = " 
                            << weight << " dE " << aStep->GetTotalEnergyDeposit();
#endif
  }
  return weight;
}

void CaloSD::update(const BeginOfRun *) {
  initRun();
} 

void CaloSD::update(const BeginOfEvent *) {
#ifdef DebugLog
  edm::LogInfo("CaloSim")  << "CaloSD: Dispatched BeginOfEvent for " 
                           << GetName() << " !" ;
#endif
  clearHits();
}

void CaloSD::update(const EndOfTrack * trk) {
  int id = (*trk)()->GetTrackID();
  TrackInformation *trkI =(TrackInformation *)((*trk)()->GetUserInformation());
  int lastTrackID = -1;
  if (trkI) lastTrackID = trkI->getIDonCaloSurface();
  if (id == lastTrackID) {
    const TrackContainer * trksForThisEvent = m_trackManager->trackContainer();
    if (trksForThisEvent != nullptr) {
      int it = (int)(trksForThisEvent->size()) - 1;
      if (it >= 0) {
        TrackWithHistory * trkH = (*trksForThisEvent)[it];
        if (trkH->trackID() == (unsigned int)(id)) tkMap[id] = trkH;
#ifdef DebugLog
        edm::LogInfo("CaloSim") << "CaloSD: get track " << it << " from "
                                << "Container of size " << trksForThisEvent->size()
                                << " with ID " << trkH->trackID();
      } else {
        edm::LogInfo("CaloSim") << "CaloSD: get track " << it << " from "
                                << "Container of size " << trksForThisEvent->size()
                                << " with no ID";
#endif
      }
    }
  }
}

void CaloSD::update(const ::EndOfEvent * ) {
  int count = 0, wrong = 0;
  bool ok;
  
  slave.get()->ReserveMemory(theHC->entries());

  for (int i=0; i<theHC->entries(); ++i) {
    ok = saveHit((*theHC)[i]);
    ++count;
    if (!ok)  ++wrong;
  }
  
  edm::LogInfo("CaloSim") << "CaloSD: " << GetName() << " store " << count
                          << " hits recorded with " << wrong 
                          << " track IDs not given properly and "
                          << totalHits-count << " hits not passing cuts";
  summarize();

  tkMap.erase (tkMap.begin(), tkMap.end());
}

void CaloSD::clearHits() {  
  if (useMap) hitMap.erase (hitMap.begin(), hitMap.end());
  for (unsigned int i = 0; i<reusehit.size(); ++i) delete reusehit[i];
  std::vector<CaloG4Hit*>().swap(reusehit);
  cleanIndex  = 0;
  previousID.reset();
  primIDSaved = -99;
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Clears hit vector for " << GetName() 
                          << " and initialise slave: " << slave;
#endif
  slave.get()->Initialize();
}

void CaloSD::initRun() {}

int CaloSD::getTrackID(const G4Track* aTrack) {

  int primaryID = 0;
  forceSave = false;
  TrackInformation* trkInfo=(TrackInformation *)(aTrack->GetUserInformation());
  if (trkInfo) {
    int id = trkInfo->getIDonCaloSurface();
    if(id > 0) { primaryID = id; } 
  } else {
    primaryID = aTrack->GetTrackID();
  }
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD::getTrackID for " << GetName() 
                          << " trackID= " << aTrack->GetTrackID()
                          << " primaryID= " << primaryID;
#endif
  return primaryID;
}

int CaloSD::setTrackID(const G4Step* aStep) {

  auto const theTrack = aStep->GetTrack();
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
    primaryID = theTrack->GetTrackID();
  }

  if (primaryID != previousID.trackID()) {
    resetForNewPrimary(aStep);
  }
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD::setTrackID for " << GetName() 
                          << " trackID= " << aStep->GetTrack()->GetTrackID()
                          << " primaryID= " << primaryID;
#endif
  return primaryID;
}

uint16_t CaloSD::getDepth(const G4Step*) { return 0; }

bool CaloSD::filterHit(CaloG4Hit* hit, double time) {
  double emin(eminHit);
  if (hit->getDepth() > 0) emin = eminHitD;
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD::filterHit(..) Depth " << hit->getDepth() 
                          << " Emin = " << emin 
                          << " (" << eminHit << ", " << eminHitD << ")";
#endif   
  return ((time <= tmaxHit) && (hit->getEnergyDeposit() > emin));
}

double CaloSD::getResponseWt(const G4Track* aTrack) {
  double wt = 1.0;
  if (meanResponse.get()) {
    TrackInformation * trkInfo = (TrackInformation *)(aTrack->GetUserInformation());
    wt = meanResponse.get()->getWeight(trkInfo->genParticlePID(), trkInfo->genParticleP());
  }
  return wt;
}

void CaloSD::storeHit(CaloG4Hit* hit) {
  if (previousID.trackID()<0) return;
  if (hit == nullptr) {
    edm::LogWarning("CaloSim") << "CaloSD: hit to be stored is NULL !!";
    return;
  }
  
  theHC->insert(hit);
  if (useMap) hitMap.insert(std::pair<CaloHitID,CaloG4Hit*>(previousID,hit));
}

bool CaloSD::saveHit(CaloG4Hit* aHit) {  
  int tkID;
  bool ok   = true;
  if (m_trackManager) {
    tkID = m_trackManager->giveMotherNeeded(aHit->getTrackID());
    if (tkID == 0) {
      if (m_trackManager->trackExists(aHit->getTrackID())) tkID = (aHit->getTrackID());
      else {
        ok = false;
      }
    }
  } else {
    tkID = aHit->getTrackID();
    ok = false;
  }
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CalosD: Track ID " << aHit->getTrackID() 
                          << " changed to " << tkID << " by SimTrackManager"
                          << " Status " << ok;
#endif
  double time = aHit->getTimeSlice();
  if (corrTOFBeam) time += correctT;
  slave.get()->processHits(aHit->getUnitID(), aHit->getEM()/GeV, 
                           aHit->getHadr()/GeV, time, tkID, aHit->getDepth());
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Store Hit at " << std::hex 
                          << aHit->getUnitID() << std::dec << " " 
                          << aHit->getDepth() << " due to " << tkID 
                          << " in time " << time << " of energy " 
                          << aHit->getEM()/GeV << " GeV (EM) and " 
                          << aHit->getHadr()/GeV << " GeV (Hadr)";
#endif
  return ok;
}

void CaloSD::summarize() {}

void CaloSD::update(const BeginOfTrack * trk) {
  int primary = -1;
  TrackInformation * trkInfo = (TrackInformation *)((*trk)()->GetUserInformation());
  if ( trkInfo->isPrimary() ) primary = (*trk)()->GetTrackID();
  
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "New track: isPrimary " << trkInfo->isPrimary() 
                          << " primary ID = " << primary 
                          << " primary ancestor ID " << primAncestor;
#endif
  
  // update the information if a different primary track ID 
  
  if (primary > 0 && primary != primAncestor) {
    primAncestor = primary;
    
    // clean the hits information
    
    if (theHC->entries()>0) cleanHitCollection();
    
  }
}

void CaloSD::cleanHitCollection() {
  std::vector<CaloG4Hit*>* theCollection = theHC->GetVector();

#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: collection before merging, size = " 
                          << theHC->entries();
#endif
  
  selIndex.reserve(theHC->entries()-cleanIndex);
  if ( reusehit.empty() ) reusehit.reserve(theHC->entries()-cleanIndex); 

  // if no map used, merge before hits to have the save situation as a map
  if ( !useMap ) {
    hitvec.swap(*theCollection);
    sort((hitvec.begin()+cleanIndex), hitvec.end(), CaloG4HitLess());
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD::cleanHitCollection: sort hits in buffer "
                            << "starting from element = " << cleanIndex;
    for (unsigned int i = 0; i<hitvec.size(); ++i) 
      edm::LogInfo("CaloSim") << i << " " << *hitvec[i];
#endif
    unsigned int i, j;
    CaloG4HitEqual equal;
    for (i=cleanIndex; i<hitvec.size(); ++i) {
      selIndex.push_back(i-cleanIndex);
      int jump = 0;
      for (j = i+1; j <hitvec.size() && equal(hitvec[i], hitvec[j]); ++j) {
        ++jump;
        // merge j to i
        (*hitvec[i]).addEnergyDeposit(*hitvec[j]);
        (*hitvec[j]).setEM(0.);
        (*hitvec[j]).setHadr(0.);
        reusehit.push_back(hitvec[j]);
      }
      i+=jump;
    }
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD: cleanHitCollection merge the hits in buffer ";
    for (unsigned int i = 0; i<hitvec.size(); ++i) 
      edm::LogInfo("CaloSim") << i << " " << *hitvec[i];
#endif
    for ( unsigned int i = cleanIndex; i < cleanIndex+selIndex.size(); ++i ) {
      hitvec[i] = hitvec[selIndex[i-cleanIndex]+cleanIndex];
    }
    hitvec.resize(cleanIndex+selIndex.size());
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD::cleanHitCollection: remove the merged hits in buffer,"
                            << " new size = " << hitvec.size();
    for (unsigned int i = 0; i<hitvec.size(); ++i) 
      edm::LogInfo("CaloSim") << i << " " << *hitvec[i];
#endif
    hitvec.swap(*theCollection);
    std::vector<CaloG4Hit*>().swap(hitvec);
    selIndex.clear();
    totalHits = theHC->entries();
  }

#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: collection after merging, size= " << theHC->entries()
                          << " Size of reusehit= " << reusehit.size()
                          << "\n      starting hit selection from index = " << cleanIndex;
#endif
  
  int addhit = 0;
  selIndex.reserve(theCollection->size()-cleanIndex);
  for (unsigned int i = cleanIndex; i<theCollection->size(); ++i) {   
    CaloG4Hit* aHit((*theCollection)[i]);
    
    // selection
    
    double time = aHit->getTimeSlice();
    if (corrTOFBeam) time += correctT;
    if (!filterHit(aHit,time)) {
#ifdef DebugLog
      edm::LogInfo("CaloSim") << "CaloSD: dropped CaloG4Hit " << " " << *aHit; 
#endif
      
      // create the list of hits to be reused
      
      reusehit.push_back((*theCollection)[i]);
      ++addhit;
    } else {
      selIndex.push_back(i-cleanIndex);
    }
  }

#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Size of reusehit after selection = " 
                          << reusehit.size() << " Number of added hit = " 
                          << addhit;
#endif
  if (useMap) {
    if ( addhit>0 ) {
      int offset = reusehit.size()-addhit;
      for (int ii = addhit-1; ii>=0; --ii) {
        CaloHitID theID = reusehit[offset+ii]->getID();
        hitMap.erase(theID);
      }
    }
  }
  for (unsigned int j = 0; j<selIndex.size(); ++j) {
    (*theCollection)[cleanIndex+j] = (*theCollection)[cleanIndex+selIndex[j]];
  }

  theCollection->resize(cleanIndex+selIndex.size());
  std::vector<unsigned int>().swap(selIndex);

#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: hit collection after selection, size = "
                          << theHC->entries();
  theHC->PrintAllHits();
#endif
    
  cleanIndex = theHC->entries();
}
