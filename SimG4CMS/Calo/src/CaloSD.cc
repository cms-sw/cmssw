///////////////////////////////////////////////////////////////////////////////
// File: CaloSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
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

CaloSD::CaloSD(G4String name, const DDCompactView & cpv,
        const SensitiveDetectorCatalog & clg,
        edm::ParameterSet const & p, const SimTrackManager* manager,
        float timeSliceUnit, bool ignoreTkID) : 
  SensitiveCaloDetector(name, cpv, clg, p),
  G4VGFlashSensitiveDetector(), theTrack(0), preStepPoint(0), eminHit(0), 
  eminHitD(0), m_trackManager(manager), currentHit(0), runInit(false),
  timeSlice(timeSliceUnit), ignoreTrackID(ignoreTkID), hcID(-1), theHC(0), 
  meanResponse(0) {
  //Add Hcal Sentitive Detector Names

  collectionName.insert(name);

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
  for (unsigned int k=0; k<hcn.size(); ++k) {
    if (name == (G4String)(hcn[k])) {
      if (k < eminHits.size()) eminHit = eminHits[k]*MeV;
      if (k < eminHitX.size()) eminHitD= eminHitX[k]*MeV;
      if (k < tmaxHits.size()) tmaxHit = tmaxHits[k]*ns;
      if (k < useResMap.size() && useResMap[k] > 0) meanResponse = new CaloMeanResponse(p);
      break;
    }
  }
#ifdef DebugLog
  LogDebug("CaloSim") << "***************************************************" 
                      << "\n"
                      << "*                                                 *" 
                      << "\n"
                      << "* Constructing a CaloSD  with name " << GetName()
                      << "\n"
                      << "*                                                 *" 
                      << "\n"
                      << "***************************************************";
#endif
  slave      = new CaloSlaveSD(name);
  currentID  = CaloHitID(timeSlice, ignoreTrackID);
  previousID = CaloHitID(timeSlice, ignoreTrackID);
  
  primAncestor = 0;
  cleanIndex = 0;
  totalHits = 0;
  forceSave = false;

  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  const std::vector<std::string>& lvNames = clg.logicalNames(name);
  this->Register();
  for (std::vector<std::string>::const_iterator it=lvNames.begin(); it !=lvNames.end(); ++it) {
    this->AssignSD(*it);
#ifdef DebugLog
    LogDebug("CaloSim") << "CaloSD : Assigns SD to LV " << (*it);
#endif
  }

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
                          << " MeV (for nonzero depths); Time Slice Unit " 
                          << timeSlice << " Ignore TrackID Flag " << ignoreTrackID;
}

CaloSD::~CaloSD() { 
  if (slave)           delete slave; 
  if (theHC)           delete theHC;
  if (meanResponse)    delete meanResponse;
}

bool CaloSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {
  
  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    if (getStepInfo(aStep)) {
      if (hitExists() == false && edepositEM+edepositHAD>0.) 
        currentHit = createNewHit();
    }
  }
  return true;
} 

bool CaloSD::ProcessHits(G4GFlashSpot* aSpot, G4TouchableHistory*) { 

  if (aSpot != NULL) {   
    theTrack = const_cast<G4Track *>(aSpot->GetOriginatorTrack()->GetPrimaryTrack());
    G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
    
    if (particleCode == emPDG ||
        particleCode == epPDG ||
        particleCode == gammaPDG ) {
      edepositEM  = aSpot->GetEnergySpot()->GetEnergy();
      edepositHAD = 0.;
    } else {
      edepositEM  = 0.;
      edepositHAD = 0.;
    }
 
    if (edepositEM>0.) {
      G4Step *      fFakeStep          = new G4Step();
      preStepPoint                     = fFakeStep->GetPreStepPoint();
      G4StepPoint * fFakePostStepPoint = fFakeStep->GetPostStepPoint();
      preStepPoint->SetPosition(aSpot->GetPosition());
      fFakePostStepPoint->SetPosition(aSpot->GetPosition());
      
      G4TouchableHandle fTouchableHandle   = aSpot->GetTouchableHandle();
      preStepPoint->SetTouchableHandle(fTouchableHandle);
      fFakeStep->SetTotalEnergyDeposit(aSpot->GetEnergySpot()->GetEnergy());
      
      double       time   = 0;
      unsigned int unitID = setDetUnitId(fFakeStep);
      int          primaryID = getTrackID(theTrack);
      uint16_t     depth = getDepth(fFakeStep);

      if (unitID > 0) {
        currentID.setID(unitID, time, primaryID, depth);
#ifdef DebugLog
        LogDebug("CaloSim") << "CaloSD:: GetSpotInfo for"
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
            entrancePoint  = aSpot->GetPosition();
            entranceLocal  = aSpot->GetTouchableHandle()->GetHistory()->
                                      GetTopTransform().TransformPoint(entrancePoint);
            incidentEnergy = theTrack->GetKineticEnergy();
#ifdef DebugLog
            LogDebug("CaloSim") << "CaloSD: Incident energy " 
                                << incidentEnergy/GeV << " GeV and" 
                                << " entrance point " << entrancePoint 
                                << " (Global) " << entranceLocal << " (Local)";
#endif
          }

          if (checkHit() == false) currentHit = createNewHit();
        }
      }
      delete  fFakeStep;
    }
    return true;
  } 
  return false;
}                                   

double CaloSD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void CaloSD::Initialize(G4HCofThisEvent * HCE) { 
  totalHits = 0;
  
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD : Initialize called for " << GetName(); 
#endif
  
  //This initialization is performed at the beginning of an event
  //------------------------------------------------------------
  theHC = new CaloG4HitCollection(GetName(), collectionName[0]);
  
  if (hcID<0) hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);
}

void CaloSD::EndOfEvent(G4HCofThisEvent* ) {
  // clean the hits for the last tracks
  
  cleanHitCollection();
  
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: EndofEvent entered with " << theHC->entries()
                          << " entries";
#endif
  //  TimeMe("CaloSD:sortAndMergeHits",false);
}

void CaloSD::clear() {} 

void CaloSD::DrawAll() {} 

void CaloSD::PrintAll() {
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Collection " << theHC->GetName();
#endif
  theHC->PrintAllHits();
} 

void CaloSD::fillHits(edm::PCaloHitContainer& c, std::string n) {
  if (slave->name() == n) c=slave->hits();
  slave->Clean();
}

bool CaloSD::getStepInfo(G4Step* aStep) {  

  preStepPoint = aStep->GetPreStepPoint(); 
  theTrack     = aStep->GetTrack();   
  
  double       time  = (aStep->GetPostStepPoint()->GetGlobalTime())/nanosecond;
  unsigned int unitID= setDetUnitId(aStep);
  uint16_t     depth = getDepth(aStep);
  int          primaryID = getTrackID(theTrack);
  
  bool flag = (unitID > 0);
  if (flag) {
    currentID.setID(unitID, time, primaryID, depth);
#ifdef DebugLog
    G4TouchableHistory* touch =(G4TouchableHistory*)(theTrack->GetTouchable());
    edm::LogInfo("CaloSim") << "CaloSD:: GetStepInfo for"
			    << " PV "     << touch->GetVolume(0)->GetName()
			    << " PVid = " << touch->GetReplicaNumber(0)
			    << " MVid = " << touch->GetReplicaNumber(1)
			    << " Unit   " << currentID.unitID() 
			    << " Edeposit = " << edepositEM << " " << edepositHAD;
  } else {
    G4TouchableHistory* touch =(G4TouchableHistory*)(theTrack->GetTouchable());
    edm::LogInfo("CaloSim") << "CaloSD:: GetStepInfo for"
			    << " PV "     << touch->GetVolume(0)->GetName()
			    << " PVid = " << touch->GetReplicaNumber(0)
			    << " MVid = " << touch->GetReplicaNumber(1)
			    << " Unit   " << std::hex << unitID << std::dec 
			    << " Edeposit = " << edepositEM << " " << edepositHAD;
#endif
  }
  
  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  if (particleCode == emPDG ||
      particleCode == epPDG ||
      particleCode == gammaPDG ) {
    edepositEM  = getEnergyDeposit(aStep);
    edepositHAD = 0.;
  } else {
    edepositEM  = 0.;
    edepositHAD = getEnergyDeposit(aStep);
  }

  return flag;
}

G4ThreeVector CaloSD::setToLocal(const G4ThreeVector& global, const G4VTouchable* touch) {

  G4ThreeVector localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  
  return localPoint;  
}

G4ThreeVector CaloSD::setToGlobal(const G4ThreeVector& local, const G4VTouchable* touch) {

  G4ThreeVector globalPoint = touch->GetHistory()->GetTopTransform().Inverse().TransformPoint(local);
  
  return globalPoint;  
}

G4bool CaloSD::hitExists() {
#ifdef DebugLog
  if (currentID.trackID()<1)
    edm::LogWarning("CaloSim") << "***** CaloSD error: primaryID = " 
                               << currentID.trackID()
                               << " maybe detector name changed";
#endif  
  // Update if in the same detector, time-slice and for same track   
  if (currentID == previousID) {
    updateHit(currentHit);
    return true;
  }
  
  // Reset entry point for new primary
  posGlobal = preStepPoint->GetPosition();
  if (currentID.trackID() != previousID.trackID()) 
    resetForNewPrimary(preStepPoint->GetPosition(), preStepPoint->GetKineticEnergy());
  
  return checkHit();
}

G4bool CaloSD::checkHit() {  
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
    
    for (int j=maxhit; j>minhit&&!found; --j) {
      if ((*theHC)[j]->getID() == currentID) {
        currentHit = (*theHC)[j];
        found      = true;
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

CaloG4Hit* CaloSD::createNewHit() {
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD::CreateNewHit for"
			  << " Unit " << currentID.unitID() 
			  << " " << currentID.depth()
			  << " Edeposit = " << edepositEM << " " << edepositHAD;
  edm::LogInfo("CaloSim") << " primary "    << currentID.trackID()
			  << " time slice " << currentID.timeSliceID()
			  << " For Track  " << theTrack->GetTrackID()
			  << " which is a " <<theTrack->GetDefinition()->GetParticleName()
			  << " of energy "  << theTrack->GetKineticEnergy()/GeV
			  << " " << theTrack->GetMomentum().mag()/GeV
			  << " daughter of part. " << theTrack->GetParentID()
			  << " and created by " ;
  
  if (theTrack->GetCreatorProcess()!=NULL)
    edm::LogInfo("CaloSim") << theTrack->GetCreatorProcess()->GetProcessName();
  else 
    edm::LogInfo("CaloSim") << "NO process";
#endif  
  
  CaloG4Hit* aHit;
  if (reusehit.size() > 0) {
    aHit = reusehit[0];
    aHit->setEM(0.);
    aHit->setHadr(0.);
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
    //edm::LogInfo("CaloSim") << "CaloSD: set save the track " << currentID.trackID()
    //      << " etrack " << etrack << " eCut " << energyCut << " flag " << forceSave;
    if (etrack >= energyCut || forceSave) {
      TrackInformation* trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
      trkInfo->storeTrack(true);
      trkInfo->putInHistory();
      //      trkInfo->setAncestor();
#ifdef DebugLog
      edm::LogInfo("CaloSim") << "CaloSD: set save the track " 
			      << currentID.trackID() << " with Hit";
#endif
    }
  } else {
    TrackWithHistory * trkh = tkMap[currentID.trackID()];
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD : TrackwithHistory pointer for " 
			    << currentID.trackID() << " is " << trkh;
#endif
    if (trkh != NULL) {
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
  if (useMap) totalHits++;
  return aHit;
}  

void CaloSD::updateHit(CaloG4Hit* aHit) {
  if (edepositEM+edepositHAD != 0) {
    aHit->addEnergyDeposit(edepositEM,edepositHAD);
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD: Add energy deposit in " << currentID 
			    << " em " << edepositEM/MeV << " hadronic " 
			    << edepositHAD/MeV << " MeV"; 
#endif
  }

  // buffer for next steps:
  previousID = currentID;
}

void CaloSD::resetForNewPrimary(const G4ThreeVector& point, double energy) { 
  entrancePoint  = point;
  entranceLocal  = setToLocal(entrancePoint, preStepPoint->GetTouchable());
  incidentEnergy = energy;
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Incident energy " << incidentEnergy/GeV 
			  << " GeV and" << " entrance point " << entrancePoint 
			  << " (Global) " << entranceLocal << " (Local)";
#endif
}

double CaloSD::getAttenuation(G4Step* aStep, double birk1, double birk2, double birk3) {
  double weight = 1.;
  double charge = aStep->GetPreStepPoint()->GetCharge();

  if (charge != 0. && aStep->GetStepLength() > 0) {
    G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    double density = mat->GetDensity();
    double dedx    = aStep->GetTotalEnergyDeposit()/aStep->GetStepLength();
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
  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  emPDG = theParticleTable->FindParticle(particleName="e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle(particleName="e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle(particleName="gamma")->GetPDGEncoding();
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Particle code for e- = " << emPDG
			  << " for e+ = " << epPDG << " for gamma = " << gammaPDG;
#endif
  initRun();
  runInit = true;
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
    if (trksForThisEvent != NULL) {
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
  
  slave->ReserveMemory(theHC->entries());

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
  edm::LogInfo("CaloSim") << "CaloSD: Clears hit vector for " << GetName() << " " << slave;
#endif
  slave->Initialize();
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Initialises slave SD for " << GetName();
#endif
}

void CaloSD::initRun() {}

int CaloSD::getTrackID(G4Track* aTrack) {
  int primaryID = 0;
  forceSave = false;
  TrackInformation* trkInfo=(TrackInformation *)(aTrack->GetUserInformation());
  if (trkInfo) {
    primaryID = trkInfo->getIDonCaloSurface(); 
#ifdef DebugLog
    edm::LogInfo("CaloSim") << "CaloSD: hit update from track Id on Calo Surface " 
			    << trkInfo->getIDonCaloSurface();
#endif   
  } else {
    primaryID = aTrack->GetTrackID();
#ifdef DebugLog
    edm::LogWarning("CaloSim") << "CaloSD: Problem with primaryID **** set by "
                               << "force to TkID **** " << primaryID << " in "
                               << preStepPoint->GetTouchable()->GetVolume(0)->GetName();
#endif
  }
  return primaryID;
}

uint16_t CaloSD::getDepth(G4Step*) { return 0; }

bool CaloSD::filterHit(CaloG4Hit* hit, double time) {
  double emin(eminHit);
  if (hit->getDepth() > 0) emin = eminHitD;
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "Depth " << hit->getDepth() << " Emin = " << emin 
			  << " (" << eminHit << ", " << eminHitD << ")";
#endif   
  return ((time <= tmaxHit) && (hit->getEnergyDeposit() > emin));
}

double CaloSD::getResponseWt(G4Track* aTrack) {
  if (meanResponse) {
    TrackInformation * trkInfo = (TrackInformation *)(aTrack->GetUserInformation());
    return meanResponse->getWeight(trkInfo->genParticlePID(), trkInfo->genParticleP());
  } else {
    return 1;
  }
}

void CaloSD::storeHit(CaloG4Hit* hit) {
  if (previousID.trackID()<0) return;
  if (hit == 0) {
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
  //  edm::LogInfo("CaloSim") << "CalosD: Track ID " << aHit->getTrackID() << " changed to " << tkID << " by SimTrackManager" << " Status " << ok;
#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CalosD: Track ID " << aHit->getTrackID() 
			  << " changed to " << tkID << " by SimTrackManager"
			  << " Status " << ok;
#endif
  double time = aHit->getTimeSlice();
  if (corrTOFBeam) time += correctT;
  slave->processHits(aHit->getUnitID(), aHit->getEM()/GeV, 
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
  edm::LogInfo("CaloSim") << "CaloSD: collection before merging, size = " << theHC->entries();
#endif
  
  selIndex.reserve(theHC->entries()-cleanIndex);
  if ( reusehit.size() == 0 ) reusehit.reserve(theHC->entries()-cleanIndex); 

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
  edm::LogInfo("CaloSim") << "CaloSD: collection after merging, size = " << theHC->entries();
#endif

  int addhit = 0;

#ifdef DebugLog
  edm::LogInfo("CaloSim") << "CaloSD: Size of reusehit after merge = " << reusehit.size();
  edm::LogInfo("CaloSim") << "CaloSD: Starting hit selection from index = " << cleanIndex;
#endif
  
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
