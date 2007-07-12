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

CaloSD::CaloSD(G4String name, const DDCompactView & cpv,
	       SensitiveDetectorCatalog & clg, 
	       edm::ParameterSet const & p, const SimTrackManager* manager) : 
  SensitiveCaloDetector(name, cpv, clg, p),
  G4VGFlashSensitiveDetector(), 
  theTrack(0), preStepPoint(0), m_trackManager(manager), currentHit(0),
  hcID(-1), theHC(0) {
  //Add Hcal Sentitive Detector Names

  collectionName.insert(name);

  //Parameters
  //  static SimpleConfigurable<float> pEmin(1.0,"CaloSD:EminTrack");
  //   static SimpleConfigurable<int>   pcheckHit(25,"CaloSD:CheckHits");
  //   static SimpleConfigurable<bool>  pUseMap(false,"CaloSD:UseMap");
  edm::ParameterSet m_CaloSD = p.getParameter<edm::ParameterSet>("CaloSD");
  energyCut = m_CaloSD.getParameter<double>("EminTrack")*GeV;
  checkHits = m_CaloSD.getUntrackedParameter<int>("CheckHits", 25);
  useMap    = m_CaloSD.getUntrackedParameter<bool>("UseMap", true);
  int verbn = m_CaloSD.getUntrackedParameter<int>("Verbosity", 0);
  bool on   = m_CaloSD.getUntrackedParameter<bool>("DetailedTiming");
  corrTOFBeam = m_CaloSD.getUntrackedParameter<bool>("CorrectTOFBeam", false);
  double beamZ= m_CaloSD.getUntrackedParameter<double>("BeamPosition",0.0)*cm;
  correctT    = beamZ/c_light/nanosecond;

  SetVerboseLevel(verbn);
  LogDebug("CaloSim") << "***************************************************" 
		      << "\n"
		      << "*                                                 *" 
		      << "\n"
		      << "* Constructing a CaloSD  with name " << GetName()
		      << "\n"
		      << "*                                                 *" 
		      << "\n"
		      << "***************************************************";

  slave      = new CaloSlaveSD(name);
  currentID  = CaloHitID();
  previousID = CaloHitID();

  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  std::vector<std::string> lvNames = clg.logicalNames(name);
  this->Register();
  for (std::vector<std::string>::iterator it=lvNames.begin();
       it !=lvNames.end(); it++){
    this->AssignSD(*it);
    LogDebug("CaloSim") << "CaloSD : Assigns SD to LV " << (*it);
  }

  // timer initialization
  if (on) {
  //     string trname("CaloSD:");
  //     theHitTimer.init( trname + name + ":hits", true);
  //   }
  //   else {
  //     theHitTimer.init( "CaloSensitiveDetector:hits", true);
  }

  edm::LogInfo("CaloSim") << "CaloSD: Minimum energy of track for saving it " 
			  << energyCut/GeV  << " GeV" << "\n"
			  << "        Use of HitID Map " << useMap << "\n"
			  << "        Check last " << checkHits 
			  << " before saving the hit\n" 
			  << "        Correct TOF globally by " << correctT
			  << " ns (Flag =" << corrTOFBeam << ")";
}

CaloSD::~CaloSD() { 
  if (slave)           delete slave; 
  if (theHC)           delete theHC;
}

bool CaloSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {
  //  TimeMe t1( theHitTimer, false);
  
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
    G4String particleType = theTrack->GetDefinition()->GetParticleName();
    TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
		
    if (particleType == "e-" ||
	particleType == "e+" ||
	particleType == "gamma" ) {
      edepositEM  = aSpot->GetEnergySpot()->GetEnergy(); edepositHAD = 0.;
    } else {
      edepositEM  = 0.;  edepositHAD = 0.;
    }
	
    if (edepositEM>0.)  {
      G4Step *      fFakeStep          = new G4Step();
      G4StepPoint * fFakePreStepPoint  = fFakeStep->GetPreStepPoint();
      G4StepPoint * fFakePostStepPoint = fFakeStep->GetPostStepPoint();
      fFakePreStepPoint->SetPosition(aSpot->GetPosition());
      fFakePostStepPoint->SetPosition(aSpot->GetPosition());
      
      G4TouchableHandle fTouchableHandle   = aSpot->GetTouchableHandle();
      fFakePreStepPoint->SetTouchableHandle(fTouchableHandle);
      fFakeStep->SetTotalEnergyDeposit(aSpot->GetEnergySpot()->GetEnergy());
      
      double       time   = 0;
      unsigned int unitID = setDetUnitId(fFakeStep);
      int      primaryID;

      if (trkInfo)
	primaryID  = trkInfo->getIDonCaloSurface();
      else
	primaryID = 0;
      
      if (primaryID == 0) {
        edm::LogWarning("CaloSim") << "CaloSD: Spot Problem with primaryID "
				   << "**** set by force to  **** " 
				   << theTrack->GetTrackID(); 
        primaryID = theTrack->GetTrackID();
      }
      if (unitID > 0) {
	currentID.setID(unitID, time, primaryID);
	LogDebug("CaloSim") << "CaloSD:: GetSpotInfo for"
			    << " Unit 0x" << std::hex << currentID.unitID() 
			    << std::dec << " Edeposit = " << edepositEM << " " 
			    << edepositHAD;
	// Update if in the same detector, time-slice and for same track   
	if (currentID == previousID) {
	  updateHit(currentHit);
	} else {
        
	  posGlobal = aSpot->GetPosition();
	  // Reset entry point for new primary
	  if (currentID.trackID() != previousID.trackID()) {
	    entrancePoint  = aSpot->GetPosition();
	    entranceLocal  = aSpot->GetTouchableHandle()->GetHistory()->
	      GetTopTransform().TransformPoint(entrancePoint);
	    incidentEnergy = theTrack->GetKineticEnergy();
	    LogDebug("CaloSim") << "CaloSD: Incident energy " 
				<< incidentEnergy/GeV << " GeV and" 
				<< " entrance point " << entrancePoint 
				<< " (Global) " << entranceLocal << " (Local)";
	  }
	
	  if (checkHit() == false) currentHit = createNewHit();
	}
      }
      
      delete 	fFakeStep;
    }
    return true;
    
  } 
  return false;
}                                   

double CaloSD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void CaloSD::Initialize(G4HCofThisEvent * HCE) { 

  LogDebug("CaloSim") << "CaloSD : Initialize called for " << GetName(); 

  //This initialization is performed at the beginning of an event
  //------------------------------------------------------------
  theHC = new CaloG4HitCollection(GetName(), collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);
}

bool CaloSD::getStepInfo(G4Step* aStep) {
  
  preStepPoint = aStep->GetPreStepPoint(); 
  theTrack     = aStep->GetTrack();   

  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  if (particleType == "e-" ||
      particleType == "e+" ||
      particleType == "gamma" ){
    edepositEM  = getEnergyDeposit(aStep); edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = getEnergyDeposit(aStep);
  }

  double       time  = (aStep->GetPostStepPoint()->GetGlobalTime())/nanosecond;
  unsigned int unitID= setDetUnitId(aStep);
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int      primaryID;

  if (trkInfo)
    primaryID = trkInfo->getIDonCaloSurface();	
  else
    primaryID = 0;

  if (primaryID == 0) {
    edm::LogWarning("CaloSim") << "CaloSD: Problem with primaryID **** set by "
			       << "force to TkID **** " 
			       << theTrack->GetTrackID() << " in Volume "
			       << preStepPoint->GetTouchable()->GetVolume(0)->GetName();
    primaryID = theTrack->GetTrackID();
  }

  bool flag = (unitID > 0);
  G4TouchableHistory* touch =(G4TouchableHistory*)(theTrack->GetTouchable());
  if (flag) {
    currentID.setID(unitID, time, primaryID);

    LogDebug("CaloSim") << "CaloSD:: GetStepInfo for"
			<< " PV "     << touch->GetVolume(0)->GetName()
			<< " PVid = " << touch->GetReplicaNumber(0)
			<< " MVid = " << touch->GetReplicaNumber(1)
			<< " Unit   " << currentID.unitID() 
			<< " Edeposit = " << edepositEM << " " << edepositHAD;
  } else {
    LogDebug("CaloSim") << "CaloSD:: GetStepInfo for"
			<< " PV "     << touch->GetVolume(0)->GetName()
			<< " PVid = " << touch->GetReplicaNumber(0)
			<< " MVid = " << touch->GetReplicaNumber(1)
			<< " Unit   " << std::hex << unitID << std::dec 
			<< " Edeposit = " << edepositEM << " " << edepositHAD;
  }
  return flag;
}

G4bool CaloSD::hitExists() {
   
  if (currentID.trackID()<1) {
    edm::LogWarning("CaloSim") << "***** CaloSD error: primaryID = " 
			       << currentID.trackID()
			       << " maybe detector name changed";
  }
  
  // Update if in the same detector, time-slice and for same track   
  if (currentID == previousID) {
    updateHit(currentHit);
    return true;
  }
   
  // Reset entry point for new primary
  posGlobal = preStepPoint->GetPosition();
  if (currentID.trackID() != previousID.trackID()) 
    resetForNewPrimary(preStepPoint->GetPosition(),
		       preStepPoint->GetKineticEnergy());

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

    for (int j=maxhit; j>minhit&&!found; j--) {
      CaloG4Hit* aPreviousHit = (*theHC)[j];
      if (aPreviousHit->getID() == currentID) {
	currentHit = aPreviousHit;
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

void CaloSD::resetForNewPrimary(G4ThreeVector point, double energy) {
  
  entrancePoint  = point;
  entranceLocal  = setToLocal(entrancePoint, preStepPoint->GetTouchable());
  incidentEnergy = energy;
  LogDebug("CaloSim") << "CaloSD: Incident energy " << incidentEnergy/GeV 
		      << " GeV and" << " entrance point " << entrancePoint 
		      << " (Global) " << entranceLocal << " (Local)";
}

double CaloSD::getAttenuation(G4Step* aStep, double birk1, double birk2) {

  double weight = 1.;
  double charge = aStep->GetPreStepPoint()->GetCharge();

  if (charge != 0. && aStep->GetStepLength() > 0) {
    G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    double density = mat->GetDensity();
    double dedx    = aStep->GetTotalEnergyDeposit()/aStep->GetStepLength();
    double rkb     = birk1/density;
    double c       = birk2/(density*density);
    if (std::abs(charge) >= 2.) rkb *= 7.2/12.6; // based on alpha particle data
    weight = 1./(1.+rkb*dedx+c*dedx*dedx);
    LogDebug("CaloSim") << "CaloSD::getAttenuation in " << mat->GetName() 
			<< " Charge " << charge << " dE/dx " << dedx 
			<< " Birk Const " << rkb << ", " << c << " Weight = " 
			<< weight << " dE " << aStep->GetTotalEnergyDeposit();
  }
  return weight;
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

CaloG4Hit* CaloSD::createNewHit() {

  LogDebug("CaloSim") << "CaloSD::CreateNewHit for"
		      << " Unit " << currentID.unitID() 
		      << " Edeposit = " << edepositEM << " " << edepositHAD;
  LogDebug("CaloSim") << " primary "    << currentID.trackID()
		      << " time slice " << currentID.timeSliceID()
		      << " For Track  " << theTrack->GetTrackID()
		      << " which is a " <<theTrack->GetDefinition()->GetParticleName()
		      << " of energy "  << theTrack->GetKineticEnergy()/GeV
		      << " " << theTrack->GetMomentum().mag()/GeV
		      << " daughter of part. " << theTrack->GetParentID()
		      << " and created by " ;
  
  if (theTrack->GetCreatorProcess()!=NULL)
    LogDebug("CaloSim") << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    LogDebug("CaloSim") << "NO process";
  
  CaloG4Hit* aHit = new CaloG4Hit;
  aHit->setID(currentID);
  aHit->setEntry(entrancePoint);
  aHit->setEntryLocal(entranceLocal);
  aHit->setPosition(posGlobal);
  aHit->setIncidentEnergy(incidentEnergy);
  updateHit(aHit);
  
  storeHit(aHit);
  double etrack = 0;
  if (currentID.trackID() == primIDSaved) { // The track is saved; nothing to be done
  } else if (currentID.trackID() == theTrack->GetTrackID()) {
    etrack= theTrack->GetKineticEnergy();
    if (etrack >= energyCut) {
      TrackInformation * trkInfo = 
	(TrackInformation *)(theTrack->GetUserInformation());
      trkInfo->storeTrack(true);
      trkInfo->putInHistory();
      LogDebug("CaloSim") << "CaloSD: set save the track " 
			  << currentID.trackID() << " with Hit";
    }
  } else {
    TrackWithHistory * trkh = tkMap[currentID.trackID()];
    LogDebug("CaloSim") << "CaloSD : TrackwithHistory pointer for " 
			<< currentID.trackID() << " is " << trkh;
    if (trkh != NULL) {
      etrack = trkh->momentum().mag();
      if (etrack >= energyCut) {
	trkh->save();
	LogDebug("CaloSim") << "CaloSD: set save the track " 
			    << currentID.trackID() << " with Hit";
      }
    }
  }
  primIDSaved = currentID.trackID();
  return aHit;
}	 

void CaloSD::updateHit(CaloG4Hit* aHit) {

  if (edepositEM+edepositHAD != 0) {
    aHit->addEnergyDeposit(edepositEM,edepositHAD);
    LogDebug("CaloSim") << "CaloSD: Add energy deposit in " << currentID 
			<< " em " << edepositEM/MeV << " hadronic " 
			<< edepositHAD/MeV << " MeV"; 
  }

  // buffer for next steps:
  previousID = currentID;
}

G4ThreeVector CaloSD::setToLocal(G4ThreeVector global, 
				 const G4VTouchable* touch) {

  G4ThreeVector localPoint = 
    touch->GetHistory()->GetTopTransform().TransformPoint(global);
  
  return localPoint;  
}

void CaloSD::EndOfEvent(G4HCofThisEvent* ) {

  LogDebug("CaloSim") << "CaloSD: EndofEvent entered with " << theHC->entries()
		      << " entries";
  //  TimeMe("CaloSD:sortAndMergeHits",false);

  hitvec.reserve(theHC->entries());

  // here we loop over transient hits and make them persistent
  for (int ii=0; ii<theHC->entries(); ii++) {
    CaloG4Hit* aHit = (*theHC)[ii];
    LogDebug("CaloSim") << "CaloHit " << ii << " " << *aHit; 
    hitvec.push_back(aHit);
  }

  LogDebug("CaloSim") << "CaloSD: EndofEvent transfer " << hitvec.size() 
		      << " hits to vector" << " for " << GetName();
}

void CaloSD::update(const ::EndOfEvent * ) {
  
  LogDebug("CaloSim") << "CaloSD::update: Start saving hits for " << GetName()
		      << " with " << hitvec.size() << " hits";
  int kount = 0, count = 0, wrong = 0;
  std::vector<CaloG4Hit*>::iterator i;

  if (useMap) {
    for (i=hitvec.begin(); i!=hitvec.end(); i++) {
      if (!saveHit(*i)) wrong++;
      kount++;
    }
    count = kount;
  } else {
    sort(hitvec.begin(), hitvec.end(), CaloG4HitLess());
    LogDebug("CaloSim") << "CaloSD: EndofEvent sort the hits in buffer ";

    std::vector<CaloG4Hit*>::iterator j;
    CaloG4HitEqual equal;
    for (i=hitvec.begin(); i!=hitvec.end(); i++) {
      int jump = 0;
      for (j = i+1; j<hitvec.end() && equal(*i, *j); j++) {
	jump++;
	// merge j to i
	(**i).addEnergyDeposit(**j);
	double em = 0.;
	double eh = 0.;
	(**j).setEM(em);
	(**j).setHadr(eh);
      }

      kount++;
      count++;
      LogDebug("CaloSim") << "CaloSD: Merge " << jump << " hits to hit " 
			  << kount;
      if (!saveHit(*i)) wrong++;
      i+=jump;
      kount += jump;
    }
  }
  
  edm::LogInfo("CaloSim") << "CaloSD: " << GetName() << " store " << count 
			  << " hits out of " << kount << " recorded with " 
			  << wrong << " track IDs not given properly";
  summarize();
}

bool CaloSD::saveHit(CaloG4Hit* aHit) {

  int tkID;
  bool ok = true;
  if (m_trackManager) {
    tkID = m_trackManager->idSavedTrack(aHit->getTrackID());
    if (tkID == 0) ok = false;
  } else {
    tkID = aHit->getTrackID();
    ok = false;
  }
  LogDebug("CaloSim") << "CalosD: Track ID " << aHit->getTrackID() 
		      << " changed to " << tkID << " by SimTrackManager"
		      << " Status " << ok;

  double time = aHit->getTimeSlice();
  if (corrTOFBeam) time += correctT;
  slave->processHits(aHit->getUnitID(), aHit->getEM()/GeV, aHit->getHadr()/GeV,
		     time, tkID);
  LogDebug("CaloSim") << "CaloSD: Store Hit at " << std::hex 
		      << aHit->getUnitID() << std::dec << " due to " << tkID 
		      << " in time " << time << " of energy " 
		      << aHit->getEM()/GeV << " GeV (EM) and " 
		      << aHit->getHadr()/GeV << " GeV (Hadr)";
  return ok;
}

void CaloSD::summarize() {}

void CaloSD::clear() {} 

void CaloSD::DrawAll() {} 

void CaloSD::PrintAll() {
  LogDebug("CaloSim") << "CaloSD: Collection " << theHC->GetName();
  theHC->PrintAllHits();
} 

void CaloSD::update(const BeginOfEvent *) {
  LogDebug("CaloSim")  << "CaloSD: Dispatched BeginOfEvent for " << GetName() 
		       << " !" ;
  clearHits();
  tkMap.erase (tkMap.begin(), tkMap.end());
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
      LogDebug("CaloSim") << "CaloSD: get track " << it << " from "
                          << "Container of size " << trksForThisEvent->size();
      if (it >= 0) {
        TrackWithHistory * trkH = (*trksForThisEvent)[it];
        LogDebug("CaloSim") << " with ID " << trkH->trackID();
        if (trkH->trackID() == (unsigned int)(id)) tkMap[id] = trkH;
      }
    }
  }
}


void CaloSD::clearHits(){

  hitvec.erase (hitvec.begin(), hitvec.end()); 
  hitMap.erase (hitMap.begin(), hitMap.end());
  previousID.reset();
  primIDSaved    = -99;
  LogDebug("CaloSim") << "CaloSD: Clears hit vector for " << GetName() << " " 
		      << slave;
  slave->Initialize();
  LogDebug("CaloSim") << "CaloSD: Initialises slave SD for " << GetName();
}

void CaloSD::fillHits(edm::PCaloHitContainer& c, std::string n){
  if (slave->name() == n) c=slave->hits();
}
