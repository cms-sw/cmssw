///////////////////////////////////////////////////////////////////////////////
// File: CaloSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/CaloMap.h"
#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"
#include "SimG4Core/Geometry/interface/SDCatalog.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Application/interface/EventAction.h"

#include "G4EventManager.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#ifdef G4v7
#include "G4GFlashSpot.hh"
#endif

#define debug

CaloSD::CaloSD(G4String name, const DDCompactView & cpv,
	       edm::ParameterSet const & p, const SimTrackManager* manager) : 
  SensitiveCaloDetector(name, cpv, p),
#ifdef G4v7
  G4VGFlashSensitiveDetector(), 
#endif
  theTrack(0), preStepPoint(0), m_trackManager(manager), hcID(-1), theHC(0), 
  currentHit(0) {
  //Add Hcal Sentitive Detector Names

  //   Observer<const BeginOfEvent *>::init();
  collectionName.insert(name);

  //Parameters
  //  static SimpleConfigurable<float> pEmin(1.0,"CaloSD:EminTrack");
  //   static SimpleConfigurable<int>   pcheckHit(25,"CaloSD:CheckHits");
  //   static SimpleConfigurable<bool>  pUseMap(false,"CaloSD:UseMap");
  edm::ParameterSet m_CaloSD = p.getParameter<edm::ParameterSet>("CaloSD");
  energyCut = m_CaloSD.getParameter<double>("EminTrack")*GeV;
  checkHits = m_CaloSD.getParameter<int>("CheckHits");
  useMap    = m_CaloSD.getParameter<bool>("UseMap");
  int verbn = m_CaloSD.getParameter<int>("Verbosity");
  bool on   = m_CaloSD.getParameter<bool>("DetailedTiming");

  SetVerboseLevel(verbn);
  if (verboseLevel > 0) 
    std::cout << "***************************************************" 
	      << std::endl
	      << "*                                                 *" 
	      << std::endl
	      << "* Constructing a CaloSD  with name " << GetName()
	      << std::endl
	      << "*                                                 *" 
	      << std::endl
	      << "***************************************************" 
	      << std::endl;

  slave  = new CaloSlaveSD(name);

  //
  // Now attach the right detectors (LogicalVolumes) to me
  //
  vector<string> lvNames = 
    SensitiveDetectorCatalog::instance()->logicalNames(name);
  this->Register();
  for (vector<string>::iterator it=lvNames.begin();  it !=lvNames.end(); it++){
    this->AssignSD(*it);
#ifdef debug
    if (verboseLevel > 0) 
      std::cout << "CaloSD : Assigns SD to LV " << (*it) << std::endl;
#endif
  }

  // timer initialization
  if (on) {
  //     string trname("CaloSD:");
  //     theHitTimer.init( trname + name + ":hits", true);
  //   }
  //   else {
  //     theHitTimer.init( "CaloSensitiveDetector:hits", true);
  }

  if (verboseLevel > 0) 
    std::cout << "CaloSD: Minimum energy of track for saving it " 
	      << energyCut/GeV  << " GeV" << std::endl 
	      << "        Use of HitID Map " << useMap << std::endl
	      << "        Check last " << checkHits 
	      << " before saving the hit" << std::endl;
}

CaloSD::~CaloSD() { 
  if (slave)           delete slave; 
}

bool CaloSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {
  //  TimeMe t1( theHitTimer, false);
  
  if (aStep == NULL) {
    return true;
  } else {
    getStepInfo(aStep);
    if (hitExists() == false && edepositEM+edepositHAD>0.) 
      createNewHit();
  }
  return true;
} 

#ifdef G4v7
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
#ifdef debug
	if (verboseLevel > 2) 
	  std::cout << "CaloSD: Spot Problem with primaryID **** set by force"
		    << " to  **** " << theTrack->GetTrackID() << std::endl;
#endif
	primaryID = theTrack->GetTrackID();
      }
      currentID.setID(unitID, time, primaryID);
#ifdef debug
      if (verboseLevel > 2) 
	std::cout << "CaloSD:: GetSpotInfo for"
		  << " Unit 0x" << std::hex << currentID.unitID() << std::dec
		  << " Edeposit = " << edepositEM << " " << edepositHAD 
		  << std::endl;
#endif
      // Update if in the same detector, time-slice and for same track   
      if (currentID == previousID) {
	updateHit();
      } else {
	
	posGlobal = aSpot->GetPosition();
	// Reset entry point for new primary
	if (currentID.trackID() != previousID.trackID()) {
	  entrancePoint  = aSpot->GetPosition();
	  entranceLocal  = aSpot->GetTouchableHandle()->GetHistory()->
	    GetTopTransform().TransformPoint(entrancePoint);
	  incidentEnergy = theTrack->GetKineticEnergy();
#ifdef debug
	  if (verboseLevel > 2) 
	    std::cout << "CaloSD: Incident energy " << incidentEnergy/GeV 
		      << " GeV and" << " entrance point " << entrancePoint 
		      << " (Global) " << entranceLocal << " (Local)" 
		      << std::endl;
#endif
	}
	
	if (checkHit() == false) createNewHit();
      }
      
      delete 	fFakeStep;
    }
    return true;
    
  } 
  return false;
}                                   
#endif

double CaloSD::getEnergyDeposit(G4Step* aStep) {
  return aStep->GetTotalEnergyDeposit();
}

void CaloSD::Initialize(G4HCofThisEvent * HCE) { 

#ifdef debug
  if (verboseLevel > 1) 
    std::cout << "CaloSD : Initialize called for " << GetName() << std::endl;
#endif

  //This initialization is performed at the beginning of an event
  //------------------------------------------------------------
  theHC = new CaloG4HitCollection(GetName(), collectionName[0]);
  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);
}

void CaloSD::getStepInfo(G4Step* aStep) {
  
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
#ifdef debug
    if (verboseLevel > 2) 
      std::cout << "CaloSD: Problem with primaryID **** set by force to TkID"
		<< " **** " << theTrack->GetTrackID() << std::endl;
#endif
    primaryID = theTrack->GetTrackID();
  }

  currentID.setID(unitID, time, primaryID);

#ifdef debug
  if (verboseLevel > 2) {
    G4TouchableHistory* touch =(G4TouchableHistory*)(theTrack->GetTouchable());
    std::cout << "CaloSD:: GetStepInfo for"
	      << " PV "     << touch->GetVolume(0)->GetName()
	      << " PVid = " << touch->GetReplicaNumber(0)
	      << " MVid = " << touch->GetReplicaNumber(1)
	      << " Unit 0x" << std::hex << currentID.unitID() << std::dec
	      << " Edeposit = " << edepositEM << " " << edepositHAD 
	      << std::endl;
  }
#endif
}

G4bool CaloSD::hitExists() {
   
  if (currentID.trackID()<1) {
    if (verboseLevel > 0) 
      std::cout << "***** CaloSD error: primaryID = " << currentID.trackID()
		<< " maybe detector name changed" << std::endl;
  }
   
  // Update if in the same detector, time-slice and for same track   
  if (currentID == previousID) {
    updateHit();
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
  bool found = false;
  if (useMap) {
    map<CaloHitID,CaloG4Hit*>::const_iterator it = hitMap.find(currentID);
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
    updateHit();
    return true;
  } else {
    return false;
  }    

}

void CaloSD::resetForNewPrimary(G4ThreeVector point, double energy) {
  
  entrancePoint  = point;
  entranceLocal  = setToLocal(entrancePoint, preStepPoint->GetTouchable());
  incidentEnergy = energy;
#ifdef debug
  if (verboseLevel > 2) 
    std::cout << "CaloSD: Incident energy " << incidentEnergy/GeV 
	      << " GeV and" << " entrance point " << entrancePoint 
	      << " (Global) " << entranceLocal << " (Local)" << std::endl;
#endif
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
    if (abs(charge) >= 2.) rkb *= 7.2/12.6; // based on alpha particle data
    weight = 1./(1.+rkb*dedx+c*dedx*dedx);
#ifdef debug
    if (verboseLevel > 2) 
      std::cout << "CaloSD::getAttenuation in " << mat->GetName() 
		<< " Charge " << charge << " dE/dx " << dedx 
		<< " Birk Const " << rkb << ", " << c << " Weight = " 
		<< weight << " dE " << aStep->GetTotalEnergyDeposit() 
		<< std::endl;
#endif
  }
  return weight;
}

void CaloSD::storeHit(CaloG4Hit* hit) {

  if (previousID.trackID()<0) return;
  if (hit == 0) {
    if (verboseLevel > 0) 
      std::cout << "CaloSD: hit to be stored is NULL !!" << std::endl;
    return;
  }

  theHC->insert(hit);
  if (useMap) hitMap.insert(pair<CaloHitID,CaloG4Hit*>(previousID,hit));

}

void CaloSD::createNewHit() {

#ifdef debug
  if (verboseLevel > 1) {
    std::cout << "CaloSD::CreateNewHit for"
	      << " Unit 0x" << std::hex << currentID.unitID() << std::dec
	      << " Edeposit = " << edepositEM << " " << edepositHAD
	      << std::endl;
    std::cout << " primary "    << currentID.trackID()
	      << " time slice " << currentID.timeSliceID()
	      << " For Track  " << theTrack->GetTrackID()
	      << " which is a " <<theTrack->GetDefinition()->GetParticleName()
	      << " of energy "  << theTrack->GetKineticEnergy()/GeV
	      << " " << theTrack->GetMomentum().mag()/GeV
	      << " daughter of part. " << theTrack->GetParentID()
	      << " and created by " ;
  
    if (theTrack->GetCreatorProcess()!=NULL)
      std::cout << theTrack->GetCreatorProcess()->GetProcessName() ;
    else 
      std::cout << "NO process";
    std::cout << std::endl;
  }
#endif          
    
  currentHit = new CaloG4Hit;
  currentHit->setID(currentID);
  currentHit->setEntry(entrancePoint);
  currentHit->setEntryLocal(entranceLocal);
  currentHit->setPosition(posGlobal);
  currentHit->setIncidentEnergy(incidentEnergy);
  updateHit();
  
  storeHit(currentHit);
  double etrack = 0;
  if (currentID.trackID() == primIDSaved) { // The track is saved; nothing to be done
  } else if (currentID.trackID() == theTrack->GetTrackID()) {
    etrack= theTrack->GetKineticEnergy();
    if (etrack >= energyCut) {
      TrackInformation * trkInfo = 
	(TrackInformation *)(theTrack->GetUserInformation());
      trkInfo->storeTrack(true);
      trkInfo->putInHistory();
#ifdef debug
      if (verboseLevel > 1) 
	std::cout << "CaloSD: set save the track " << currentID.trackID() 
		  << " with Hit" << std::endl;
#endif
    }
  } else {
    TrackWithHistory * trkh = 
      CaloMap::instance()->getTrack(currentID.trackID());
#ifdef debug
    if (verboseLevel > 2) 
      cout << "CaloSD : TrackwithHistory pointer for " 
	   << currentID.trackID() << " is " << trkh << std::endl;
#endif
    if (trkh != NULL) {
      etrack = trkh->momentum().mag();
      if (etrack >= energyCut) {
	trkh->save();
#ifdef debug
	if (verboseLevel > 1) 
	  std::cout << "CaloSD: set save the track " << currentID.trackID()
		    << " with Hit" << std::endl;
#endif
      }
    }
  }
  primIDSaved = currentID.trackID();

}	 

void CaloSD::updateHit() {

  if (edepositEM+edepositHAD != 0) {
    currentHit->addEnergyDeposit(edepositEM,edepositHAD);
#ifdef debug
    if (verboseLevel > 2) 
      cout << "CaloSD: Add energy deposit in " << currentID << " em " 
	   << edepositEM/MeV << " hadronic " << edepositHAD/MeV 
	   << " MeV" << std::endl;
#endif
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

#ifdef debug
  if (verboseLevel > 1) 
    std::cout << "CaloSD: EndofEvent entered with " << theHC->entries() 
	      << " entries" << std::endl;
#endif
  //  TimeMe("CaloSD:sortAndMergeHits",false);

  hitvec.reserve(theHC->entries());

  // here we loop over transient hits and make them persistent
  for (int ii=0; ii<theHC->entries(); ii++) {
    CaloG4Hit* aHit = (*theHC)[ii];
#ifdef debug
    if (verboseLevel > 2) 
      std::cout << "CaloHit " << ii << " " << *aHit << std::endl;
#endif
    hitvec.push_back(aHit);
  }

#ifdef debug
  if (verboseLevel > 1) 
    std::cout << "CaloSD: EndofEvent transfer " << hitvec.size() 
	      << " hits to vector" << " for " << GetName() << std::endl;
#endif
  
  int kount = 0, count = 0, wrong = 0;
  vector<CaloG4Hit*>::iterator i;

  if (useMap) {
    for (i=hitvec.begin(); i!=hitvec.end(); i++) {
      if (!saveHit(*i)) wrong++;
      kount++;
    }
    count = kount;
  } else {
    sort(hitvec.begin(), hitvec.end(), CaloG4HitLess());
#ifdef debug_verbose
    if (verboseLevel > 1) 
      std::cout << "CaloSD: EndofEvent sort the hits in buffer " << std::endl;
#endif

    vector<CaloG4Hit*>::iterator j;
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
#ifdef debug
      if (verboseLevel > 2) 
	cout << "CaloSD: Merge " << jump << " hits to hit " << kount 
	     << std::endl;
#endif
      if (!saveHit(*i)) wrong++;
      i+=jump;
      kount += jump;
    }
  }
  
  if (verboseLevel > 0) 
    std::cout << "CaloSD: " << GetName() << " store " << count 
	      << " hits out of " << kount << " recorded with " << wrong 
	      << " track IDs not given properly" << std::endl;
  summarize();
}

bool CaloSD::saveHit(CaloG4Hit* aHit) {

  int tkID;
  bool ok = true;
  if (m_trackManager) {
    tkID = m_trackManager->g4ToSim(aHit->getTrackID());
    if (tkID == EventAction::InvalidID) ok = false;
  } else {
    tkID = aHit->getTrackID();
    ok = false;
  }
#ifdef debug
  if (verboseLevel > 1) 
    std::cout << "CalosD: Track ID " << aHit->getTrackID() << " changed to "
	      << tkID << " by SimTrackManager" << std::endl;
#endif
  slave->processHits(aHit->getUnitID(), aHit->getEnergyDeposit()/GeV,
		     aHit->getTimeSlice(), tkID);
#ifdef debug
  if (verboseLevel > 2) 
    std::cout << "CaloSD: Store Hit at 0x" << std::hex << aHit->getUnitID() 
	      << std::dec << " due to " << tkID << " in time " 
	      << aHit->getTimeSlice() << " of energy " 
	      << aHit->getEnergyDeposit()/GeV << " GeV" << std::endl;
#endif
  return ok;
}

void CaloSD::summarize() {}

void CaloSD::clear() {} 

void CaloSD::DrawAll() {} 

void CaloSD::PrintAll() {
  if (verboseLevel > 0) 
    std::cout << "CaloSD: Collection " << theHC->GetName() << std::endl;
  theHC->PrintAllHits();
} 

void CaloSD::update(const BeginOfEvent *) {
#ifdef debug
  if (verboseLevel > 1) 
    std::cout << "CaloSD: Dispatched BeginOfEvent for " << GetName() << " !" 
	      << std::endl;
#endif
  clearHits();
}

void CaloSD::clearHits(){

  hitvec.erase (hitvec.begin(), hitvec.end()); 
  hitMap.erase (hitMap.begin(), hitMap.end());
  previousID.reset();
  primIDSaved    = -99;
#ifdef debug
  if (verboseLevel > 1) 
    std::cout << "CaloSD: Clears hit vector for " << GetName() << " " << slave 
	      << std::endl;
#endif
  slave->Initialize();
#ifdef debug_verbose
  if (verboseLevel > 1) 
    std::cout << "CaloSD: Initialises slave SD for " << GetName() << std::endl;
#endif
}

void CaloSD::fillHits(edm::PCaloHitContainer& c, std::string n){
  if (slave->name() == n) c=slave->hits();
}
