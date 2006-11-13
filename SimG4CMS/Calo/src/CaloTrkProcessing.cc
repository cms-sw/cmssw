#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "SimG4CMS/Calo/interface/CaloTrkProcessing.h"
#include "SimG4CMS/Calo/interface/CaloMap.h"

#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4EventManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"

CaloTrkProcessing::CaloTrkProcessing(G4String name, 
				     const DDCompactView & cpv,
				     edm::ParameterSet const & p,
				     const SimTrackManager* manager) : 
  SensitiveCaloDetector(name, cpv, p), lastTrackID(-1),
  m_trackManager(manager) {  

  //Initialise the parameter set
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("CaloTrkProcessing");
  testBeam  = m_p.getParameter<bool>("TestBeam");
  eMin      = m_p.getParameter<double>("EminTrack")*MeV;

  edm::LogInfo("CaloSim") << "CaloTrkProcessing: Initailised with TestBeam = " 
			  << testBeam << " Emin = " << eMin << " MeV";

  //Get the names 
  G4String attribute = "ReadOutName"; 
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,name,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());

  G4String value     = "Calorimeter";
  caloNames          = getNames (value, sv);
  edm::LogInfo("CaloSim") << "CaloTrkProcessing: Names for " << value << ":";
  for (unsigned int i=0; i<caloNames.size(); i++)
    edm::LogInfo("CaloSim") << " (" << i << ") " << caloNames[i];

  value              = "Inside";
  insideNames        = getNames (value, sv);
  edm::LogInfo("CaloSim") << "CaloTrkProcessing: Names for " << value << ":";
  for (unsigned int i=0; i<insideNames.size(); i++)
    edm::LogInfo("CaloSim") << " (" << i << ") " << insideNames[i];
}

CaloTrkProcessing::~CaloTrkProcessing() {
  edm::LogInfo("CaloSim") << "CaloTrkProcessing: Deleted";
}

void CaloTrkProcessing::update(const BeginOfEvent * evt) {

  CaloMap::instance()->clear((*evt)()->GetEventID());
  lastTrackID = -1;
}

void CaloTrkProcessing::update(const EndOfTrack * trk) {

  int id = (*trk)()->GetTrackID();
  if (id == lastTrackID) {
    const TrackContainer * trksForThisEvent = m_trackManager->trackContainer();
    if (trksForThisEvent != NULL) {
      int it = (int)(trksForThisEvent->size()) - 1;
      LogDebug("CaloSim") << "CaloTrkProcessing: get track " << it << " from "
			  << "Container of size " << trksForThisEvent->size();
      if (it >= 0) {
	TrackWithHistory * trkH = (*trksForThisEvent)[it];
	LogDebug("CaloSim") << " with ID " << trkH->trackID();
	if (trkH->trackID() == (unsigned int)(id))
	  CaloMap::instance()->setTrack(id, trkH);
      } 
    }
  }
}

void CaloTrkProcessing::update(const G4Step * aStep) {
  
  // define if you are at the surface of CALO  
  
  G4Track* theTrack = aStep->GetTrack();   
  int      id       = theTrack->GetTrackID();

  TrackInformation* trkInfo = dynamic_cast<TrackInformation*>
    (theTrack->GetUserInformation());
  
  if (trkInfo == 0) {
    edm::LogError("CaloSim") << "CaloTrkProcessing: No trk info !!!! abort ";
    throw cms::Exception("Unknown", "CaloTrkProcessing")
      << "cannot get trkInfo for Track " << id << "\n";
  }
  
  if (testBeam) {
    if (trkInfo->getIDonCaloSurface() == 0) {
      LogDebug("CaloSim") << "CaloTrkProcessing set IDonCaloSurface to " << id 
			  << " at stepNumber " << theTrack->GetCurrentStepNumber();
      trkInfo->setIDonCaloSurface(id);
      lastTrackID = id;
      if (theTrack->GetKineticEnergy()/MeV > eMin)
	trkInfo->putInHistory();
    } 
  } else {
    LogDebug("CaloSim") << "CaloTrkProcessing Entered for " << id 
			<< " at stepNumber "<< theTrack->GetCurrentStepNumber()
			<< " IDonCaloSur.. " << trkInfo->getIDonCaloSurface()
			<< " CaloCheck " << trkInfo->caloIDChecked();
    if (trkInfo->getIDonCaloSurface() != 0) {
      if (trkInfo->caloIDChecked() == false) {
        G4StepPoint*        postStepPoint = aStep->GetPostStepPoint();   
        const G4VTouchable* post_touch    = postStepPoint->GetTouchable();
        int                 post_levels   = detLevels(post_touch);
        std::string         post_name     = "NotFound";
        if (post_levels > 1)post_name     = detName(post_touch,post_levels,3);
        LogDebug("CaloSim") << "CaloTrkProcessing: Post volume with " 
			    << post_levels << " levels at Level 3 " 
			    << post_name;
        if (isItInside(post_name)) {
          trkInfo->setIDonCaloSurface(0);
        } else {
          trkInfo->setCaloIDChecked(true);
        }
      }
    } else {

      G4StepPoint*        preStepPoint = aStep->GetPreStepPoint(); 
      const G4VTouchable* pre_touch    = preStepPoint->GetTouchable();
      int                 pre_levels   = detLevels(pre_touch);
      std::string         pre_name     = detName(pre_touch, pre_levels, 3);
    
      LogDebug("CaloSim") << "CaloTrkProcessing: Previous volume with " 
			  << pre_levels;
      if (pre_levels > 0) {
	G4String name1[20]; int copyno1[20];
	detectorLevel(pre_touch, pre_levels, copyno1, name1);
	for (int i1=0; i1<pre_levels; i1++) 
	  LogDebug("CaloSim") << " " << i1 << " " << name1[i1] << " " 
			      << copyno1[i1];
      }

      G4StepPoint*        postStepPoint = aStep->GetPostStepPoint();   
      const G4VTouchable* post_touch    = postStepPoint->GetTouchable();
      int                 post_levels   = detLevels(post_touch);
      if (post_levels == 0) return;
      std::string         post_name     = detName(post_touch, post_levels, 3);
      
      LogDebug("CaloSim") << "CaloTrkProcessing: Post volume with " 
			  << post_levels << " levels at Level 3 " << post_name;
      if (post_levels > 0) {
        G4String name2[20]; int copyno2[20];
        detectorLevel(post_touch, post_levels, copyno2, name2);
        for (int i2=0; i2<post_levels; i2++) 
          LogDebug("CaloSim") << " " << i2 << " " << name2[i2] << " " 
			      << copyno2[i2];
      }
      
      if (isItCalo(post_name) && (isItInside(pre_name) || 
                                  (theTrack->GetCurrentStepNumber()==1))) {
        trkInfo->setIDonCaloSurface(id);
        lastTrackID = id;
        if (theTrack->GetKineticEnergy()/MeV > eMin)
          trkInfo->putInHistory();
        LogDebug("CaloSim") << "CaloTrkProcessing: set ID on Calo surface to " 
			    << id << " of a Track with Kinetic Energy " 
			    << theTrack->GetKineticEnergy()/MeV << " MeV";
      }
    }
  }
}

int CaloTrkProcessing::detLevels(const G4VTouchable* touch) const {

  //Return number of levels
  if (touch) 
    return ((touch->GetHistoryDepth())+1);
  else
    return 0;
}

std::vector<std::string> CaloTrkProcessing::getNames(const G4String str,
						     const DDsvalues_type &sv){

  LogDebug("CaloSim") << "CaloTrkProcessing::getNames called for " << str;

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    LogDebug("CaloSim") << value;
    const std::vector<std::string> & fvec = value.strings();
    int nval = fvec.size();
    if (nval < 1) {

	edm::LogError("CaloSim") << "CaloTrkProcessing: # of " << str 
				 << " bins " << nval << " < 1 ==> illegal ";
	throw cms::Exception("Unknown", "CaloTrkProcessing")
	  << "nval < 2 for array " << str << "\n";
    }
    
    return fvec;
  } else {
    edm::LogError("CaloSim") << "CaloTrkProcessing: cannot get array " << str ;
    throw cms::Exception("Unknown", "CaloTrkProcessing")
      << "cannot get array " << str << "\n";
  }
}

bool CaloTrkProcessing::isItCalo(std::string name) {

  std::vector<std::string>::const_iterator it = caloNames.begin();
  for (; it != caloNames.end(); it++) 
    if (name == *it) return true;
  return false;
}

bool CaloTrkProcessing::isItInside(std::string name) {

  std::vector<std::string>::const_iterator it = insideNames.begin();
  for (; it != insideNames.end(); it++) 
    if (name == *it) return true;
  return false;
}

G4String CaloTrkProcessing::detName(const G4VTouchable* touch, int level,
				    int currentlevel) const {

  //Go down to current level
  if (level > 0 && level >= currentlevel) {
    int ii = level - currentlevel; 
    return touch->GetVolume(ii)->GetName();
  } else {
    return "NotFound";
  }
}

void CaloTrkProcessing::detectorLevel(const G4VTouchable* touch, int& level,
				      int* copyno, G4String* name) const {

  static const std::string unknown("Unknown");
  //Get name and copy numbers
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      if (pv != 0) 
	name[ii] = pv->GetName();
      else
	name[ii] = unknown;
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}
