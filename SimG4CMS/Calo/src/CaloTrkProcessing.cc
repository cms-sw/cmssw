#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"


#include "SimG4CMS/Calo/interface/CaloTrkProcessing.h"

#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4EventManager.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include "G4SystemOfUnits.hh"

//#define DebugLog

CaloTrkProcessing::CaloTrkProcessing(G4String name, 
				     const DDCompactView & cpv,
				     SensitiveDetectorCatalog & clg, 
				     edm::ParameterSet const & p,
				     const SimTrackManager* manager) : 
  SensitiveCaloDetector(name, cpv, clg, p), lastTrackID(-1),
  m_trackManager(manager) {  

  //Initialise the parameter set
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("CaloTrkProcessing");
  testBeam   = m_p.getParameter<bool>("TestBeam");
  eMin       = m_p.getParameter<double>("EminTrack")*MeV;
  putHistory = m_p.getParameter<bool>("PutHistory");

  edm::LogInfo("CaloSim") << "CaloTrkProcessing: Initailised with TestBeam = " 
			  << testBeam << " Emin = " << eMin << " MeV and"
			  << " History flag " << putHistory;

  //Get the names 
  G4String attribute = "ReadOutName"; 
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,name,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());

  G4String value                     = "Calorimeter";
  std::vector<std::string> caloNames = getNames (value, sv);
#ifdef DebugLog
  LogDebug("CaloSim") << "CaloTrkProcessing: Names for " << value << ":";
  for (unsigned int i=0; i<caloNames.size(); i++)
    LogDebug("CaloSim") << " (" << i << ") " << caloNames[i];
#endif

  value                                = "Levels";
  std::vector<double>      levels      = getNumbers (value, sv);
#ifdef DebugLog
  LogDebug("CaloSim") << "CaloTrkProcessing: Names for " << value << ":";
  for (unsigned int i=0; i<levels.size(); i++)
    LogDebug("CaloSim") << " (" << i << ") " << levels[i];
#endif

  value                                = "Neighbours";
  std::vector<double>      neighbours  = getNumbers (value, sv);
#ifdef DebugLog
  LogDebug("CaloSim") << "CaloTrkProcessing: Names for " << value << ":";
  for (unsigned int i=0; i<neighbours.size(); i++)
    LogDebug("CaloSim") << " (" << i << ") " << neighbours[i];
#endif

  value                                = "Inside";
  std::vector<std::string> insideNames = getNames (value, sv);
#ifdef DebugLog
  LogDebug("CaloSim") << "CaloTrkProcessing: Names for " << value << ":";
  for (unsigned int i=0; i<insideNames.size(); i++)
    LogDebug("CaloSim") << " (" << i << ") " << insideNames[i];
#endif

  value                                = "InsideLevel";
  std::vector<double>      insideLevel = getNumbers (value, sv);
#ifdef DebugLog
  LogDebug("CaloSim") << "CaloTrkProcessing: Names for " << value << ":";
  for (unsigned int i=0; i<insideLevel.size(); i++)
    LogDebug("CaloSim") << " (" << i << ") " << insideLevel[i];
#endif

  if (caloNames.size() < neighbours.size()) {
    edm::LogError("CaloSim") << "CaloTrkProcessing: # of Calorimeter bins " 
			     << caloNames.size() << " does not match with "
			     << neighbours.size() << " ==> illegal ";
    throw cms::Exception("Unknown", "CaloTrkProcessing")
      << "Calorimeter array size does not match with size of neighbours\n";
  }

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  int istart = 0;
  for (unsigned int i=0; i<caloNames.size(); i++) {
    G4LogicalVolume* lv     = 0;
    G4String         name   = caloNames[i];
    int              number = static_cast<int>(neighbours[i]);
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
      if ((*lvcite)->GetName() == name) {
	lv = (*lvcite);
	break;
      }
    }
    if (lv != 0) {
     CaloTrkProcessing::Detector detector;
     detector.name  = name;
     detector.lv    = lv;
     detector.level = static_cast<int>(levels[i]);
     if (istart+number > (int)(insideNames.size())) {
       edm::LogError("CaloSim") << "CaloTrkProcessing: # of InsideNames bins " 
	 		        << insideNames.size() <<" too few compaerd to "
		 	        << istart+number << " requested ==> illegal ";
       throw cms::Exception("Unknown", "CaloTrkProcessing")
	 << "InsideNames array size does not match with list of neighbours\n";
     }
     std::vector<std::string>      inside;
     std::vector<G4LogicalVolume*> insideLV;
     std::vector<int>              insideLevels;
     for (int k = 0; k < number; k++) {
       lv   = 0;
       name = insideNames[istart+k];
       for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
	 if ((*lvcite)->GetName() == name) {
	   lv = (*lvcite);
	   break;
	 }
       inside.push_back(name);
       insideLV.push_back(lv);
       insideLevels.push_back(static_cast<int>(insideLevel[istart+k]));
     }
     detector.fromDets   = inside;
     detector.fromDetL   = insideLV;
     detector.fromLevels = insideLevels;
     detectors.push_back(detector);
    }
    istart += number;
  }

  edm::LogInfo("CaloSim") << "CaloTrkProcessing: with " << detectors.size()
			  << " calorimetric volumes";
  for (unsigned int i=0; i<detectors.size(); i++) {
    edm::LogInfo("CaloSim") << "CaloTrkProcessing: Calorimeter volume " << i
			    << " " << detectors[i].name << " LV "
			    << detectors[i].lv << " at level "
			    << detectors[i].level << " with "
			    << detectors[i].fromDets.size() << " neighbours";
    for (unsigned int k=0; k<detectors[i].fromDets.size(); k++) 
      edm::LogInfo("CaloSim") << "                   Element " << k << " "
			      << detectors[i].fromDets[k] << " LV "
			      << detectors[i].fromDetL[k] << " at level "
			      << detectors[i].fromLevels[k];
  }
}

CaloTrkProcessing::~CaloTrkProcessing() {
  edm::LogInfo("CaloSim") << "CaloTrkProcessing: Deleted";
}

void CaloTrkProcessing::update(const BeginOfEvent * evt) {
  lastTrackID = -1;
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
#ifdef DebugLog
      LogDebug("CaloSim") << "CaloTrkProcessing set IDonCaloSurface to " << id 
			  << " at step Number "
			  << theTrack->GetCurrentStepNumber();
#endif
      trkInfo->setIDonCaloSurface(id,0,0,
				  theTrack->GetDefinition()->GetPDGEncoding(),
				  theTrack->GetMomentum().mag());
      lastTrackID = id;
      if (theTrack->GetKineticEnergy()/MeV > eMin)
	trkInfo->putInHistory();
    } 
  } else {
    if (putHistory) {
      trkInfo->putInHistory();
      //      trkInfo->setAncestor();
    }
#ifdef DebugLog
    LogDebug("CaloSim") << "CaloTrkProcessing Entered for " << id 
			<< " at stepNumber "<< theTrack->GetCurrentStepNumber()
			<< " IDonCaloSur.. " << trkInfo->getIDonCaloSurface()
			<< " CaloCheck " << trkInfo->caloIDChecked();
#endif
    if (trkInfo->getIDonCaloSurface() != 0) {
      if (trkInfo->caloIDChecked() == false) {
        G4StepPoint*        postStepPoint = aStep->GetPostStepPoint();   
        const G4VTouchable* post_touch    = postStepPoint->GetTouchable();

        if (isItInside(post_touch, trkInfo->getIDCaloVolume(),
		       trkInfo->getIDLastVolume()) > 0) {
          trkInfo->setIDonCaloSurface(0, -1, -1, 0, 0);
        } else {
          trkInfo->setCaloIDChecked(true);
        }
      }
    } else {
      G4StepPoint*        postStepPoint = aStep->GetPostStepPoint();   
      const G4VTouchable* post_touch    = postStepPoint->GetTouchable();
      int                 ical          = isItCalo(post_touch);
      if (ical >= 0) {
	G4StepPoint*        preStepPoint = aStep->GetPreStepPoint(); 
	const G4VTouchable* pre_touch    = preStepPoint->GetTouchable();
	int                 inside       = isItInside(pre_touch, ical, -1);
	if (inside >= 0 ||  (theTrack->GetCurrentStepNumber()==1)) {
	  trkInfo->setIDonCaloSurface(id,ical,inside,
				      theTrack->GetDefinition()->GetPDGEncoding(),
				      theTrack->GetMomentum().mag());
          trkInfo->setCaloIDChecked(true);
	  lastTrackID = id;
	  if (theTrack->GetKineticEnergy()/MeV > eMin) trkInfo->putInHistory();
#ifdef DebugLog
	  LogDebug("CaloSim") <<"CaloTrkProcessing: set ID on Calo " << ical
			      << " surface (Inside " << inside << ") to " 
			      << id << " of a Track with Kinetic Energy " 
			      << theTrack->GetKineticEnergy()/MeV << " MeV";
#endif
	}
      }
    }
  }
}

std::vector<std::string> CaloTrkProcessing::getNames(const G4String str,
						     const DDsvalues_type &sv){

#ifdef DebugLog
  LogDebug("CaloSim") << "CaloTrkProcessing::getNames called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("CaloSim") << value;
#endif
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

std::vector<double> CaloTrkProcessing::getNumbers(const G4String str,
						  const DDsvalues_type &sv) {

#ifdef DebugLog
  LogDebug("CaloSim") << "CaloTrkProcessing::getNumbers called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("CaloSim") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
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

int CaloTrkProcessing::isItCalo(const G4VTouchable* touch) {

  int lastLevel = -1;
  G4LogicalVolume* lv=0;
  for (unsigned int it=0; it < detectors.size(); it++) {
    if (lastLevel != detectors[it].level) {
      lastLevel = detectors[it].level;
      lv        = detLV(touch, lastLevel);
#ifdef DebugLog
      std::string  name1 = "Unknown";
      if (lv != 0) name1 = lv->GetName(); 
      LogDebug("CaloSim") << "CaloTrkProcessing: volume " << name1
			  << " at Level " << lastLevel;
      int levels = detLevels(touch);
      if (levels > 0) {
	G4String name2[20]; int copyno2[20];
	detectorLevel(touch, levels, copyno2, name2);
	for (int i2=0; i2<levels; i2++) 
	  LogDebug("CaloSim") << " " << i2 << " " << name2[i2] << " " 
			      << copyno2[i2];
      }
#endif
    }
    bool ok = (lv   == detectors[it].lv);
    if (ok) return it;
  }
  return -1;
}

int CaloTrkProcessing::isItInside(const G4VTouchable* touch, int idcal,
				  int idin) {
  int lastLevel = -1;
  G4LogicalVolume* lv=0;
  int id1, id2;
  if (idcal < 0) {id1 = 0; id2 = static_cast<int>(detectors.size());}
  else           {id1 = idcal; id2 = id1+1;}
  for (int it1 = id1; it1 < id2; it1++) {
    if (idin < 0) {
      for (unsigned int it2 = 0; it2 < detectors[it1].fromDets.size(); it2++) {
	if (lastLevel != detectors[it1].fromLevels[it2]) {
	  lastLevel = detectors[it1].fromLevels[it2];
	  lv        = detLV(touch,lastLevel);
#ifdef DebugLog
	  std::string  name1 = "Unknown";
	  if (lv != 0) name1 = lv->GetName(); 
	  LogDebug("CaloSim") << "CaloTrkProcessing: volume " << name1
			      << " at Level " << lastLevel;
	  int levels = detLevels(touch);
	  if (levels > 0) {
	    G4String name2[20]; int copyno2[20];
	    detectorLevel(touch, levels, copyno2, name2);
	    for (int i2=0; i2<levels; i2++) 
	      LogDebug("CaloSim") << " " << i2 << " " << name2[i2] << " " 
				  << copyno2[i2];
	  }
#endif
	}
	bool ok = (lv   == detectors[it1].fromDetL[it2]);
	if (ok) return it2;
      }
    } else {
      lastLevel = detectors[it1].fromLevels[idin];
      lv        = detLV(touch,lastLevel);
#ifdef DebugLog
      std::string  name1 = "Unknown";
      if (lv != 0) name1 = lv->GetName(); 
      LogDebug("CaloSim") << "CaloTrkProcessing: volume " << name1
			  << " at Level " << lastLevel;
      int levels = detLevels(touch);
      if (levels > 0) {
	G4String name2[20]; int copyno2[20];
	detectorLevel(touch, levels, copyno2, name2);
	for (int i2=0; i2<levels; i2++) 
	  LogDebug("CaloSim") << " " << i2 << " " << name2[i2] << " " 
			      << copyno2[i2];
      }
#endif
      bool ok = (lv   == detectors[it1].fromDetL[idin]);
      if (ok) return idin;
    }
  }
  return -1;
}

int CaloTrkProcessing::detLevels(const G4VTouchable* touch) const {

  //Return number of levels
  if (touch) 
    return ((touch->GetHistoryDepth())+1);
  else
    return 0;
}

G4LogicalVolume* CaloTrkProcessing::detLV(const G4VTouchable* touch,
					  int currentlevel) const {

  G4LogicalVolume* lv=0;
  if (touch) {
    int level = ((touch->GetHistoryDepth())+1);
    if (level > 0 && level >= currentlevel) {
      int ii = level - currentlevel; 
      lv     = touch->GetVolume(ii)->GetLogicalVolume();
      return lv;
    } 
  }
  return lv;
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
