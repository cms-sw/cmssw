#ifdef messageimpl
//#include "Utilities/Configuration/interface/Architecture.h"

#include "CaloSim/CaloSD/interface/CaloTrkProcessing.h"
#include "CaloSim/CaloSD/interface/CaloMap.h"

#include "Mantis/MantisApplication/interface/EventAction.h"
#include "Mantis/MantisNotification/interface/BeginOfRun.h"
#include "Mantis/MantisNotification/interface/BeginOfEvent.h"
#include "Mantis/MantisNotification/interface/EndOfTrack.h"
#include "Mantis/MantisNotification/interface/TrackWithHistory.h"
#include "Mantis/MantisNotification/interface/TrackInformation.h"
#include "DDD/DDCore/interface/DDFilter.h"
#include "DDD/DDCore/interface/DDFilteredView.h"
#include "DDD/DDCore/interface/DDSolid.h"
#include "DDD/DDCore/interface/DDValue.h"
#include "Utilities/GenUtil/interface/CMSexception.h"
#include "Utilities/UI/interface/SimpleConfigurable.h"

#include "G4EventManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#define ddebug

UserVerbosity CaloTrkProcessing::cout("CaloTrkProcessing","silent","CaloSD");

CaloTrkProcessing::CaloTrkProcessing() : rinCalo(1233), zinCalo(2935),
					 lastTrackID(-1) {  

  Observer<const BeginOfRun *>::init();
  Observer<const BeginOfEvent *>::init();
  Observer<const EndOfTrack *>::init();
  Observer<const G4Step *>::init();

  static SimpleConfigurable<bool> caloTrOn(false,"CaloTrkProcessing:TestBeam");
  testBeam = caloTrOn.value();

  cout.infoOut << "CaloTrkProcessing: Initailised with TestBeam = " << testBeam
	       << endl;

}

void CaloTrkProcessing::upDate(const BeginOfRun * ) {

  std::string attribute = "Volume"; 
  std::string value     = "Calorimeter";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDCompactView cpv;
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();
  bool ok    = false;
  while (dodet) {
    const DDSolid & sol = fv.logicalPart().solid();
    const std::vector<double> & paras = sol.parameters();
#ifdef ddebug
    cout.testOut << "CaloTrkProcessing:: logical volume " 
		 << fv.logicalPart().name() << endl;
    cout.testOut << "CaloTrkProcessing:: solid " << sol.name() << " Shape " 
		 << sol.shape() << " " << ddpolycone_rz << " " 
		 << ddpolycone_rrz << endl;
    for (unsigned int i = 0; i < paras.size(); i++) {
      cout.testOut << "\tElement " << i << " " << paras[i];
      if (i%10 == 5) cout.testOut << endl;
    }
    cout.testOut << endl;
#endif
    if (sol.shape() == ddpolycone_rrz) {
      int nz  = (paras.size()-2)/3;
      int nz2 = (nz-1)/2;
      rinCalo = paras[3*nz2+3];
      zinCalo = paras[3*nz2+5];
      ok = true;
    } else if (sol.shape() == ddtubs) {
      rinCalo = paras[1];
      zinCalo = paras[0];
      ok = true;
    }
    dodet = fv.nextSibling();
  }

  cout.infoOut << "CaloTrkProcessing: Flag " << ok << " for loading geometry"
	       << " specs (" << rinCalo << " "  << zinCalo << " for inner r/z"
	       << " extent of Calo)" << endl;
}

void CaloTrkProcessing::upDate(const BeginOfEvent * evt) {

  CaloMap::instance()->clear((*evt)()->GetEventID());
  lastTrackID = -1;
}

void CaloTrkProcessing::upDate(const EndOfTrack * trk) {

  const G4Track* theTrack = (*trk)(); // recover G4 pointer if wanted
  int id = theTrack->GetTrackID();
  if (id == lastTrackID) {
    EventAction * eventAction = (EventAction *)(G4EventManager::GetEventManager()->GetUserEventAction());
    TrackContainer * trksForThisEvent = eventAction->trackContainer();
    if (trksForThisEvent != NULL) {
      int it = (int)(trksForThisEvent->size()) - 1;
#ifdef ddebug
      cout.testOut << "CaloTrkProcessing: get track " << it << " from "
		   << "Container of size " << trksForThisEvent->size();
#endif
      if (it >= 0) {
	TrackWithHistory * trkH = (*trksForThisEvent)[it];
#ifdef ddebug
	cout.testOut << " with ID " << trkH->trackID() << endl;
#endif
	if (trkH->trackID() == (unsigned int)(id))
	  CaloMap::instance()->setTrack(id, trkH);
      } else {
#ifdef ddebug
	cout.testOut << endl;
#endif
      }
    }
  }
}

void CaloTrkProcessing::upDate(const G4Step * aStep) {
  
  // define if you are at the surface of CALO  
  
  G4Track* theTrack = aStep->GetTrack();   

  TrackInformation* trkInfo = dynamic_cast<TrackInformation*>
    (theTrack->GetUserInformation());
  
  if (trkInfo == 0) {
    cout.infoOut << "CaloTrkProcessing: No trk info !!!! abort " << endl;
    throw Genexception("CaloTrkProcessing: cannot get trkInfo");
  } 

  if (testBeam) {
    if (trkInfo->getIDonCaloSurface() == 0) {
      int id = theTrack->GetTrackID();
#ifdef ddebug
      cout.debugOut << "CaloTrkProcessing set IDonCaloSurface to " << id 
		    << " at stepNumber " << theTrack->GetCurrentStepNumber() 
		    << endl;
#endif
      trkInfo->setIDonCaloSurface(id);
      lastTrackID = id;
      if (theTrack->GetKineticEnergy()/MeV > 0.01)
	trkInfo->putInHistory();
    } 
  } else {
    int id = theTrack->GetTrackID();
#ifdef ddebug
    cout.debugOut << "CaloTrkProcessing Entered for " << id 
		  << " at stepNumber " << theTrack->GetCurrentStepNumber() 
		  << " IDonCaloSur.. " << trkInfo->getIDonCaloSurface()
		  << " CaloCheck " << trkInfo->caloIDChecked() << endl;
#endif
    if (trkInfo->getIDonCaloSurface() != 0) {
      if (trkInfo->caloIDChecked() == false) {
	const G4ThreeVector pos = theTrack->GetPosition();
	if (pos.perp() < rinCalo && abs(pos.z()) < zinCalo) {
	  trkInfo->setIDonCaloSurface(0);
	} else {
	  trkInfo->setCaloIDChecked(true);
	}
      }
    }

    if (trkInfo->getIDonCaloSurface() == 0) {
      G4StepPoint*        preStepPoint = aStep->GetPreStepPoint(); 
      const G4VTouchable* pre_touch    = preStepPoint->GetTouchable();
      int                 pre_levels   = detLevels(pre_touch);
      G4String            pre_name     = detName(pre_touch, pre_levels, 3);
    
#ifdef ddebug
      cout.debugOut << "CaloTrkProcessing: Previous volume with " << pre_levels
		    << " levels at Level 3 " << pre_name << endl;
      if (pre_levels > 0) {
	G4String name1[20]; int copyno1[20];
	detectorLevel(pre_touch, pre_levels, copyno1, name1);
	for (int i1=0; i1<pre_levels; i1++) 
	  cout.debugOut << " " << i1 << " " << name1[i1] << " " << copyno1[i1];
	cout.debugOut << endl;
      }
#endif

      G4StepPoint*        postStepPoint = aStep->GetPostStepPoint();   
      const G4VTouchable* post_touch    = postStepPoint->GetTouchable();
      int                 post_levels   = detLevels(post_touch);
      if (post_levels == 0) return;
      G4String            post_name     = detName(post_touch, post_levels, 3);

#ifdef ddebug
      cout.debugOut << "CaloTrkProcessing: Post volume with " << post_levels 
		   << " levels at Level 3 " << post_name << endl;
      if (post_levels > 0) {
	G4String name2[20]; int copyno2[20];
	detectorLevel(post_touch, post_levels, copyno2, name2);
	for (int i2=0; i2<post_levels; i2++) 
	  cout.debugOut << " " << i2 << " " << name2[i2] << " " << copyno2[i2];
	cout.debugOut << endl;
      }
#endif
      
      if (post_name == "CALO" && 
	  (pre_name == "TRAK" || (theTrack->GetCurrentStepNumber()==1))) {
	trkInfo->setIDonCaloSurface(id);
	lastTrackID = id;
	if (theTrack->GetKineticEnergy()/MeV > 0.01)
	  trkInfo->putInHistory();
#ifdef ddebug
	cout.debugOut << "CaloTrkProcessing: set ID on Calo surface to " << id 
		      << " of a Track with Kinetic Energy " 
		      << theTrack->GetKineticEnergy()/MeV << " MeV" << endl;
#endif
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

  //Get name and copy numbers
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      if (pv != 0) 
	name[ii] = pv->GetName();
      else
	name[ii] = "Unknown";
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}


#include "Utilities/GenUtil/interface/PackageInitializer.h"
#include "Utilities/UI/interface/PackageBuilderUI.h"

static PKBuilder<CaloTrkProcessing>   	observeCaloStep("CaloTrkProcessing");

#endif
