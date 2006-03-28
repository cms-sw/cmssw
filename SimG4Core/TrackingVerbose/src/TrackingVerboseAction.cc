///////////////////////////////////////////////////////////////////////////////
// File: TrackingVerboseAction.cc
// Creation: P.Arce  09/01
// Modifications: porting to CMSSW by M. Stavrianakou 22/03/06
// Description:
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/TrackingVerbose/interface/TrackingVerboseAction.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Application/interface/TrackingAction.h"


#include "G4Track.hh"
#include "G4Event.hh"
#include "G4ios.hh"
#include "G4TrackingManager.hh"
#include "G4EventManager.hh"

// #define DEBUG 

TrackingVerboseAction::TrackingVerboseAction(edm::ParameterSet const & p) 
{
    //----- Set which events are verbose
    fTVEventMin  = p.getUntrackedParameter<int>("EventMin",0);
    fTVEventMax  = p.getUntrackedParameter<int>("EventMax",int(1E10));
    fTVEventStep = p.getUntrackedParameter<int>("EventStep",1);

    //----- Set which tracks of those events are verbose
    fTVTrackMin  = p.getUntrackedParameter<int>("TrackMin",0);
    fTVTrackMax  = p.getUntrackedParameter<int>("TrackMax",int(1E10));
    fTVTrackStep = p.getUntrackedParameter<int>("TrackStep",1);

    //----- Set the verbosity level
    fVerboseLevel = p.getUntrackedParameter<int>("VerboseLevel",1);

#ifdef DEBUG
    std::cout << "TV: fTVTrackMin " << fTVTrackMin   << " fTVTrackMax "  <<  fTVTrackMax 
	      <<  " fTVTrackStep "  << fTVTrackStep  << " fTVEventMin "  << fTVEventMin 
	      << " fTVEventMax "    << fTVEventMax   << " fTVEventStep " << fTVEventStep 
	      << " fVerboseLevel "  << fVerboseLevel << std::endl;
#endif

    //----- Set verbosity off to start
    fTrackingVerboseON = 0;
    fTkVerbThisEventON = 0;

    theTrackingManager = 0;
}

TrackingVerboseAction::~TrackingVerboseAction() {}

void TrackingVerboseAction::update(const BeginOfRun * run)
{
    TrackingAction * ta = 
	dynamic_cast<TrackingAction*>(G4EventManager::GetEventManager()->GetUserTrackingAction());
    theTrackingManager = ta->getTrackManager();
}

void TrackingVerboseAction::update(const BeginOfEvent * evt)
{
    const G4Event * anEvent = (*evt)();
    //----------- Set /tracking/verbose for this event 
    int eventNo = anEvent->GetEventID();
#ifdef DEBUG
    std::cout << "TV: trackID: NEW EVENT " << eventNo << std::endl;
#endif

    fTkVerbThisEventON = false;
    //----- Check if event is in the selected range
    bool trackingVerboseThisEvent = false;
    if (eventNo >= fTVEventMin && eventNo <= fTVEventMax) 
    {
	if ((eventNo-fTVEventMin) % fTVEventStep == 0) fTkVerbThisEventON = true;
    }

#ifdef DEBUG
    std::cout << " TV: fTkVerbThisEventON " <<  fTkVerbThisEventON 
	      << " fTrackingVerboseON " << fTrackingVerboseON 
	      << " fTVEventMin " << fTVEventMin << " fTVEventMax " << fTVEventMax << std::endl;
#endif
    //----- check if verbosity has to be changed
    if (fTkVerbThisEventON && !fTrackingVerboseON) 
    {
	SetTrackingVerbose(fVerboseLevel);
	fTrackingVerboseON = 1;
#ifdef DEBUG
	std::cout << "TV: VERBOSEet1 " << eventNo << std::endl;
#endif
    } 
    else if (trackingVerboseThisEvent == 0 && fTrackingVerboseON == 1) 
    {
	SetTrackingVerbose(0);
	fTrackingVerboseON = 0;
#ifdef DEBUG
	std::cout << "TV: VERBOSEet0 " << eventNo << std::endl;
#endif
    }

}

void TrackingVerboseAction::update(const BeginOfTrack * trk)
{
    const G4Track * aTrack = (*trk)();

    //---------- Set /tracking/verbose
    //----- track is verbose only if event is verbose
    if (fTkVerbThisEventON) 
    {
	int trackNo = aTrack->GetTrackID();    
	bool trackingVerboseThisTrack = false;
	//----- Check if track is in the selected range
	if (trackNo >= fTVTrackMin && trackNo <= fTVTrackMax) 
	{
	    if ((trackNo-fTVTrackMin) % fTVTrackStep == 0) trackingVerboseThisTrack = true;
	}
    
	//----- Set the /tracking/verbose for this track 
    if (trackingVerboseThisTrack == 1 && fTrackingVerboseON == 0) 
    {
	SetTrackingVerbose(fVerboseLevel);
	fTrackingVerboseON = 1;
#ifdef DEBUG
	std::cout << "TV: VERBOSEtt1 " << trackNo << std::endl;
#endif
    } 
    else if (!trackingVerboseThisTrack && ( fTrackingVerboseON )) 
    {
	SetTrackingVerbose(0);
	fTrackingVerboseON = 0;
#ifdef DEBUG
	std::cout << "TV: VERBOSEtt0 " << trackNo << std::endl;
#endif
    }
  }
}

void TrackingVerboseAction::SetTrackingVerbose(int verblev)
{
#ifdef DEBUG
    std::cout << " setting verbose level " << verblev <<std::endl;
#endif
    if (theTrackingManager!=0) theTrackingManager->SetVerboseLevel(verblev);
}
 
