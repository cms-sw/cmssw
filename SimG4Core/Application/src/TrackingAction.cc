#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
 
#include "G4TrackingManager.hh"

TrackingAction::TrackingAction(EventAction * e, const edm::ParameterSet & p) 
: eventAction_(e),currentTrack_(0),
  detailedTiming(p.getUntrackedParameter<bool>("DetailedTiming",false)),
  trackMgrVerbose(p.getUntrackedParameter<int>("G4TrackManagerVerbosity",0)){}

TrackingAction::~TrackingAction() {}

void TrackingAction::PreUserTrackingAction(const G4Track * aTrack)
{
    CurrentG4Track::setTrack(aTrack);
    if (currentTrack_ != 0) 
	throw SimG4Exception("TrackingAction: currentTrack is a mess...");
    currentTrack_ = new TrackWithHistory(aTrack);
    BeginOfTrack bt(aTrack);
    m_beginOfTrackSignal(&bt);
}

void TrackingAction::PostUserTrackingAction(const G4Track * aTrack)
{
    CurrentG4Track::postTracking(aTrack);
    if (eventAction_->trackContainer() != 0)
    {
        TrackInformationExtractor extractor;
        if (extractor(aTrack).storeTrack()) currentTrack_->save();
        if (extractor(aTrack).isInHistory())
        {
            currentTrack_->checkAtEnd(aTrack);  // check with end-of-track information
            eventAction_->addTrack(currentTrack_, true);
        }
        else
        {
            eventAction_->addTrack(currentTrack_, false);
	    delete currentTrack_;
	}
    }
    EndOfTrack et(aTrack);
    m_endOfTrackSignal(&et);
    currentTrack_ = 0; // reset for next track
}

G4TrackingManager * TrackingAction::getTrackManager()
{
    G4TrackingManager * theTrackingManager = 0;
    theTrackingManager = fpTrackingManager;
    theTrackingManager->SetVerboseLevel(trackMgrVerbose);
    return theTrackingManager;
}
 
