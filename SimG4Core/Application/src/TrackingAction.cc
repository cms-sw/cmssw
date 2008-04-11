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

#include "G4UImanager.hh" 
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

    /*
      Trick suggested by Vladimir I. in order to debug with high 
      level verbosity only a single problematic tracks
    */      

    /*
    if ( aTrack->GetTrackID() == palce_here_the_trackid_of_problematic_tracks  ) {
      G4UImanager::GetUIpointer()->ApplyCommand("/tracking/verbose 6");
    } else if ( aTrack->GetTrackID() == place_here_the_trackid_of_following_track_to_donwgrade_the_severity ) {
      G4UImanager::GetUIpointer()->ApplyCommand("/tracking/verbose 0");
    }
    */
    BeginOfTrack bt(aTrack);
    m_beginOfTrackSignal(&bt);
}

void TrackingAction::PostUserTrackingAction(const G4Track * aTrack)
{
    CurrentG4Track::postTracking(aTrack);
    if (eventAction_->trackContainer() != 0)
    {

      TrackInformationExtractor extractor;
      if (extractor(aTrack).storeTrack())
	{
	  currentTrack_->save();
	  
	  math::XYZVectorD pos((aTrack->GetStep()->GetPostStepPoint()->GetPosition()).x(),
			       (aTrack->GetStep()->GetPostStepPoint()->GetPosition()).y(),
			       (aTrack->GetStep()->GetPostStepPoint()->GetPosition()).z());
	  math::XYZTLorentzVectorD mom;
	  
	  uint32_t id = aTrack->GetTrackID();
	  
	  std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> p(pos,mom);
	  eventAction_->addTkCaloStateInfo(id,p);
	  
	}
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
 
