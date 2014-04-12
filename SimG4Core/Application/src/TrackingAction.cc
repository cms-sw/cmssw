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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4UImanager.hh" 
#include "G4TrackingManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4TransportationManager.hh"

//#define DebugLog

//using namespace std;

TrackingAction::TrackingAction(EventAction * e, const edm::ParameterSet & p) 
  : eventAction_(e),currentTrack_(0),
  detailedTiming(p.getUntrackedParameter<bool>("DetailedTiming",false)),
  checkTrack(p.getUntrackedParameter<bool>("CheckTrack",false)),
  trackMgrVerbose(p.getUntrackedParameter<int>("G4TrackManagerVerbosity",0)) 
{
  worldSolid = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume()->GetLogicalVolume()->GetSolid();
}

TrackingAction::~TrackingAction() {}

void TrackingAction::PreUserTrackingAction(const G4Track * aTrack)
{
  CurrentG4Track::setTrack(aTrack);

  if (currentTrack_ != 0) {
    throw SimG4Exception("TrackingAction: currentTrack is a mess...");
  }
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

  TrackInformation * trkInfo = (TrackInformation *)aTrack->GetUserInformation();
  if(trkInfo && trkInfo->isPrimary()) {
    eventAction_->prepareForNewPrimary();
  }
  /*
    G4cout << "Track " << aTrack->GetTrackID() << " R " 
    << (aTrack->GetVertexPosition()).r() << " Z " 
    << std::abs((aTrack->GetVertexPosition()).z()) << G4endl << "Top Solid " 
    << worldSolid->GetName() << " is it inside " 
    << worldSolid->Inside(aTrack->GetVertexPosition()) 
    << " compared to " << kOutside << G4endl;
  */
  // VI: why this check is TrackingAction?
  if (worldSolid->Inside(aTrack->GetVertexPosition()) == kOutside) {
    //      G4cout << "Kill Track " << aTrack->GetTrackID() << G4endl;
    G4Track* theTrack = (G4Track *)(aTrack);
    theTrack->SetTrackStatus(fStopAndKill);
  }      
}

void TrackingAction::PostUserTrackingAction(const G4Track * aTrack)
{
  CurrentG4Track::postTracking(aTrack);
  if (eventAction_->trackContainer() != 0) {

    TrackInformationExtractor extractor;
    if (extractor(aTrack).storeTrack()) {
      currentTrack_->save();
	  
      math::XYZVectorD pos((aTrack->GetStep()->GetPostStepPoint()->GetPosition()).x(),
			   (aTrack->GetStep()->GetPostStepPoint()->GetPosition()).y(),
			   (aTrack->GetStep()->GetPostStepPoint()->GetPosition()).z());
      math::XYZTLorentzVectorD mom;
	  
      uint32_t id = aTrack->GetTrackID();
	  
      std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> p(pos,mom);
      eventAction_->addTkCaloStateInfo(id,p);
#ifdef DebugLog
      LogDebug("SimTrackManager") << "TrackingAction addTkCaloStateInfo " 
				  << id << " of momentum " << mom << " at " << pos;
#endif
    }

    bool withAncestor = 
      ((extractor(aTrack).getIDonCaloSurface() == aTrack->GetTrackID()) 
       || (extractor(aTrack).isAncestor()));

    if (extractor(aTrack).isInHistory()) {

      // check with end-of-track information
      if(checkTrack) { currentTrack_->checkAtEnd(aTrack); }

      eventAction_->addTrack(currentTrack_, true, withAncestor);
      /*
      cout << "TrackingAction addTrack "  
	   << currentTrack_->trackID() << " E(GeV)= " << aTrack->GetKineticEnergy()
	   << "  " << aTrack->GetDefinition()->GetParticleName()
	   << " added= " << withAncestor 
	   << " at " << aTrack->GetPosition() << endl;
      */
#ifdef DebugLog
      math::XYZVectorD pos((aTrack->GetStep()->GetPostStepPoint()->GetPosition()).x(),
			   (aTrack->GetStep()->GetPostStepPoint()->GetPosition()).y(),
			   (aTrack->GetStep()->GetPostStepPoint()->GetPosition()).z());
      LogDebug("SimTrackManager") << "TrackingAction addTrack "  
				  << currentTrack_->trackID() 
				  << " added with " << true << " and " << withAncestor 
				  << " at " << pos;
#endif

    } else {
      eventAction_->addTrack(currentTrack_, false, false);

#ifdef DebugLog
      LogDebug("SimTrackManager") << "TrackingAction addTrack " 
				  << currentTrack_->trackID() << " added with " 
				  << false << " and " << false;
#endif

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
