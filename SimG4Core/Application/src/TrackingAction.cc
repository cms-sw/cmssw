#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4UImanager.hh" 
#include "G4TrackingManager.hh"

//#define DebugLog

TrackingAction::TrackingAction(EventAction * e, const edm::ParameterSet & p,
			       CMSSteppingVerbose* sv) 
  : eventAction_(e),currentTrack_(nullptr),steppingVerbose_(sv),g4Track_(nullptr),
  checkTrack_(p.getUntrackedParameter<bool>("CheckTrack",false))
{}

TrackingAction::~TrackingAction() {}

void TrackingAction::PreUserTrackingAction(const G4Track * aTrack)
{
  g4Track_ = aTrack;
  currentTrack_ = new TrackWithHistory(aTrack);

  BeginOfTrack bt(aTrack);
  m_beginOfTrackSignal(&bt);
 
  TrackInformation * trkInfo = (TrackInformation *)aTrack->GetUserInformation();
  if(trkInfo && trkInfo->isPrimary()) {
    eventAction_->prepareForNewPrimary();
  }
  if(nullptr != steppingVerbose_) { 
    steppingVerbose_->TrackStarted(aTrack, false); 
  }
}

void TrackingAction::PostUserTrackingAction(const G4Track * aTrack)
{
  if (eventAction_->trackContainer() != nullptr) {

    if (extractor_(aTrack).storeTrack()) {
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
      ((extractor_(aTrack).getIDonCaloSurface() == aTrack->GetTrackID()) 
       || (extractor_(aTrack).isAncestor()));

    if (extractor_(aTrack).isInHistory()) {

      // check with end-of-track information
      if(checkTrack_) { currentTrack_->checkAtEnd(aTrack); }

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
  if(nullptr != steppingVerbose_) { steppingVerbose_->TrackEnded(aTrack); }

  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);
 
  currentTrack_ = nullptr; // reset for next track
}
