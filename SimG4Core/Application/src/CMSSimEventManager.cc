#include "SimG4Core/Application/interface/CMSSimEventManager.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"
#include "G4Event.hh"
#include "G4TrajectoryContainer.hh"
#include "G4PrimaryTransformer.hh"
#include "G4TrackingManager.hh"
#include "G4TrackStatus.hh"
#include "G4UserSteppingAction.hh"

#include "G4SDManager.hh"
#include "G4StateManager.hh"
#include "G4ApplicationState.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"

CMSSimEventManager::CMSSimEventManager(const edm::ParameterSet& iConfig)
    : verbose_(iConfig.getParameter<int>("EventVerbose")) {
  m_stateManager = G4StateManager::GetStateManager();
  m_defTrackManager = new G4TrackingManager();
  m_primaryTransformer = new G4PrimaryTransformer();
  m_sdManager = G4SDManager::GetSDMpointerIfExist();
  m_navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  m_tracks.reserve(1000);
}

CMSSimEventManager::~CMSSimEventManager() {
  delete m_primaryTransformer;
  delete m_defTrackManager;
  delete m_eventAction;
  delete m_stackingAction;
}

void CMSSimEventManager::ProcessOneEvent(G4Event* anEvent) {
  trackID_ = 0;
  m_stateManager->SetNewState(G4State_EventProc);

  // Resetting Navigator has been moved to CMSSimEventManager,
  // so that resetting is now done for every event.
  G4ThreeVector center(0, 0, 0);
  m_navigator->LocateGlobalPointAndSetup(center, nullptr, false);

  G4Track* track = nullptr;

  anEvent->SetHCofThisEvent(m_sdManager->PrepareNewEvent());
  m_sdManager->PrepareNewEvent();
  m_eventAction->BeginOfEventAction(anEvent);

  // Fill primary tracks
  StackTracks(m_primaryTransformer->GimmePrimaries(anEvent), true);

  if (0 < verbose_) {
    edm::LogVerbatim("CMSSimEventManager::ProcessOneEvent")
        << "### Event #" << anEvent->GetEventID() << "  " << trackID_ << " primary tracks";
  }

  // Loop over main stack of tracks
  do {
    track = m_tracks.back();
    m_tracks.pop_back();
    m_defTrackManager->ProcessOneTrack(track);
    G4TrackVector* secondaries = m_defTrackManager->GimmeSecondaries();
    StackTracks(secondaries, false);
    delete track;
  } while (!m_tracks.empty());

  m_sdManager->TerminateCurrentEvent(anEvent->GetHCofThisEvent());
  m_eventAction->EndOfEventAction(anEvent);
  m_stateManager->SetNewState(G4State_GeomClosed);
}

void CMSSimEventManager::StackTracks(G4TrackVector* trackVector, bool IDisSet) {
  if (trackVector == nullptr || trackVector->empty())
    return;
  for (auto& newTrack : *trackVector) {
    ++trackID_;
    if (!IDisSet) {
      newTrack->SetTrackID(trackID_);
      auto pp = newTrack->GetDynamicParticle()->GetPrimaryParticle();
      if (pp != nullptr) {
        pp->SetTrackID(trackID_);
      }
    }
    if (m_stackingAction->ClassifyNewTrack(newTrack) == fKill) {
      delete newTrack;
    } else {
      newTrack->SetOriginTouchableHandle(newTrack->GetTouchableHandle());
      m_tracks.push_back(newTrack);
    }
  }
  trackVector->clear();
}

void CMSSimEventManager::SetUserAction(EventAction* ptr) { m_eventAction = ptr; }

void CMSSimEventManager::SetUserAction(StackingAction* ptr) { m_stackingAction = ptr; }

void CMSSimEventManager::SetUserAction(TrackingAction* ptr) {
  m_trackingAction = ptr;
  m_defTrackManager->SetUserAction((G4UserTrackingAction*)ptr);
}

void CMSSimEventManager::SetUserAction(G4UserSteppingAction* ptr) {
  m_defTrackManager->SetUserAction(ptr);
}
