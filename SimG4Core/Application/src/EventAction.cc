#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Notification/interface/TmpSimEvent.h"
#include "SimG4Core/Notification/interface/TmpSimVertex.h"
#include "SimG4Core/Notification/interface/TmpSimTrack.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Randomize.hh"

EventAction::EventAction(const edm::ParameterSet& p,
                         SimRunInterface* rm,
                         SimTrackManager* iManager,
                         CMSSteppingVerbose* sv)
    : m_runInterface(rm),
      m_trackManager(iManager),
      m_SteppingVerbose(sv),
      m_stopFile(p.getParameter<std::string>("StopFile")),
      m_printRandom(p.getParameter<bool>("PrintRandomSeed")),
      m_debug(p.getUntrackedParameter<bool>("debug", false)) {}

void EventAction::BeginOfEventAction(const G4Event* anEvent) {
  m_trackManager->reset();

  BeginOfEvent e(anEvent);
  m_beginOfEventSignal(&e);

  if (m_printRandom) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "BeginOfEvent " << anEvent->GetEventID() << " Random number: " << G4UniformRand();
  }

  if (nullptr != m_SteppingVerbose) {
    m_SteppingVerbose->beginOfEvent(anEvent);
  }
}

void EventAction::EndOfEventAction(const G4Event* anEvent) {
  if (m_printRandom) {
    edm::LogVerbatim("SimG4CoreApplication")
        << " EndOfEvent " << anEvent->GetEventID() << " Random number: " << G4UniformRand();
  }
  if (!m_stopFile.empty() && std::ifstream(m_stopFile.c_str())) {
    edm::LogWarning("SimG4CoreApplication")
        << "EndOfEventAction: termination signal received at event " << anEvent->GetEventID();
    // soft abort run
    m_runInterface->abortRun(true);
  }
  if (anEvent->GetNumberOfPrimaryVertex() == 0) {
    edm::LogWarning("SimG4CoreApplication") << "EndOfEventAction: event " << anEvent->GetEventID()
                                            << " must have failed (no G4PrimaryVertices found) and will be skipped ";
    return;
  }

  m_trackManager->storeTracks();

  // dispatch now end of event
  EndOfEvent e(anEvent);
  m_endOfEventSignal(&e);

  // delete transient objects
  m_trackManager->reset();
}

void EventAction::abortEvent() { m_runInterface->abortEvent(); }
