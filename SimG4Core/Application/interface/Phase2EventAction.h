#ifndef SimG4Core_Phase2EventAction_H
#define SimG4Core_Phase2EventAction_H
//
// Package:     Application
// Class  :     Phase2EventAction
//
// Description: Manage MC truth 
// Created:     08.04.2025 V.Ivantchenko
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4UserEventAction.hh"

#include <vector>
#include <string>

class SimRunInterface;
class BeginOfEvent;
class EndOfEvent;
class CMSSteppingVerbose;

class Phase2EventAction : public G4UserEventAction {
public:
  explicit Phase2EventAction(const edm::ParameterSet& ps, SimRunInterface*, SimTrackManager*, CMSSteppingVerbose*);
  ~Phase2EventAction() override = default;

  void BeginOfEventAction(const G4Event* evt) override;
  void EndOfEventAction(const G4Event* evt) override;

  void abortEvent();

  const TrackContainer* trackContainer() const { return m_trackManager->trackContainer(); }

  TrackWithHistory* getTrackByID(int id) const { return m_trackManager->getTrackByID(id); }

  SimActivityRegistry::BeginOfEventSignal m_beginOfEventSignal;
  SimActivityRegistry::EndOfEventSignal m_endOfEventSignal;

  Phase2EventAction(const Phase2EventAction&) = delete;
  const Phase2EventAction& operator=(const Phase2EventAction&) = delete;
  
private:
  SimRunInterface* m_runInterface;
  SimTrackManager* m_trackManager;
  CMSSteppingVerbose* m_SteppingVerbose;
  std::string m_stopFile;
  bool m_printRandom;
  bool m_debug;
};

#endif
