#ifndef SimG4Core_EventAction_H
#define SimG4Core_EventAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4UserEventAction.hh"

#include <vector>
#include <map>
#include <string>

class SimRunInterface;
class BeginOfEvent;
class EndOfEvent;
class CMSSteppingVerbose;

class EventAction : public G4UserEventAction {
public:
  explicit EventAction(const edm::ParameterSet& ps, SimRunInterface*, SimTrackManager*, CMSSteppingVerbose*);
  ~EventAction() override = default;

  void BeginOfEventAction(const G4Event* evt) override;
  void EndOfEventAction(const G4Event* evt) override;

  void abortEvent();

  inline const TrackContainer* trackContainer() const { return m_trackManager->trackContainer(); }

  TrackWithHistory* getTrackByID(unsigned int id) const { return m_trackManager->getTrackByID(id); }

  SimActivityRegistry::BeginOfEventSignal m_beginOfEventSignal;
  SimActivityRegistry::EndOfEventSignal m_endOfEventSignal;

private:
  SimRunInterface* m_runInterface;
  SimTrackManager* m_trackManager;
  CMSSteppingVerbose* m_SteppingVerbose;
  std::string m_stopFile;
  bool m_printRandom;
  bool m_debug;
};

#endif
