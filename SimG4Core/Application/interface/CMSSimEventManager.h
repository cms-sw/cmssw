//
// CMSSimEventManager is designed on top of G4EventManager
//
// 13.03.2023 V.Ivanchenko
//

// --------------------------------------------------------------------
#ifndef SimG4Core_Application_CMSSimEventManager_hh
#define SimG4Core_Application_CMSSimEventManager_hh 1

#include <vector>
#include "globals.hh"
#include "G4Track.hh"
#include "G4TrackVector.hh"

namespace edm {
  class ParameterSet;
}

class G4Event;
class G4UserEventAction;
class G4UserStackingAction;
class G4UserTrackingAction;
class G4UserSteppingAction;
class G4SDManager;
class G4StateManager;
class G4PrimaryTransformer;
class G4TrackingManager;
class G4Navigator;

class CMSSimEventManager {
public:
  CMSSimEventManager(const edm::ParameterSet& iConfig);
  ~CMSSimEventManager();

  void InitialiseWorker();

  // This method is the main entry to this class for simulating an event.
  void ProcessOneEvent(G4Event* anEvent);

  // This method aborts the processing of the current event.
  void AbortCurrentEvent();

  void SetUserAction(G4UserEventAction* ptr);
  void SetUserAction(G4UserStackingAction* ptr);
  void SetUserAction(G4UserTrackingAction* ptr);
  void SetUserAction(G4UserSteppingAction* ptr);

  CMSSimEventManager(const CMSSimEventManager& right) = delete;
  CMSSimEventManager& operator=(const CMSSimEventManager& right) = delete;

private:
  void StackTracks(G4TrackVector*, bool IDisSet);

  G4StateManager* m_stateManager;
  G4TrackingManager* m_defTrackManager;
  G4SDManager* m_sdManager;
  G4PrimaryTransformer* m_primaryTransformer;
  G4Navigator* m_navigator;

  G4UserEventAction* m_eventAction;
  G4UserTrackingAction* m_trackingAction;
  G4UserStackingAction* m_stackingAction;

  G4int trackID_{0};
  G4int verbose_;

  std::vector<G4Track*> m_tracks;
};

#endif
