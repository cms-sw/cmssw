#ifndef SimG4Core_CMSSteppingVerbose_h
#define SimG4Core_CMSSteppingVerbose_h

//---------------------------------------------------------------
//
// CMSSteppingVerbose is intend to replace Geant4 default stepping
//                    verbosity class in order to keep necessary
//                    verbosity options when G4VERBOSE flag is disabled.
//                    The goal is to provide easy way to print
//                    per event, per track, per step.
//
// V.Ivanchenko 10.06.2016
//
//---------------------------------------------------------------

#include "globals.hh"
#include <vector>

class G4Event;
class G4Track;
class G4Step;
class G4SteppingManager;
class G4SteppingVerbose;

class CMSSteppingVerbose {
public:
  explicit CMSSteppingVerbose(
      G4int verb, G4double ekin, std::vector<G4int>& evtNum, std::vector<G4int>& primV, std::vector<G4int>& trNum);
  ~CMSSteppingVerbose() = default;

  void beginOfEvent(const G4Event*);
  void trackStarted(const G4Track*, bool isKilled);
  void trackEnded(const G4Track*) const;
  void stackFilled(const G4Track*, bool isKilled) const;
  void nextStep(const G4Step*, const G4SteppingManager* ptr, bool isKilled) const;

  void stopEventPrint();
  void setVerbose(int val) { m_verbose = val; }

private:
  G4bool m_PrintEvent;
  G4bool m_PrintTrack;
  G4bool m_smInitialized;
  G4int m_verbose;
  G4int m_nEvents;
  G4int m_nVertex;
  G4int m_nTracks;
  std::vector<G4int> m_EventNumbers;
  std::vector<G4int> m_PrimaryVertex;
  std::vector<G4int> m_TrackNumbers;
  G4double m_EkinThreshold;
  G4SteppingVerbose* m_g4SteppingVerbose;
};

#endif
