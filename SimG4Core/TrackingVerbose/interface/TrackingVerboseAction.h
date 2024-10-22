///////////////////////////////////////////////////////////////////////////////
// File : TrackingVerboseAction.h
// Author: P.Arce  12.09.01
// Ported to CMSSW by: M. Stavrianakou  22.03.06
// Description:
// Modifications:
// Class with the commands to switch on/off the verbosity of tracking and event
// see TrackingVerboseAction for a detailed explanation
// for a given range of tracks each 'n' tracks
// the GEANT4 command '/tracking/verbose N' will be executed when trackNo is
//     fTVTrackMin <= trackNo <= fTVTrackMax
// each fTVTrackStep tracks (starting at 1, not 0) and if the trackNo is
//     fTVTrackMin <= trackNo <= fTVTrackMax
// each fTVTrackStep tracks (starting at 1, not 0)
//
///////////////////////////////////////////////////////////////////////////////

#ifndef SimG4Core_TrackingVerbose_h
#define SimG4Core_TrackingVerbose_h 1

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"

#include <vector>

class BeginOfTrack;
class EndOfTrack;
class BeginOfEvent;
class BeginOfRun;
class G4Track;
class G4TrackingManager;
class G4VSteppingVerbose;

class TrackingVerboseAction : public SimWatcher,
                              public Observer<const BeginOfRun *>,
                              public Observer<const BeginOfEvent *>,
                              public Observer<const BeginOfTrack *>,
                              public Observer<const EndOfTrack *>,
                              public Observer<const G4Step *> {
public:
  TrackingVerboseAction(edm::ParameterSet const &p);
  ~TrackingVerboseAction() override;
  void update(const BeginOfRun *) override;
  void update(const BeginOfEvent *) override;
  void update(const BeginOfTrack *) override;
  void update(const EndOfTrack *) override;
  void update(const G4Step *) override;

private:
  void setTrackingVerbose(int verblev);
  bool checkTrackingVerbose(const G4Track *);
  void printTrackInfo(const G4Track *);

private:
  int fLarge;
  bool fDEBUG;
  bool fG4Verbose;
  bool fHighEtPhotons;
  int fTVTrackMin;
  int fTVTrackMax;
  int fTVTrackStep;
  int fTVEventMin;
  int fTVEventMax;
  int fTVEventStep;
  int fVerboseLevel;
  bool fTrackingVerboseON;
  bool fTkVerbThisEventON;
  std::vector<int> fPdgIds;
  G4TrackingManager *theTrackingManager;
  G4VSteppingVerbose *fVerbose;
};

#endif
