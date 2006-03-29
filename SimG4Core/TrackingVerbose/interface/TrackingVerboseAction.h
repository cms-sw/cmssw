///////////////////////////////////////////////////////////////////////////////
// File : TrackingVerboseAction.h
// Author: P.Arce  12.09.01
// Ported to CMSSW by: M. Stavrianakou  22.03.06
// Description:
// Modifications:
// Class with the commands to switch on/off the verbosity of tracking and event, 
// see TrackingVerboseAction for a detailed explanation
// for a given range of tracks each 'n' tracks
// the GEANT4 command '/tracking/verbose N' will be executed when the trackNo is
//     fTVTrackMin <= trackNo <= fTVTrackMax
// each fTVTrackStep tracks (starting at 1, not 0) and if the trackNo is
//     fTVTrackMin <= trackNo <= fTVTrackMax
// each fTVTrackStep tracks (starting at 1, not 0)
// 
///////////////////////////////////////////////////////////////////////////////

#ifndef SimG4Core_TrackingVerbose_h
#define SimG4Core_TrackingVerbose_h 1
class G4TrackingManager;

#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class BeginOfTrack;
class BeginOfEvent;
class BeginOfRun;

class TrackingVerboseAction :  public Observer<const BeginOfRun *>, 
			       public Observer<const BeginOfEvent *>, 
			       public Observer<const BeginOfTrack *>
{
 public:
    TrackingVerboseAction(edm::ParameterSet const & p);
    ~TrackingVerboseAction();
    void update(const BeginOfRun *);
    void update(const BeginOfEvent *);
    void update(const BeginOfTrack *);
private:
    void SetTrackingVerbose(int verblev);
private:
    bool fDEBUG;
    int fTVTrackMin;
    int fTVTrackMax;
    int fTVTrackStep;
    int fTVEventMin;
    int fTVEventMax;
    int fTVEventStep;
    int fVerboseLevel;
    bool fTrackingVerboseON;
    bool fTkVerbThisEventON;
    G4TrackingManager * theTrackingManager;
};

#endif
