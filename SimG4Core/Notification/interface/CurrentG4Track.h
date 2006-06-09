#ifndef SimG4Core_CurrentG4Track_H
#define SimG4Core_CurrentG4Track_H

#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "G4Track.hh"

class TrackingAction;

/** This class is NOT intended for general use.
 *  It provides immediate access to the currently tracked G4Track
 *  for places that can't access this information easily,
 *  like StackingAction.
 *  If an acceptable  geant4 mechanism is found for this,
 *  this class will be removed.
 */

class CurrentG4Track 
{
public:
    static int id() { check(); return m_track->GetTrackID(); }
    static const G4Track * track() { check(); return m_track; }
private:
    static const G4Track * m_track;
    static bool m_tracking;
    static void setTrack(const G4Track *);
    static void postTracking(const G4Track *);
    static void check()	
    { if (m_track == 0) throw SimG4Exception("CurrentG4Track requested but not set"); }
    friend class TrackingAction;
};

#endif
