#ifndef SimG4Core_CurrentG4Track_H
#define SimG4Core_CurrentG4Track_H

#include "G4Track.hh"

/** This class is NOT intended for general use.
 *  It provides immediate access to the currently tracked G4Track
 *  for places that can't access this information easily,
 *  like StackingAction.
 *  If an acceptable  geant4 mechanism is found for this,
 *  this class will be removed.
 */

class CurrentG4Track {
public:
  static const G4Track *track();
  static void setTrack(const G4Track *);

private:
  static thread_local const G4Track *m_track;
};

#endif
