#ifndef SimG4Core_CurrentG4Track_H
#define SimG4Core_CurrentG4Track_H

#include "G4Track.hh"

/** This class is NOT intended for general use.
 *  It includes static methods needed in exceptional cases.
 *  It provides immediate access to the currently tracked G4Track
 *  for places that can't access this information easily,
 *  like StackingAction.
 *  It also provide an access to the total number of workers in the run
 */

class CurrentG4Track {
public:
  static const G4Track *track();
  static int numberOfWorkers();
  static void setNumberOfWorkers(int);
  static void setTrack(const G4Track *);

private:
  static thread_local const G4Track *m_track;
  static int m_nWorkers;
};

#endif
