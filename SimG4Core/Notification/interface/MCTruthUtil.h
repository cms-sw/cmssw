#ifndef SimG4Core_MCTruthUtil_H
#define SimG4Core_MCTruthUtil_H

#include "G4TrackStatus.hh"

class G4Track;

/* 
 *  Creation and filling initial MC truth information for
 *  primary and secondary G4Track objects, addition to G4Track
 *  an extra CMS container TrackInformation.
 *  Currently is used in StackingAction.
 */

class MCTruthUtil {
public:
  static void primary(G4Track* aPrimary);
  static void secondary(G4Track* aSecondary, const G4Track* mother, int);
  static void updateSecondary(G4Track* aSecondary);
  static bool isInBTL(const G4Track*);
  // suspended tracks are resumed later (e.g. quirk pairs), not new tracks
  static bool isSuspended(G4TrackStatus status) { return status == fSuspend || status == fSuspendAndWait; }
  static bool isSuspended(const G4Track*);
};

#endif
