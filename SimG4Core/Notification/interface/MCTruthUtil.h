#ifndef SimG4Core_MCTruthUtil_H
#define SimG4Core_MCTruthUtil_H

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
  static void secondary(G4Track* aSecondary, const G4Track& mother, int);
  static bool isInBTL(const G4Track*);
};

#endif
