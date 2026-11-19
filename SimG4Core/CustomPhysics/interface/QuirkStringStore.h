#ifndef SimG4Core_CustomPhysics_QuirkStringStore_H
#define SimG4Core_CustomPhysics_QuirkStringStore_H

// Per-thread owner of the two coupled string states of the quirk pair.
// Index 0 is the quirk with positive PDG code, 1 the antiquirk.

#include "SimG4Core/CustomPhysics/interface/QuirkInfracolorForce.h"

class QuirkStringStore {
public:
  static QuirkStringStore& instance();

  QuirkInfracolorForce& stringFor(G4int pdg) { return (pdg > 0) ? m_strings[0] : m_strings[1]; }

  void configure(
      G4double stringForce, G4double firstStringLength, G4double maxBoost, G4double maxMergeT, G4double maxMergeMag);

  void clear() {
    m_strings[0].Clear();
    m_strings[1].Clear();
  }

  // true once per event: the first suspension waits for the urgent stack, later ones alternate
  G4bool firstSuspension(G4int eventID) {
    if (eventID == m_eventID)
      return false;
    m_eventID = eventID;
    return true;
  }

  QuirkStringStore(const QuirkStringStore&) = delete;
  QuirkStringStore& operator=(const QuirkStringStore&) = delete;

private:
  QuirkStringStore();

  QuirkInfracolorForce m_strings[2];
  G4int m_eventID{-1};
};

#endif
