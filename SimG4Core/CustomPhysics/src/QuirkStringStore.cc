#include "SimG4Core/CustomPhysics/interface/QuirkStringStore.h"

QuirkStringStore::QuirkStringStore() {
  m_strings[0].SetReactionForce(&m_strings[1]);
  m_strings[1].SetReactionForce(&m_strings[0]);
}

QuirkStringStore& QuirkStringStore::instance() {
  // one Geant4 worker per thread, so one string pair per thread
  static thread_local QuirkStringStore store;
  return store;
}

void QuirkStringStore::configure(
    G4double stringForce, G4double firstStringLength, G4double maxBoost, G4double maxMergeT, G4double maxMergeMag) {
  // setters update both ends of the string
  QuirkInfracolorForce& s = m_strings[0];
  s.SetStringForce(stringForce);
  s.SetFirstStringLength(firstStringLength);
  s.SetMaxBoost(maxBoost);
  s.SetMaxMergeT(maxMergeT);
  s.SetMaxMergeMag(maxMergeMag);
}
