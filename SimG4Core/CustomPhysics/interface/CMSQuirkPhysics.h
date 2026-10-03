#ifndef SimG4Core_CustomPhysics_CMSQuirkPhysics_H
#define SimG4Core_CustomPhysics_CMSQuirkPhysics_H

// Quirk physics: the quirk/antiquirk definitions and their processes
// (QuirkTransportation, hadron-like EM energy loss, QuirkWatcher).
// Enabled by g4SimHits.Physics.QuirkMass > 0.

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class Quirk;

class CMSQuirkPhysics : public G4VPhysicsConstructor {
public:
  explicit CMSQuirkPhysics(const edm::ParameterSet& p);
  ~CMSQuirkPhysics() override = default;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4int m_verbose;
  G4int m_pdg;
  G4double m_mass;
  G4double m_charge;
  G4double m_stringForce;
  G4double m_firstStringLength;
  G4double m_maxBoost;
  G4double m_maxMergeT;
  G4double m_maxMergeMag;
  G4int m_looperTrials;
  G4double m_rMax;
  G4double m_zMax;
  G4bool m_keepStopped;
  G4bool m_moveTinySteps;
  G4double m_crossingLength;
  G4double m_stopThreshold;
  Quirk* m_quirk{nullptr};
  Quirk* m_antiQuirk{nullptr};
};

#endif
