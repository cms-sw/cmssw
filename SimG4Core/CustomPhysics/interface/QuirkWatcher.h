#ifndef SimG4Core_CustomPhysics_QuirkWatcher_H
#define SimG4Core_CustomPhysics_QuirkWatcher_H

// Strongly forced post-step process: suspends a quirk when its string deque is
// empty, handing control to the partner; kills orphans and quirks leaving the
// detector envelope (r > rMax or |z| > zMax). Optionally a quirk brought to
// rest by energy loss is kept alive while its partner still moves, since the
// string keeps pulling it.
// Ported from Athena Simulation/G4Extensions/Quirks.

#include "G4VProcess.hh"
#include "G4ParticleChange.hh"

class QuirkWatcher : public G4VProcess {
public:
  explicit QuirkWatcher(G4double rMax = DBL_MAX,
                        G4double zMax = DBL_MAX,
                        G4bool keepStopped = false,
                        G4double stopThreshold = 0.0);
  ~QuirkWatcher() override = default;

  G4double PostStepGetPhysicalInteractionLength(const G4Track& track,
                                                G4double previousStepSize,
                                                G4ForceCondition* condition) override;
  G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& stepData) override;

  G4double AlongStepGetPhysicalInteractionLength(
      const G4Track&, G4double, G4double, G4double&, G4GPILSelection*) override {
    return -1.0;
  }
  G4double AtRestGetPhysicalInteractionLength(const G4Track&, G4ForceCondition*) override { return -1.0; }
  G4VParticleChange* AlongStepDoIt(const G4Track&, const G4Step&) override { return nullptr; }
  G4VParticleChange* AtRestDoIt(const G4Track&, const G4Step&) override { return nullptr; }

private:
  G4ParticleChange m_particleChange;
  G4double m_rMax2;
  G4double m_zMax;
  G4bool m_keepStopped;
  G4double m_stopThreshold;  // partner below this kinetic energy: the pair has stopped
};

#endif
