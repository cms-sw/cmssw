#ifndef SimG4Core_CustomPhysics_QuirkHyperbolaStepper_H
#define SimG4Core_CustomPhysics_QuirkHyperbolaStepper_H

// Fake G4MagIntegratorStepper that replays precomputed hyperbola segments.
// State: x, y, z, px, py, pz, Ekin, t_lab, step length (in place of t_proper).
// Ported from Athena Simulation/G4Extensions/Quirks.

#include "SimG4Core/CustomPhysics/interface/QuirkInfracolorForce.h"
#include "SimG4Core/CustomPhysics/interface/QuirkHyperbolaStep.h"

#include "G4LorentzVector.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4FieldTrack.hh"

#include <vector>

class G4Track;
class G4Field;

class QuirkHyperbolaStepper : public G4MagIntegratorStepper {
public:
  QuirkHyperbolaStepper(QuirkInfracolorForce& string, const G4Track& track, const G4Field* field = nullptr);
  ~QuirkHyperbolaStepper() override;

  void Stepper(const G4double y[], const G4double dydx[], G4double h, G4double yout[], G4double yerr[]) override;
  G4double DistChord() const override { return 0; }
  G4int IntegratorOrder() const override { return 1; }

  G4double GetForce() const { return m_string.GetStringForce(); }
  const G4Field* GetField() const { return m_field; }
  G4double GetMass() const { return m_mass; }
  G4double GetCharge() const { return m_charge; }
  const G4LorentzVector& GetStartMomentum() const { return m_startMomentum; }
  G4double GetMaxExpRapidity() const { return m_maxExpRapidity; }
  G4double GetMaxLength() const { return m_steps.back().GetMaxLength(); }

  // consume the strings used by the step, push the reflected ones to the partner;
  // moveOnForcedStep: a step below the navigation tolerance also moves the quirk
  void Update(G4FieldTrack& fieldTrack, G4bool forceStep, G4bool moveOnForcedStep = false);

  void SetDebug(G4bool debug) { m_debug = debug; }

private:
  void SetCurrStep(G4double length);

  static constexpr G4int m_NUM_VARS = 9;

  QuirkInfracolorForce& m_string;
  const G4Field* const m_field;
  const G4double m_mass;
  const G4double m_charge;
  const G4LorentzVector m_startMomentum;
  const G4double m_maxExpRapidity;

  std::vector<QuirkHyperbolaStep> m_steps;
  QuirkHyperbolaStep m_currStep;
  std::vector<QuirkHyperbolaStep>::size_type m_nPrevSteps;

  G4bool m_debug;
};

#endif
