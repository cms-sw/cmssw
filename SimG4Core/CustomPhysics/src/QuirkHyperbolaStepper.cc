#include "SimG4Core/CustomPhysics/interface/QuirkHyperbolaStepper.h"
#include "SimG4Core/CustomPhysics/interface/QuirkDummyEquation.h"

#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4Field.hh"
#include "G4ios.hh"

QuirkHyperbolaStepper::QuirkHyperbolaStepper(QuirkInfracolorForce& string, const G4Track& track, const G4Field* field)
    : G4MagIntegratorStepper(new QuirkDummyEquation(), m_NUM_VARS),
      m_string(string),
      m_field(field),
      m_mass(track.GetDynamicParticle()->GetMass()),
      m_charge(track.GetDynamicParticle()->GetCharge()),
      m_startMomentum(track.GetDynamicParticle()->Get4Momentum()),
      m_maxExpRapidity(string.GetMaxExpRapidity()),
      m_currStep(this, string, track),
      m_nPrevSteps(0),
      m_debug(false) {
  // precompute the steps
  m_currStep.PrepareNextStep();
  m_steps.push_back(m_currStep);
  m_currStep.Step(m_currStep.GetMaxLength());
  while (m_currStep.GetStringPtr() != string.GetStringVectors().end()) {
    QuirkHyperbolaStep next = m_currStep;
    next.PrepareNextStep();
    if (next.IsBoostLimited())
      break;  // no 2nd+ step if the boost limit would cut it
    m_steps.push_back(next);
    m_currStep = next;
    m_currStep.Step(m_currStep.GetMaxLength());
  }
  m_nPrevSteps = m_steps.size() - 1;
}

QuirkHyperbolaStepper::~QuirkHyperbolaStepper() { delete GetEquationOfMotion(); }

void QuirkHyperbolaStepper::SetCurrStep(G4double length) {
  if (m_currStep.GetLength() == length)
    return;
  std::vector<QuirkHyperbolaStep>::size_type i = 0;
  while (i + 1 < m_steps.size() && m_steps[i + 1].GetLength() < length)
    ++i;
  m_currStep = m_steps[i];
  m_currStep.Step(length);
  m_nPrevSteps = i;
}

void QuirkHyperbolaStepper::Stepper(const G4double y[], const G4double[], G4double h, G4double yout[], G4double yerr[]) {
  G4int maxvar = GetNumberOfStateVariables();
  for (G4int i = 0; i < maxvar; ++i)
    yout[i] = y[i];

  if (m_debug)
    G4cout << "QuirkHyperbolaStepper: asked to move " << h << "\n  start = " << y[0] << ", " << y[1] << ", " << y[2]
           << " [" << y[8] << "]" << G4endl;

  SetCurrStep(y[8] + h);
  m_currStep.Dump(yout);

  if (m_debug)
    G4cout << "QuirkHyperbolaStepper: end   = " << yout[0] << ", " << yout[1] << ", " << yout[2] << " [" << yout[8]
           << "]" << G4endl;

  for (G4int i = 0; i < m_NUM_VARS; ++i)
    yerr[i] = 0;
}

void QuirkHyperbolaStepper::Update(G4FieldTrack& fieldTrack, G4bool forceStep, G4bool moveOnForcedStep) {
  if (forceStep) {
    // propagator refused a too short step: take it here, update string and
    // 4-momentum (position too if moveOnForcedStep)
    m_currStep = m_steps.back();
    m_currStep.Step(m_currStep.GetMaxLength());
    m_nPrevSteps = m_steps.size() - 1;
    G4LorentzVector p = m_currStep.GetMomentum();
    fieldTrack.UpdateFourMomentum(p.t() - m_mass, p.vect().unit());
    if (moveOnForcedStep) {
      // else a head-on pair never gets through its crossing point
      G4double y[m_NUM_VARS];
      m_currStep.Dump(y);
      fieldTrack.SetPosition(G4ThreeVector(y[0], y[1], y[2]));
      fieldTrack.SetLabTimeOfFlight(y[7]);
      fieldTrack.SetCurveLength(y[8]);
      fieldTrack.SetProperTimeOfFlight(y[8]);
    }
  } else {
    SetCurrStep(fieldTrack.GetProperTimeOfFlight());
  }

  // update the string vectors
  m_string.PopTo(m_currStep.GetStringPtr(), m_currStep.GetFracLeft());
  for (std::vector<QuirkHyperbolaStep>::size_type i = 0; i < m_nPrevSteps; ++i) {
    if (m_debug)
      G4cout << "QuirkHyperbolaStepper: pushing vector " << m_steps[i].GetStringOut().lv() << G4endl;
    m_string.GetReactionForce()->PushStringVector(m_steps[i].GetStringOut());
  }
  if (m_debug)
    G4cout << "QuirkHyperbolaStepper: pushing vector " << m_currStep.GetStringOut().lv() << G4endl;
  m_string.GetReactionForce()->PushStringVector(m_currStep.GetStringOut());
}
