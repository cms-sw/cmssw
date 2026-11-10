#ifndef SimG4Core_CustomPhysics_QuirkHyperbolaStep_H
#define SimG4Core_CustomPhysics_QuirkHyperbolaStep_H

// One analytic step: absorb an incoming string vector, reflect it, move on a
// hyperbola under the constant string force, then a first-order Lorentz kick.
// Ported from Athena Simulation/G4Extensions/Quirks.

#include "SimG4Core/CustomPhysics/interface/QuirkStringVector.h"

#include "G4ThreeVector.hh"
#include "G4LorentzVector.hh"

#include <deque>

class G4Track;
class QuirkHyperbolaStepper;
class QuirkInfracolorForce;

class QuirkHyperbolaStep {
public:
  QuirkHyperbolaStep(const QuirkHyperbolaStepper* stepper, const QuirkInfracolorForce& string, const G4Track& track);

  void PrepareNextStep();
  void Step(G4double length);
  void Dump(G4double y[]) const;

  std::deque<QuirkStringVector>::const_iterator GetStringPtr() const { return m_stringPtr; }
  G4double GetFracLeft() const { return m_fracLeft; }
  G4bool IsBoostLimited() const { return m_maxFracTaken != 1.0; }
  G4double GetMaxLength() const { return m_maxLength; }
  G4double GetLength() const { return m_length; }
  const QuirkStringVector& GetStringOut() const { return m_stringOut; }
  const G4LorentzVector& GetMomentum() const { return m_momentum; }

private:
  const QuirkHyperbolaStepper* m_stepper;

  std::deque<QuirkStringVector>::const_iterator m_stringPtr;
  std::deque<QuirkStringVector>::const_iterator m_stringEnd;
  G4double m_fracLeft;
  G4ThreeVector m_position;
  G4double m_time;
  G4LorentzVector m_momentum;

  QuirkStringVector m_stringIn;
  G4double m_maxFracTaken;
  QuirkStringVector m_stringOut;
  G4double m_maxLength;
  G4double m_length;
};

#endif
