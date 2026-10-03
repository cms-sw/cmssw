#ifndef SimG4Core_CustomPhysics_QuirkInfracolorForce_H
#define SimG4Core_CustomPhysics_QuirkInfracolorForce_H

// String state seen by one quirk: the deque of string vectors traveling
// towards it, coupled to the partner's state through m_reactionForce.
// Ported from Athena Simulation/G4Extensions/Quirks.

#include "SimG4Core/CustomPhysics/interface/QuirkStringVector.h"

#include "G4LorentzVector.hh"

#include <deque>

class G4Track;

class QuirkInfracolorForce {
public:
  QuirkInfracolorForce() = default;
  QuirkInfracolorForce(const QuirkInfracolorForce&) = delete;
  QuirkInfracolorForce& operator=(const QuirkInfracolorForce&) = delete;

  void SetReactionForce(QuirkInfracolorForce* reactionForce) { m_reactionForce = reactionForce; }
  const QuirkInfracolorForce* GetReactionForce() const { return m_reactionForce; }
  QuirkInfracolorForce* GetReactionForce() { return m_reactionForce; }

  void StartTracking(const G4Track* dest);
  void TrackKilled() { m_killed = true; }
  G4bool IsSourceAlive() const { return !m_reactionForce->m_killed; }
  G4bool IsSourceInitialized() const { return m_reactionForce->m_initialized; }
  G4bool HasNextStringVector() const { return !m_stringVectors.empty(); }
  const std::deque<QuirkStringVector>& GetStringVectors() const { return m_stringVectors; }
  void PopTo(std::deque<QuirkStringVector>::const_iterator stringPtr, G4double fracLeft);
  void PushStringVector(const QuirkStringVector& v);

  // setters act on both ends of the string
  void SetStringForce(G4double stringForce);
  void SetFirstStringLength(G4double firstStringLength);
  void SetMaxBoost(G4double maxBoost);
  void SetMaxMergeT(G4double maxMergeT);
  void SetMaxMergeMag(G4double maxMergeMag);

  G4double GetStringForce() const { return m_stringForce; }
  G4double GetMaxExpRapidity() const { return m_maxExpRapidity; }
  G4int GetNStrings() const { return m_stringVectors.size(); }
  G4LorentzVector GetSumStrings() const;
  G4ThreeVector GetAngMomentum() const;
  G4ThreeVector GetMomentOfE() const;
  void Clear();

  // 4-velocity after the last step, used to restart the string at a crossing
  void SetVelocity(const G4LorentzVector& u) { m_lastU = u; }
  // both quirks at one point: new first string, as at production
  void Restart(const G4LorentzVector& u);

  // kinetic energy after the last step (-1: not stepped yet), used to tell a stopped pair
  void SetKineticEnergy(G4double ekin) { m_kineticEnergy = ekin; }
  G4double GetPartnerKineticEnergy() const { return m_reactionForce->m_kineticEnergy; }

private:
  void CombineStringVector(const QuirkStringVector& v);

  G4double m_stringForce{0.0};                     // string tension
  QuirkInfracolorForce* m_reactionForce{nullptr};  // string state of the partner quirk
  G4LorentzVector m_initU;                         // initial 4-velocity of the quirk
  G4bool m_initialized{false};                     // m_initU is set
  G4bool m_killed{false};                          // quirk was killed
  G4bool m_firstStep{false};                       // first step uses m_firstString
  std::deque<QuirkStringVector> m_stringVectors;   // vectors moving towards this quirk
  QuirkStringVector m_firstString;                 // vector used for the first step
  QuirkStringVector m_borrowedString;              // part of m_firstString to repay
  G4double m_firstStringLength{1.e-6};             // [mm]
  G4double m_maxExpRapidity{1e-3};                 // max exp(rapidity) per step
  G4double m_maxMergeT{0.0};
  G4double m_maxMergeMag{0.0};
  G4double m_kineticEnergy{-1.0};
  G4LorentzVector m_lastU;
};

#endif
