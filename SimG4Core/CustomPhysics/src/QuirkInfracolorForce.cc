#include "SimG4Core/CustomPhysics/interface/QuirkInfracolorForce.h"

#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4Exception.hh"

void QuirkInfracolorForce::SetStringForce(G4double stringForce) {
  m_stringForce = stringForce;
  m_reactionForce->m_stringForce = stringForce;
}

void QuirkInfracolorForce::SetFirstStringLength(G4double firstStringLength) {
  m_firstStringLength = firstStringLength;
  m_reactionForce->m_firstStringLength = firstStringLength;
}

void QuirkInfracolorForce::SetMaxBoost(G4double maxBoost) {
  m_maxExpRapidity = std::sqrt((1.0 + maxBoost) / (1.0 - maxBoost));
  m_reactionForce->m_maxExpRapidity = m_maxExpRapidity;
}

void QuirkInfracolorForce::SetMaxMergeT(G4double maxMergeT) {
  m_maxMergeT = maxMergeT;
  m_reactionForce->m_maxMergeT = maxMergeT;
}

void QuirkInfracolorForce::SetMaxMergeMag(G4double maxMergeMag) {
  m_maxMergeMag = maxMergeMag;
  m_reactionForce->m_maxMergeMag = maxMergeMag;
}

void QuirkInfracolorForce::StartTracking(const G4Track* dest) {
  if (nullptr == m_reactionForce) {
    G4Exception("QuirkInfracolorForce::StartTracking", "NoAntiQuirk", FatalErrorInArgument, "No antiquirk defined");
  }
  if (dest->GetCurrentStepNumber() != 0)
    return;

  // stale state from an earlier event: clear both ends together
  if (m_initialized || m_killed || m_reactionForce->m_killed) {
    Clear();
    m_reactionForce->Clear();
  }

  m_initialized = true;
  m_initU = dest->GetDynamicParticle()->Get4Momentum();
  m_initU /= m_initU.m();
  m_lastU = m_initU;

  // first string vector, once the partner is known
  if (m_reactionForce->m_initialized) {
    m_firstStep = true;
    G4double dot = m_initU * m_reactionForce->m_initU;
    G4LorentzVector firstString = m_reactionForce->m_initU - m_initU * (dot - std::sqrt(dot * dot - 1));
    firstString *= m_firstStringLength / firstString.t();
    m_firstString.set(firstString.vect(), 0);
    m_stringVectors.push_back(m_firstString);
  }
}

void QuirkInfracolorForce::Restart(const G4LorentzVector& u) {
  const G4LorentzVector partnerU = m_reactionForce->m_lastU;
  const G4double ekin = m_kineticEnergy;
  const G4double partnerEkin = m_reactionForce->m_kineticEnergy;
  Clear();
  m_reactionForce->Clear();
  m_kineticEnergy = ekin;
  m_reactionForce->m_kineticEnergy = partnerEkin;
  m_reactionForce->m_initialized = true;
  m_reactionForce->m_initU = partnerU;
  m_reactionForce->m_lastU = partnerU;
  m_initialized = true;
  m_initU = u;
  m_lastU = u;
  // same first string as in StartTracking
  m_firstStep = true;
  G4double dot = m_initU * partnerU;
  G4LorentzVector firstString = partnerU - m_initU * (dot - std::sqrt(std::max(dot * dot - 1, 0.)));
  firstString *= m_firstStringLength / firstString.t();
  m_firstString.set(firstString.vect(), 0);
  m_stringVectors.push_back(m_firstString);
}

void QuirkInfracolorForce::Clear() {
  m_initU.set(0, 0, 0, 0);
  m_initialized = false;
  m_killed = false;
  m_firstStep = false;
  m_stringVectors.clear();
  m_firstString.set(0, 0, 0, 0);
  m_borrowedString.set(0, 0, 0, 0);
  m_kineticEnergy = -1.0;
}

void QuirkInfracolorForce::PopTo(std::deque<QuirkStringVector>::const_iterator stringPtr, G4double fracLeft) {
  if (fracLeft > 1.0 || fracLeft < 0.0) {
    Clear();
    m_reactionForce->Clear();
    G4Exception("QuirkInfracolorForce::PopTo",
                "QuirkStringBadFraction",
                EventMustBeAborted,
                "invalid fraction of string vector");
  }
  if (m_firstStep) {
    if (stringPtr == m_stringVectors.begin()) {
      m_borrowedString = (1 - fracLeft) * m_firstString;
    } else {
      m_borrowedString = m_firstString;
    }
    m_firstStep = false;
    m_stringVectors.clear();
  } else {
    auto stringPtr2 = m_stringVectors.begin() + (stringPtr - m_stringVectors.cbegin());
    m_stringVectors.erase(m_stringVectors.begin(), stringPtr2);
    if (fracLeft != 1.0 && !m_stringVectors.empty()) {
      m_stringVectors[0] *= fracLeft;
    }
  }
}

void QuirkInfracolorForce::PushStringVector(const QuirkStringVector& v) {
  if (v.t() == 0)
    return;
  if (m_borrowedString.t() == 0) {
    CombineStringVector(v);
  } else if (m_borrowedString.t() <= v.t()) {
    G4double r = m_borrowedString.t() / v.t();
    CombineStringVector((1 - r) * v);
    m_borrowedString.set(0, 0, 0, 0);
  } else {
    G4double r = v.t() / m_borrowedString.t();
    m_borrowedString *= 1 - r;
    G4Exception("QuirkInfracolorForce::PushStringVector", "BorrowedStringSplit", JustWarning, "Initial step too long.");
  }
}

void QuirkInfracolorForce::CombineStringVector(const QuirkStringVector& v) {
  if (m_stringVectors.empty()) {
    m_stringVectors.push_back(v);
  } else {
    G4LorentzVector sum = m_stringVectors.back().lv() + v.lv();
    if (sum.t() < m_maxMergeT && sum.m2() < m_maxMergeMag * m_maxMergeMag) {
      m_stringVectors.back().set(sum.vect(), sum.m());
    } else {
      m_stringVectors.push_back(v);
    }
  }
}

G4LorentzVector QuirkInfracolorForce::GetSumStrings() const {
  G4LorentzVector x(0, 0, 0, 0);
  for (auto const& s : m_stringVectors) {
    x += s.lv();
  }
  return x - m_borrowedString.lv();
}

G4ThreeVector QuirkInfracolorForce::GetAngMomentum() const {
  G4LorentzVector x(0, 0, 0, 0);
  G4ThreeVector L(0, 0, 0);
  for (auto const& s : m_stringVectors) {
    G4LorentzVector dx = s.lv();
    L += m_stringForce * x.vect().cross(dx.vect());
    x += dx;
  }
  return L;
}

G4ThreeVector QuirkInfracolorForce::GetMomentOfE() const {
  G4LorentzVector x(0, 0, 0, 0);
  G4ThreeVector Excm(0, 0, 0);
  for (auto const& s : m_stringVectors) {
    G4LorentzVector dx = s.lv();
    Excm += m_stringForce * (dx.t() * x.vect() - dx.vect() * x.t());
    x += dx;
  }
  return Excm;
}
