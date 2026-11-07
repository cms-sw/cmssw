#ifndef SimG4Core_CustomPhysics_QuirkStringVector_H
#define SimG4Core_CustomPhysics_QuirkStringVector_H

// One segment of the infracolor string: a 3-vector plus a "mass".
// Ported from Athena Simulation/G4Extensions/Quirks.

#include "G4ThreeVector.hh"
#include "G4LorentzVector.hh"

class QuirkStringVector {
public:
  QuirkStringVector() = default;
  QuirkStringVector(const G4ThreeVector& p, G4double m) : m_p(p), m_m(m) {}
  QuirkStringVector(G4double x, G4double y, G4double z, G4double m) : m_p(x, y, z), m_m(m) {}

  const G4ThreeVector& vect() const { return m_p; }
  G4double mag() const { return m_m; }
  G4double x() const { return m_p.x(); }
  G4double y() const { return m_p.y(); }
  G4double z() const { return m_p.z(); }
  G4double t() const { return std::sqrt(m_p.mag2() + m_m * m_m); }
  G4LorentzVector lv() const { return G4LorentzVector(m_p, t()); }

  // end-point reflection about the 4-vector axis
  QuirkStringVector reflect(const G4LorentzVector& axis) const {
    return QuirkStringVector(axis.vect() * (2 * lv() * axis) / axis.mag2() - m_p, m_m);
  }

  void set(const G4ThreeVector& p, G4double m) {
    m_p = p;
    m_m = m;
  }
  void set(G4double x, G4double y, G4double z, G4double m) {
    m_p.set(x, y, z);
    m_m = m;
  }
  void operator*=(G4double a) {
    m_p *= a;
    m_m *= a;
  }

private:
  G4ThreeVector m_p{0, 0, 0};
  G4double m_m{0};
};

inline QuirkStringVector operator*(G4double a, const QuirkStringVector& s) {
  return QuirkStringVector(a * s.vect(), a * s.mag());
}

inline QuirkStringVector operator*(const QuirkStringVector& s, G4double a) {
  return QuirkStringVector(a * s.vect(), a * s.mag());
}

#endif
