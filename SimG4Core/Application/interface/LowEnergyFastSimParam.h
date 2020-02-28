#ifndef LowEnergyFastSimParam_h
#define LowEnergyFastSimParam_h

#include "G4Types.hh"
#include "Randomize.hh"

class LowEnergyFastSimParam {
public:
  G4double GetInPointEnergyFraction(G4double energy) const {
    const G4double e2 = energy * energy;
    const G4double e3 = e2 * energy;
    return -6.42310317e-07 * e3 + 1.96988997e-04 * e2 - 2.14064635e-02 * energy + 1.02186764e+00;
  }

  G4double GetRadius(G4double energy) const {
    constexpr const G4double r1 = 156.52094133;
    constexpr const G4double r2 = -1.02220543;
    const G4double r0 = r1 + r2 * energy;
    const G4double erand = G4UniformRand();

    return sqrt(r0 / erand - r0);
  }

  G4double GetZ() const {
    constexpr const G4double alpha = 0.02211515;
    constexpr const G4double t = 0.66968625;
    const G4double erand = G4UniformRand();

    return -log(erand) / alpha + t;
  }
};

#endif
