#ifndef LowEnergyFastSimParam_h
#define LowEnergyFastSimParam_h

#include "G4Types.hh"
#include "Randomize.hh"
#include "G4Log.hh"

class LowEnergyFastSimParam {
public:
  G4double GetInPointEnergyFraction(G4double energy) const {
    // normalisation of fit parameters to have the result
    constexpr const G4double a0 = 1.02186764;
    constexpr const G4double a1 = 2.14064635e-02 / a0;
    constexpr const G4double a2 = 1.96988997e-04 / a0;
    constexpr const G4double a3 = -6.42310317e-07 / a0;
    const G4double e2 = energy * energy;
    const G4double e3 = e2 * energy;
    return a3 * e3 + a2 * e2 - a1 * energy + 1.0;
  }

  G4double GetRadius(G4double energy) const {
    constexpr const G4double r1 = 156.52094133;
    constexpr const G4double r2 = -1.02220543;
    const G4double r0 = r1 + r2 * energy;
    return std::sqrt(r0 / G4UniformRand() - r0);
  }

  G4double GetZ() const {
    constexpr const G4double alpha = 1.0 / 0.02211515;
    constexpr const G4double t = 0.66968625;
    return -G4Log(G4UniformRand()) * alpha + t;
  }
};

#endif
