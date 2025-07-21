/**
 * @file CMSAPrime.h
 * @brief Class creating the A' particle in Geant.
 * @author Michael Revering, University of Minnesota
 */

#ifndef CMSAPrime_h
#define CMSAPrime_h

// Geant
#include "G4ParticleDefinition.hh"

class CMSAPrime : public G4ParticleDefinition {
private:
  static CMSAPrime* theAPrime;

  CMSAPrime(const G4String& Name,
            G4double mass,
            G4double width,
            G4double charge,
            G4int iSpin,
            G4int iParity,
            G4int iConjugation,
            G4int iIsospin,
            G4int iIsospin3,
            G4int gParity,
            const G4String& pType,
            G4int lepton,
            G4int baryon,
            G4int encoding,
            G4bool stable,
            G4double lifetime,
            G4DecayTable* decaytable);

  ~CMSAPrime() override = default;

public:
  static CMSAPrime* APrime(double apmass = 1000);
};

#endif
