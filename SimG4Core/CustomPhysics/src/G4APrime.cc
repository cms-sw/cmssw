#include "SimG4Core/CustomPhysics/interface/G4APrime.h"
#include "G4SystemOfUnits.hh"

G4APrime* G4APrime::theAPrime = nullptr;

G4APrime::G4APrime(const G4String& aName,
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
                   G4DecayTable* decaytable)
    : G4ParticleDefinition(aName,
                           mass,
                           width,
                           charge,
                           iSpin,
                           iParity,
                           iConjugation,
                           iIsospin,
                           iIsospin3,
                           gParity,
                           pType,
                           lepton,
                           baryon,
                           encoding,
                           stable,
                           lifetime,
                           decaytable) {}

G4APrime::~G4APrime() {}

G4APrime* G4APrime::APrime(double apmass) {
  if (!theAPrime) {
    const G4String& name = "A^1";
    G4double mass = apmass * MeV;
    G4double width = 0.;
    G4double charge = 0;
    G4int iSpin = 0;
    G4int iParity = 0;
    G4int iConjugation = 0;
    G4int iIsospin = 0;
    G4int iIsospin3 = 0;
    G4int gParity = 0;
    const G4String& pType = "APrime";
    G4int lepton = 0;
    G4int baryon = 0;
    G4int encoding = 9994;
    G4bool stable = true;
    G4double lifetime = -1;
    G4DecayTable* decaytable = nullptr;

    theAPrime = new G4APrime(name,
                             mass,
                             width,
                             charge,
                             iSpin,
                             iParity,
                             iConjugation,
                             iIsospin,
                             iIsospin3,
                             gParity,
                             pType,
                             lepton,
                             baryon,
                             encoding,
                             stable,
                             lifetime,
                             decaytable);
  }
  return theAPrime;
}
