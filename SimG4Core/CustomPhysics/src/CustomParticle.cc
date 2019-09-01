#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"

// ######################################################################
// ###                           CustomParticle                       ###
// ######################################################################

CustomParticle::CustomParticle(const G4String& aName,
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
