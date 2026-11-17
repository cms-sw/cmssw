#include "SimG4Core/CustomPhysics/interface/Quirk.h"

Quirk::Quirk(const G4String& name, G4double mass, G4double charge, G4int pdg)
    : G4ParticleDefinition(name,
                           mass,
                           0.0,                 // width
                           charge,              //
                           1,                   // 2*spin
                           0,                   // parity
                           0,                   // C-conjugation
                           0,                   // 2*isospin
                           0,                   // 2*isospin3
                           0,                   // G-parity
                           "lepton",            // type
                           (pdg > 0) ? 1 : -1,  // lepton number
                           0,                   // baryon number
                           pdg,                 //
                           true,                // stable
                           -1.0,                // lifetime
                           nullptr,             // decay table
                           false,               // shortlived
                           "quirk") {}
