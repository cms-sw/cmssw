#ifndef SimG4Core_Physics_Monopole_h
#define SimG4Core_Physics_Monopole_h 1

#include "G4ParticleDefinition.hh"
#include "globals.hh"
#include "CLHEP/Units/SystemOfUnits.h"

// ######################################################################
// ###                       Monopole                                 ###
// ######################################################################

class Monopole : public G4ParticleDefinition {

public: 
  
  Monopole (const G4String& name="Monopole", G4int pdgEncoding= 0, 
            G4double mass_=100.*CLHEP::GeV, G4int magCharge_=1, G4int elCharge_ =0);

  G4double MagneticCharge() const {return magCharge;};

private:

  ~Monopole() override;

  G4double magCharge;
};

#endif
