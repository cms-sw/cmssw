#ifndef SimG4Core_Physics_G4Monopole_h
#define SimG4Core_Physics_G4Monopole_h 1

#include "G4ParticleDefinition.hh"
#include "globals.hh"
#include "G4SystemOfUnits.hh"

// ######################################################################
// ###                       Monopole                                 ###
// ######################################################################

class G4Monopole : public G4ParticleDefinition {

public: 
  
  G4Monopole (const G4String name="Monopole", G4int pdgEncoding= 0, 
	      G4double mass_=100.*GeV, G4int magCharge_=1, G4int elCharge_ =0);

  G4double MagneticCharge() const {return magCharge;};

private:

  ~G4Monopole() override;

private:

  G4double magCharge;
};

#endif
