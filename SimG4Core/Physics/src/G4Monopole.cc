#include "SimG4Core/Physics/interface/G4Monopole.hh"
#include "G4ParticleTable.hh"

G4Monopole::G4Monopole(const G4String aName, G4int pdgEncoding, G4double mass,
		       G4int mCharge, G4int eCharge) : 
  G4ParticleDefinition(aName, mass,  0.0*MeV, eplus*eCharge, 0, 0, 0, 
		       0, 0, 0, "boson", 0, 0, pdgEncoding, true, -1.0, 0) {

  magCharge = eplus * G4double(mCharge) / fine_structure_const * 0.5;
  G4cout << "Monopole is created: m(GeV)= " << GetPDGMass()/GeV 
	 << " Qel= " << GetPDGCharge()/eplus
	 << " Qmag= " << magCharge/eplus
	 << " PDG encoding = " << pdgEncoding << G4endl;
}

G4Monopole::~G4Monopole() {}
