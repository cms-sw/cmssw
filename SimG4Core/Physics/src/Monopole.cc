#include "SimG4Core/Physics/interface/Monopole.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "G4ParticleTable.hh"
#include "CLHEP/Units/PhysicalConstants.h"

Monopole::Monopole(const G4String& aName, G4int pdgEncoding, G4double mass,
                   G4int mCharge, G4int eCharge) : G4ParticleDefinition(aName, 
                   mass,  0.0, CLHEP::eplus*eCharge, 0, 0, 0, 
		   0, 0, 0, "boson", 0, 0, pdgEncoding, true, -1.0, nullptr) {

  magCharge = CLHEP::eplus*G4double(mCharge)*0.5/CLHEP::fine_structure_const;

  edm::LogInfo("Monopole") << "Monopole is created: m(GeV)= " 
			     << GetPDGMass()/CLHEP::GeV 
			     << " Qel= " << GetPDGCharge()/CLHEP::eplus
			     << " Qmag= " << magCharge/CLHEP::eplus
			     << " PDG encoding = " << pdgEncoding;
}

Monopole::~Monopole() {}
