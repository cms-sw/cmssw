#include "SimG4Core/Physics/interface/PhysicsList.h"

PhysicsList::PhysicsList(G4LogicalVolumeToDDLogicalPartMap &,
			 const HepPDT::ParticleDataTable *,
			 sim::ChordFinderSetter *chordFinderSetter_,
			 const edm::ParameterSet &) {
}
 
PhysicsList::~PhysicsList() {
}

void PhysicsList::SetCuts() { 
}

