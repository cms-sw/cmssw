#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList(G4LogicalVolumeToDDLogicalPartMap & map,
			 const HepPDT::ParticleDataTable * table_,
			 sim::ChordFinderSetter *chordFinderSetter_,
			 const edm::ParameterSet & p) 
  : G4VModularPhysicsList(), m_pPhysics(p) {
}
 
PhysicsList::~PhysicsList() {
}

void PhysicsList::SetCuts() { 

  SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*CLHEP::cm);
  SetCutsWithDefault();
}

